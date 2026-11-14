"""Safe storage and text extraction for user-supplied attachments."""
from __future__ import annotations
import base64
import binascii
import io
import json
import mimetypes
import uuid
import zipfile
import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Any
from .config import settings

MAX_ATTACHMENT_BYTES = 12 * 1024 * 1024
MAX_TOTAL_ATTACHMENT_BYTES = 100 * 1024 * 1024
MAX_VIDEO_BYTES = 100 * 1024 * 1024
MAX_VIDEO_DURATION_SECONDS = 60
MAX_VIDEO_FRAMES = 60
VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov", ".mkv", ".avi", ".webm", ".flv", ".wmv"}
VIDEO_MIME_TYPES = {"video/mp4", "video/quicktime", "video/x-matroska", "video/x-msvideo", "video/webm", "video/x-flv", "video/x-ms-wmv"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".flac", ".wma"}
AUDIO_MIME_TYPES = {"audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp4", "audio/aac", "audio/ogg", "audio/opus", "audio/flac", "audio/x-flac", "audio/x-ms-wma"}
MAX_AUDIO_SECONDS = 30
MAX_EXTRACTED_CHARS = 80000
MAX_MODEL_INPUT_TOKENS = 32000
MAX_MODEL_TEXT_CHARS = 80000
MAX_WEB_IMAGE_AUDIO_BYTES = 12 * 1024 * 1024
MAX_DOCUMENT_UPLOAD_BYTES = 50 * 1024 * 1024
ATTACHMENT_ROOT = settings.database_path.parent.resolve() / "attachments"
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}
TEXT_EXTENSIONS = {
    ".txt", ".md", ".rst", ".csv", ".tsv", ".json", ".jsonl", ".py",
    ".html", ".htm", ".xml", ".yaml", ".yml", ".log", ".ini", ".toml",
    ".cfg", ".conf", ".js", ".ts", ".jsx", ".tsx", ".css", ".sql",
    ".sh", ".bat", ".ps1", ".java", ".c", ".h", ".cpp", ".hpp", ".rs",
    ".go", ".php", ".rb", ".kt", ".swift", ".env", ".properties",
}


def _safe_name(name: str) -> str:
    name = Path(str(name or "attachment")).name
    name = "".join(ch for ch in name if ch.isalnum() or ch in " ._()-")[:120].strip()
    return name or "attachment"


def _validate_ooxml_bytes(raw: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = archive.infolist()
            if len(members) > 5000 or sum(item.file_size for item in members) > 200_000_000:
                raise ValueError("Office-вложение превышает лимит распакованного содержимого")
            if any(item.file_size > 100_000_000 for item in members):
                raise ValueError("Office-вложение содержит слишком большой внутренний файл")
    except zipfile.BadZipFile as exc:
        raise ValueError("Вложение не является корректным Office-документом") from exc


def _valid_image_signature(mime_type: str, raw: bytes) -> bool:
    if mime_type == "image/png": return raw.startswith(b"\x89PNG\r\n\x1a\n")
    if mime_type == "image/jpeg": return raw.startswith(b"\xff\xd8\xff")
    if mime_type == "image/gif": return raw.startswith((b"GIF87a", b"GIF89a"))
    if mime_type == "image/webp": return len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP"
    return False


def _extract_document(name: str, mime_type: str, raw: bytes, *, max_chars: int = MAX_EXTRACTED_CHARS) -> str:
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 1:
        raise ValueError("max_chars must be a positive integer")
    max_chars = min(max_chars, 1_000_000)
    ext = Path(name).suffix.lower()
    if ext in TEXT_EXTENSIONS or mime_type.startswith("text/"):
        return raw.decode("utf-8-sig", errors="replace")[:max_chars]
    if ext == ".pdf" or mime_type == "application/pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(raw))
            parts = []
            for index, page in enumerate(reader.pages[:80], 1):
                text = page.extract_text() or ""
                if text.strip(): parts.append(f"[Страница {index}]\n{text}")
                if sum(map(len, parts)) >= max_chars: break
            return "\n\n".join(parts)[:max_chars] or "[PDF: текст не извлечён; возможно, документ содержит только сканы.]"
        except Exception as exc:
            raise ValueError(f"Не удалось прочитать PDF: {type(exc).__name__}: {str(exc)[:200]}") from exc
    if ext == ".docx":
        try:
            _validate_ooxml_bytes(raw)
            from docx import Document
            doc = Document(io.BytesIO(raw))
            parts = [p.text for p in doc.paragraphs if p.text.strip()]
            for table in doc.tables[:30]:
                parts.append("\n[Таблица]\n" + "\n".join(" | ".join(c.text for c in row.cells) for row in table.rows[:200]))
            return "\n".join(parts)[:max_chars]
        except Exception as exc:
            raise ValueError(f"Не удалось прочитать DOCX: {type(exc).__name__}: {str(exc)[:200]}") from exc
    if ext == ".xlsx":
        try:
            _validate_ooxml_bytes(raw)
            from openpyxl import load_workbook
            wb = load_workbook(io.BytesIO(raw), read_only=True, data_only=False)
            parts = []
            try:
                for ws in wb.worksheets[:20]:
                    parts.append(f"[Лист: {ws.title}; размер: {ws.max_row}x{ws.max_column}]")
                    for row in ws.iter_rows(min_row=1, max_row=min(ws.max_row or 1, 12), max_col=min(ws.max_column or 1, 20), values_only=True):
                        parts.append(" | ".join("" if v is None else str(v)[:300] for v in row))
                    parts.append("[Показаны первые 12 строк листа; для полной обработки используй инструменты Excel агента.]\n")
                    if sum(map(len, parts)) >= max_chars: break
            finally: wb.close()
            return "\n".join(parts)[:max_chars]
        except Exception as exc:
            raise ValueError(f"Не удалось прочитать XLSX: {type(exc).__name__}: {str(exc)[:200]}") from exc
    if ext == ".doc":
        raise ValueError("Формат .doc не поддерживается; сохраните документ как .docx или PDF")
    if ext in {".xls", ".xlsm"}:
        raise ValueError("Поддерживается .xlsx; сохраните книгу в этом формате")
    raise ValueError(f"Тип вложения не поддерживается: {ext or mime_type}")


def _extract_document_visuals(name: str, raw: bytes, *, max_images: int = 12) -> tuple[list[dict[str, Any]], str | None]:
    """Render visual-only/scanned PDF pages and extract embedded Office images."""
    ext = Path(name).suffix.lower()
    if ext not in {".pdf", ".docx", ".xlsx"}:
        return [], None
    if ATTACHMENT_ROOT.is_symlink():
        raise ValueError("Managed attachment directory cannot be a symbolic link")
    visual_root = ATTACHMENT_ROOT / "document_visuals"
    if visual_root.is_symlink():
        raise ValueError("Document visual cache cannot be a symbolic link")
    visual_root.mkdir(parents=True, exist_ok=True)
    output: list[dict[str, Any]] = []
    warning = None

    if ext == ".pdf":
        try:
            import fitz
            document = fitz.open(stream=raw, filetype="pdf")
            try:
                for index, page in enumerate(document):
                    if index >= 80:
                        warning = "Визуальная обработка ограничена первыми 80 страницами PDF."
                        break
                    text = (page.get_text("text") or "").strip()
                    try:
                        has_images = bool(page.get_images(full=True))
                        has_graphics = bool(page.get_drawings())
                    except Exception:
                        has_images, has_graphics = False, False
                    if text and not has_images and not has_graphics:
                        continue
                    rect = page.rect
                    scale = min(320.0 / max(1.0, rect.width), 320.0 / max(1.0, rect.height))
                    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                    target = visual_root / (uuid.uuid4().hex + f"_page_{index + 1}.jpg")
                    pixmap.save(str(target))
                    output.append({
                        "kind": "image", "name": f"{Path(name).name} — page {index + 1}",
                        "path": str(target), "source": "managed_attachment",
                    })
                    if len(output) >= max_images:
                        warning = f"Визуальные страницы ограничены первыми {max_images} релевантными страницами."
                        break
            finally:
                document.close()
        except ImportError:
            warning = "PDF-текст извлечён, но визуальные страницы не обработаны: установи PyMuPDF."
        except Exception as exc:
            warning = f"Не удалось подготовить визуальные страницы PDF: {type(exc).__name__}: {str(exc)[:180]}"

    else:
        media_prefix = "word/media/" if ext == ".docx" else "xl/media/"
        try:
            from PIL import Image
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                for member in archive.infolist():
                    if not member.filename.startswith(media_prefix) or member.is_dir():
                        continue
                    if member.file_size <= 0 or member.file_size > MAX_WEB_IMAGE_AUDIO_BYTES:
                        continue
                    try:
                        with Image.open(io.BytesIO(archive.read(member))) as source_image:
                            source_image.thumbnail((320, 320))
                            image = source_image.convert("RGB")
                            target = visual_root / (uuid.uuid4().hex + ".jpg")
                            image.save(target, format="JPEG", quality=82, optimize=True)
                        output.append({
                            "kind": "image", "name": f"{Path(name).name} — {Path(member.filename).name}",
                            "path": str(target), "source": "managed_attachment",
                        })
                    except Exception:
                        continue
                    if len(output) >= max_images:
                        warning = f"Встроенные изображения ограничены первыми {max_images} объектами."
                        break
                if ext == ".xlsx" and any(member.filename.startswith("xl/charts/") for member in archive.infolist()):
                    chart_warning = "Встроенные диаграммы XLSX пока не рендерятся в изображения; используй terminal/LibreOffice для их визуализации."
                    warning = (warning + " " if warning else "") + chart_warning
        except ImportError:
            warning = "Текст документа извлечён, но встроенные изображения не обработаны: установи Pillow."
        except (zipfile.BadZipFile, OSError) as exc:
            warning = f"Не удалось извлечь встроенные изображения: {type(exc).__name__}."

    return output, warning



def extract_video_frames(path: str | Path, *, max_frames: int = MAX_VIDEO_FRAMES) -> dict[str, Any]:
    """Sample a video into an ordered, timestamped set of JPEG frames for Ollama.

    Ollama's chat API accepts images but not native video payloads. This bounded
    adapter samples frames across the whole clip and sends them as ordered images
    with timestamps in the accompanying text. The caller may separately extract a
    short audio segment when the source contains an audio stream.
    """
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Video file not found: {source}")
    size = source.stat().st_size
    if size <= 0 or size > MAX_VIDEO_BYTES:
        raise ValueError(f"Video must be non-empty and no larger than {MAX_VIDEO_BYTES // (1024 * 1024)} MB")
    if source.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video extension: {source.suffix or '(none)'}")
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("Для анализа видео установите ffmpeg: sudo apt install ffmpeg")
    probe = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(source)],
        capture_output=True, text=True, timeout=20, check=False,
    )
    if probe.returncode != 0:
        raise ValueError(f"Не удалось прочитать видео: {probe.stderr.strip()[:300]}")
    try:
        duration = float(probe.stdout.strip())
    except (TypeError, ValueError):
        raise ValueError("Не удалось определить длительность видео")
    if not (0.5 <= duration <= MAX_VIDEO_DURATION_SECONDS):
        raise ValueError(f"Длительность видео должна быть от 0.5 до {MAX_VIDEO_DURATION_SECONDS} секунд")
    if not isinstance(max_frames, int) or not 1 <= max_frames <= MAX_VIDEO_FRAMES:
        raise ValueError(f"max_frames must be between 1 and {MAX_VIDEO_FRAMES}")

    cache_key = hashlib.sha256(
        f"{source}:{size}:{source.stat().st_mtime_ns}:{max_frames}".encode("utf-8")
    ).hexdigest()
    if ATTACHMENT_ROOT.is_symlink():
        raise ValueError("Managed attachment directory cannot be a symbolic link")
    cache_root = ATTACHMENT_ROOT / "video_frames"
    if cache_root.is_symlink():
        raise ValueError("Video frame cache cannot be a symbolic link")
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_dir = cache_root / cache_key
    if cache_dir.is_symlink():
        raise ValueError("Video frame cache entry cannot be a symbolic link")
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(cache_dir.glob("frame_*.jpg"))
    if any(frame.is_symlink() for frame in frames):
        raise ValueError("Cached video frame cannot be a symbolic link")
    if not frames:
        fps = min(1.0, max_frames / duration)
        vf = f"fps={fps:.8f},scale=w=320:h=320:force_original_aspect_ratio=decrease,setsar=1"
        result = subprocess.run(
            [ffmpeg, "-nostdin", "-v", "error", "-i", str(source),
             "-vf", vf, "-frames:v", str(max_frames), "-q:v", "4",
             "-y", str(cache_dir / "frame_%02d.jpg")],
            capture_output=True, text=True, timeout=120, check=False,
        )
        if result.returncode != 0:
            for partial in cache_dir.glob("frame_*.jpg"):
                partial.unlink(missing_ok=True)
            raise ValueError(f"Не удалось извлечь кадры видео: {result.stderr.strip()[:400]}")
        frames = sorted(cache_dir.glob("frame_*.jpg"))
    if not frames:
        raise ValueError("Из видео не удалось извлечь ни одного кадра")
    fps = min(1.0, max_frames / duration)
    return {
        "duration_seconds": round(duration, 2),
        "frames": [
            {"path": str(frame), "timestamp_seconds": round(index / fps, 2)}
            for index, frame in enumerate(frames)
        ],
        "frame_count": len(frames),
        "audio_included": False,
        "note": "Видео нормализовано до последовательности JPEG-кадров не чаще 1 кадра/с, не более 60 секунд; звук обрабатывается отдельным аудиофрагментом.",
    }


def extract_video_audio(path: str | Path, *, max_seconds: float = MAX_AUDIO_SECONDS) -> dict[str, Any] | None:
    """Extract a short mono 16 kHz WAV segment for Gemma 4 audio understanding.

    Returns None when the clip has no audio stream. Ollama's current Gemma audio
    path accepts normalized WAV bytes through the multimodal images transport.
    """
    source = Path(path).resolve()
    if not source.is_file() or not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        return None
    probe = subprocess.run(
        [shutil.which("ffprobe"), "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(source)],
        capture_output=True, text=True, timeout=20, check=False,
    )
    if probe.returncode != 0 or not probe.stdout.strip():
        return None
    cache_root = ATTACHMENT_ROOT / "video_audio"
    if ATTACHMENT_ROOT.is_symlink() or cache_root.is_symlink():
        raise ValueError("Video audio cache cannot be a symbolic link")
    cache_root.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{source}:{source.stat().st_size}:{source.stat().st_mtime_ns}".encode()).hexdigest()
    target = cache_root / f"{key}.wav"
    if not target.is_file():
        result = subprocess.run(
            [shutil.which("ffmpeg"), "-nostdin", "-v", "error", "-y", "-i", str(source),
             "-map", "0:a:0", "-t", str(max_seconds), "-ac", "1", "-ar", "16000",
             "-c:a", "pcm_f32le", str(target)],
            capture_output=True, text=True, timeout=90, check=False,
        )
        if result.returncode != 0 or not target.is_file():
            target.unlink(missing_ok=True)
            return None
    if target.stat().st_size > 12 * 1024 * 1024:
        target.unlink(missing_ok=True)
        raise ValueError("Extracted video audio exceeds 12 MB")
    return {"path": str(target), "size_bytes": target.stat().st_size, "max_seconds": max_seconds}


def _persist_original_in_host_work(name: str, raw: bytes) -> str:
    """Persist the untouched upload in workspace/work, mounted as /host-workspace."""
    root = Path(getattr(settings, "workspace_path", settings.database_path.parent / "workspace")).resolve()
    work = root / "work"
    if work.is_symlink():
        raise ValueError("workspace/work не может быть символической ссылкой")
    work.mkdir(parents=True, exist_ok=True)
    safe_work = work.resolve()
    safe_work.relative_to(root)
    target = safe_work / (uuid.uuid4().hex[:12] + "_" + _safe_name(name))
    target.write_bytes(raw)
    return "/host-workspace/" + target.name


def save_uploads_to_host_workspace(items: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Save untouched browser uploads under workspace/work for /host-workspace access."""
    if not items:
        return []
    if not isinstance(items, list) or len(items) > 5:
        raise ValueError("Можно приложить не более 5 файлов за одно сообщение")
    root = Path(getattr(settings, "workspace_path", settings.database_path.parent / "workspace")).resolve()
    work = root / "work"
    if work.is_symlink():
        raise ValueError("workspace/work не может быть символической ссылкой")
    work.mkdir(parents=True, exist_ok=True)
    safe_work = work.resolve()
    safe_work.relative_to(root)
    total = 0
    saved: list[dict[str, Any]] = []
    try:
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("Каждое вложение должно быть объектом")
            name = _safe_name(item.get("name", "attachment"))
            mime = str(item.get("mime_type") or "application/octet-stream").lower()
            if mime == "application/octet-stream":
                mime = mimetypes.guess_type(name)[0] or mime
            data = item.get("data_base64")
            if not isinstance(data, str) or len(data) > MAX_VIDEO_BYTES * 2:
                raise ValueError(f"Некорректные данные вложения: {name}")
            try:
                raw = base64.b64decode(data, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError(f"Некорректный base64: {name}") from exc
            ext = Path(name).suffix.lower()
            is_video = ext in VIDEO_EXTENSIONS or mime in VIDEO_MIME_TYPES
            is_audio = ext in AUDIO_EXTENSIONS or mime in AUDIO_MIME_TYPES
            is_image = ext in {".png", ".jpg", ".jpeg", ".webp", ".gif"} or mime in ALLOWED_IMAGE_TYPES
            if is_video and ext not in VIDEO_EXTENSIONS:
                raise ValueError(f"Неподдерживаемое расширение видео: {name}")
            if is_audio and ext not in AUDIO_EXTENSIONS:
                raise ValueError(f"Неподдерживаемое расширение аудио: {name}")
            if is_image and ext not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
                raise ValueError(f"Неподдерживаемое расширение изображения: {name}")
            file_limit = MAX_VIDEO_BYTES if is_video else MAX_WEB_IMAGE_AUDIO_BYTES if (is_audio or is_image) else MAX_DOCUMENT_UPLOAD_BYTES
            if not raw or len(raw) > file_limit:
                raise ValueError(f"Файл {name} пустой или превышает {file_limit // (1024 * 1024)} МБ")
            total += len(raw)
            if total > MAX_TOTAL_ATTACHMENT_BYTES:
                raise ValueError(f"Общий размер вложений превышает {MAX_TOTAL_ATTACHMENT_BYTES // (1024 * 1024)} МБ")
            target = safe_work / (uuid.uuid4().hex[:12] + "_" + name)
            target.write_bytes(raw)
            saved.append({
                "name": name,
                "mime_type": mime,
                "size_bytes": len(raw),
                "container_path": "/host-workspace/" + target.name,
            })
    except Exception:
        for item in saved:
            (safe_work / Path(item["container_path"]).name).unlink(missing_ok=True)
        raise
    return saved


def save_attachments(items: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not items: return []
    if not isinstance(items, list) or len(items) > 5:
        raise ValueError("Можно приложить не более 5 файлов за одно сообщение")
    total = 0
    result = []
    if ATTACHMENT_ROOT.is_symlink():
        raise ValueError("Каталог хранения вложений не может быть символической ссылкой")
    ATTACHMENT_ROOT.mkdir(parents=True, exist_ok=True)
    safe_attachment_root = ATTACHMENT_ROOT.resolve()
    safe_attachment_root.relative_to(settings.database_path.parent.resolve())
    for item in items:
        if not isinstance(item, dict): raise ValueError("Каждое вложение должно быть объектом")
        name = _safe_name(item.get("name", "attachment"))
        mime = str(item.get("mime_type") or "application/octet-stream").lower()
        if mime == "application/octet-stream":
            mime = mimetypes.guess_type(name)[0] or mime
        data = item.get("data_base64")
        if not isinstance(data, str) or len(data) > MAX_VIDEO_BYTES * 2:
            raise ValueError(f"Некорректные данные вложения: {name}")
        try: raw = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError) as exc: raise ValueError(f"Некорректный base64: {name}") from exc
        ext = Path(name).suffix.lower()
        is_video_input = ext in VIDEO_EXTENSIONS or mime in VIDEO_MIME_TYPES
        file_limit = MAX_VIDEO_BYTES if is_video_input else MAX_WEB_IMAGE_AUDIO_BYTES if (ext in AUDIO_EXTENSIONS or mime in AUDIO_MIME_TYPES or ext in {'.png', '.jpg', '.jpeg', '.webp', '.gif'} or mime in ALLOWED_IMAGE_TYPES) else MAX_DOCUMENT_UPLOAD_BYTES
        if not raw or len(raw) > file_limit:
            raise ValueError(f"Файл {name} пустой или превышает {file_limit // (1024 * 1024)} МБ")
        total += len(raw)
        if total > MAX_TOTAL_ATTACHMENT_BYTES:
            raise ValueError(f"Общий размер вложений превышает {MAX_TOTAL_ATTACHMENT_BYTES // (1024 * 1024)} МБ")
        if ext in VIDEO_EXTENSIONS or mime in VIDEO_MIME_TYPES:
            if ext not in VIDEO_EXTENSIONS:
                raise ValueError(f"Неподдерживаемый формат видео: {name}")
            path = safe_attachment_root / (uuid.uuid4().hex + ext)
            path.write_bytes(raw)
            if not shutil.which("ffprobe"):
                path.unlink(missing_ok=True)
                raise ValueError("Для видео установите ffmpeg: sudo apt install ffmpeg")
            probe = subprocess.run(
                [shutil.which("ffprobe"), "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                capture_output=True, text=True, timeout=20, check=False,
            )
            if probe.returncode != 0:
                path.unlink(missing_ok=True)
                raise ValueError(f"Некорректный видеофайл: {probe.stderr.strip()[:200]}")
            try:
                duration = float(probe.stdout.strip())
            except (TypeError, ValueError):
                path.unlink(missing_ok=True)
                raise ValueError("Не удалось определить длительность видео")
            if not (0.5 <= duration <= MAX_VIDEO_DURATION_SECONDS):
                path.unlink(missing_ok=True)
                raise ValueError("Видео должно длиться от 0.5 до 60 секунд")
            host_workspace_path = _persist_original_in_host_work(name, raw)
            result.append({"name": name, "mime_type": mime, "kind": "video",
                           "path": str(path), "size_bytes": len(raw), "host_workspace_path": host_workspace_path})
        elif ext in AUDIO_EXTENSIONS or mime in AUDIO_MIME_TYPES:
            if ext not in AUDIO_EXTENSIONS:
                raise ValueError(f"Unsupported audio format: {name}")
            ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
            if not ffmpeg or not ffprobe:
                raise ValueError("Audio input requires ffmpeg and ffprobe (sudo apt install ffmpeg)")
            source_path = safe_attachment_root / (uuid.uuid4().hex + ext)
            wav_path = safe_attachment_root / (uuid.uuid4().hex + ".wav")
            source_path.write_bytes(raw)
            probe = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(source_path)], capture_output=True, text=True, timeout=20, check=False)
            try: duration = float(probe.stdout.strip())
            except (TypeError, ValueError):
                source_path.unlink(missing_ok=True)
                raise ValueError(f"Could not determine audio duration for {name}")
            if probe.returncode != 0 or not (0.1 <= duration <= MAX_AUDIO_SECONDS):
                source_path.unlink(missing_ok=True)
                raise ValueError(f"Audio must be between 0.1 and {MAX_AUDIO_SECONDS} seconds")
            converted = subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(source_path), "-t", str(MAX_AUDIO_SECONDS), "-ac", "1", "-ar", "16000", "-c:a", "pcm_f32le", str(wav_path)], capture_output=True, text=True, timeout=60, check=False)
            source_path.unlink(missing_ok=True)
            if converted.returncode != 0 or not wav_path.is_file():
                wav_path.unlink(missing_ok=True)
                raise ValueError(f"Audio conversion failed for {name}: {converted.stderr.strip()[:200]}")
            host_workspace_path = _persist_original_in_host_work(name, raw)
            result.append({"name": name, "mime_type": "audio/wav", "kind": "audio", "path": str(wav_path), "size_bytes": len(raw), "duration_seconds": round(duration, 2), "host_workspace_path": host_workspace_path})
        elif mime in ALLOWED_IMAGE_TYPES or ext in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            if mime not in ALLOWED_IMAGE_TYPES: raise ValueError(f"Неподдерживаемый формат изображения: {name}")
            if not _valid_image_signature(mime, raw): raise ValueError(f"Содержимое файла не соответствует типу изображения: {name}")
            # Keep the image under an opaque server-generated filename; never trust client paths.
            path = safe_attachment_root / (uuid.uuid4().hex + ext)
            path.write_bytes(raw)
            host_workspace_path = _persist_original_in_host_work(name, raw)
            result.append({"name": name, "mime_type": mime, "kind": "image", "path": str(path), "size_bytes": len(raw), "host_workspace_path": host_workspace_path})
        else:
            text = _extract_document(name, mime, raw)
            # Keep an original copy in a dedicated workspace subdirectory so
            # the container terminal and its Python libraries can inspect it.
            workspace_root = settings.workspace_path.resolve()
            document_dir = workspace_root / ".attachments"
            if document_dir.is_symlink():
                raise ValueError("workspace/.attachments не может быть символической ссылкой")
            document_dir.mkdir(parents=True, exist_ok=True)
            safe_document_dir = document_dir.resolve()
            safe_document_dir.relative_to(workspace_root)
            stored_path = safe_document_dir / (uuid.uuid4().hex + ext)
            stored_path.write_bytes(raw)
            host_workspace_path = _persist_original_in_host_work(name, raw)
            result.append({"name": name, "mime_type": mime, "kind": "document", "extracted_text": text,
                           "path": str(stored_path), "workspace_path": ".attachments/" + stored_path.name,
                           "size_bytes": len(raw), "host_workspace_path": host_workspace_path})
    return result


def model_message_content(content: str, attachments: list[dict[str, Any]] | None) -> str:
    parts = [str(content or "")]
    for attachment in attachments or []:
        host_path = attachment.get("host_workspace_path")
        path_line = f"\nОригинал файла сохранён в контейнере: {host_path}" if host_path else ""
        if attachment.get("kind") == "document":
            parts.append(f"\n\n--- Вложенный документ: {attachment.get('name','document')}{path_line} ---\n{attachment.get('extracted_text','')}")
        elif attachment.get("kind") == "video":
            parts.append(f"\n[Video attached: {attachment.get('name','video')}. Original: {host_path}. The model receives normalized sampled frames; audio is handled separately when available.]")
        elif attachment.get("kind") == "audio":
            parts.append(f"\n[Audio attached: {attachment.get('name','audio')}. Original: {host_path}. Audio input is normalized to mono 16 kHz float WAV.]")
        else:
            parts.append(f"\n[Прикреплено изображение: {attachment.get('name','image')}; оригинал: {host_path}]")
    return "\n".join(part for part in parts if part).strip()


def cleanup_conversation_attachments(conversation_id: str) -> dict[str, int]:
    """Remove only attachment files owned by a conversation from managed roots."""
    from . import db

    allowed_roots = [
        (settings.database_path.parent / "attachments").resolve(),
        (settings.workspace_path / ".attachments").resolve(),
    ]
    candidates: set[Path] = set()
    for attachment in db.get_conversation_attachments(conversation_id):
        raw_path = attachment.get("path")
        if isinstance(raw_path, str) and raw_path:
            candidates.add(Path(raw_path))
        legacy_path = attachment.get("workspace_path")
        if isinstance(legacy_path, str) and legacy_path.startswith(".attachments/"):
            candidates.add(settings.workspace_path / legacy_path)

    removed = skipped = 0
    for raw_path in candidates:
        try:
            if raw_path.is_symlink():
                skipped += 1
                continue
            resolved = raw_path.resolve(strict=False)
            if not any(resolved == root or root in resolved.parents for root in allowed_roots):
                skipped += 1
                continue
            if resolved.is_file():
                resolved.unlink()
                removed += 1
            else:
                skipped += 1
        except OSError:
            raise
    return {"removed": removed, "skipped": skipped}
