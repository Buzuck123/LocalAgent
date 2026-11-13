from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import attachments


def test_extract_video_frames_samples_video_and_caches(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    monkeypatch.setattr(attachments, "ATTACHMENT_ROOT", tmp_path / "attachments")
    monkeypatch.setattr(attachments.shutil, "which", lambda name: f"/usr/bin/{name}")

    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[0].endswith("ffprobe"):
            return SimpleNamespace(returncode=0, stdout="10.0", stderr="")
        pattern = Path(args[-1])
        for index in range(1, 5):
            (pattern.parent / f"frame_{index:02d}.jpg").write_bytes(b"jpeg")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(attachments.subprocess, "run", fake_run)

    result = attachments.extract_video_frames(video)
    assert result["frame_count"] == 4
    assert result["duration_seconds"] == 10.0
    assert result["audio_included"] is False
    assert [item["timestamp_seconds"] for item in result["frames"]] == [0.0, 0.83, 1.67, 2.5]
    assert len(calls) == 2

    second = attachments.extract_video_frames(video)
    assert second["frame_count"] == 4
    assert len(calls) == 3  # ffprobe runs, but cached frames avoid a second ffmpeg decode
    assert calls[-1][0].endswith("ffprobe")


def test_video_requires_ffmpeg(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    monkeypatch.setattr(attachments.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="sudo apt install ffmpeg"):
        attachments.extract_video_frames(video)


def test_video_extension_is_validated(tmp_path, monkeypatch):
    video = tmp_path / "clip.bin"
    video.write_bytes(b"fake-video")
    monkeypatch.setattr(attachments.shutil, "which", lambda name: "/usr/bin/tool")
    with pytest.raises(ValueError, match="Unsupported video extension"):
        attachments.extract_video_frames(video)


def test_uploaded_video_is_stored_as_video_attachment(tmp_path, monkeypatch):
    import base64

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(attachments, "settings", SimpleNamespace(database_path=data_dir / "agent.sqlite3"))
    monkeypatch.setattr(attachments, "ATTACHMENT_ROOT", data_dir / "attachments")
    monkeypatch.setattr(attachments.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        attachments.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="12.0", stderr=""),
    )
    item = {
        "name": "clip.mp4",
        "mime_type": "video/mp4",
        "data_base64": base64.b64encode(b"video bytes").decode("ascii"),
    }
    saved = attachments.save_attachments([item])
    assert saved[0]["kind"] == "video"
    assert Path(saved[0]["path"]).is_file()
    text = attachments.model_message_content("Посмотри клип", saved)
    assert "Video attached" in text
    assert "audio is handled separately" in text


def test_ollama_chat_sends_video_frames_as_ordered_images(tmp_path, monkeypatch):
    import base64
    from agent import core

    managed = tmp_path / "managed"
    managed.mkdir()
    video = managed / "clip.mp4"
    video.write_bytes(b"video")
    frame = managed / "video_frames" / "frame_01.jpg"
    frame.parent.mkdir()
    frame.write_bytes(b"jpeg-frame")
    monkeypatch.setattr(core, "ATTACHMENT_ROOT", managed)
    monkeypatch.setattr(attachments, "ATTACHMENT_ROOT", managed)
    monkeypatch.setattr(
        attachments, "extract_video_audio", lambda path, max_seconds=30.0: None,
    )
    monkeypatch.setattr(
        attachments, "extract_video_frames",
        lambda path, max_frames=12: {
            "duration_seconds": 4.0,
            "frames": [{"path": str(frame), "timestamp_seconds": 1.5}],
            "frame_count": 1,
            "audio_included": False,
            "note": "sampled",
        },
    )

    client = core.OllamaClient()
    captured = {}
    client._post = lambda endpoint, payload, timeout=None: (
        captured.update({"endpoint": endpoint, "payload": payload}) or
        {"message": {"role": "assistant", "content": "video understood"}}
    )
    response = client.chat([{
        "role": "user",
        "content": "Что происходит в видео?",
        "attachments": [{"kind": "video", "name": "clip.mp4", "path": str(video)}],
    }])
    model_message = captured["payload"]["messages"][0]
    assert response["message"]["content"] == "video understood"
    assert model_message["images"] == [base64.b64encode(b"jpeg-frame").decode("ascii")]
    assert "1.50s" in model_message["content"]
    assert "soundtrack is included only when the selected model supports Gemma 4 audio and an audio stream was extracted" in model_message["content"]
