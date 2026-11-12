import argparse
import json
import sys
import time
from . import db
from .core import agent, client, initialize
from .config import settings

HELP = """
Команды:
  /help                 показать справку
  /new [название]       создать диалог
  /list                 список диалогов
  /use ID               переключить диалог
  /history              показать историю текущего диалога
  /tasks                список задач
  /pause ID             поставить задачу на паузу
  /resume ID            продолжить/повторить задачу
  /cancel ID            отменить задачу
  /status               проверить загруженные модели Ollama
  /unload               выгрузить основную модель из Ollama
  /quit                 выход
Обычный текст отправляется текущему диалогу как задача.
"""

def main():
    parser = argparse.ArgumentParser(description="LocalAgent terminal interface")
    parser.add_argument("--new", action="store_true", help="create a new conversation on startup")
    args = parser.parse_args()
    initialize()
    conversations = db.list_conversations()
    if args.new or not conversations:
        current = db.create_conversation()
    else:
        current = db.get_conversation(conversations[0]["id"])
    print(f"LocalAgent | модель: {settings.model} | контекст: {settings.num_ctx}")
    print(f"Текущий диалог: {current['title']} [{current['id']}]")
    print("Введите /help для справки.")
    while True:
        try:
            line = input(f"\n[{current['title']}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break
        if not line:
            continue
        if line == "/quit":
            break
        if line == "/help":
            print(HELP); continue
        if line == "/new" or line.startswith("/new "):
            title = line[5:].strip() or "Новый диалог"
            current = db.create_conversation(title)
            print(f"Создан: {current['title']} [{current['id']}]"); continue
        if line == "/list":
            for c in db.list_conversations():
                print(f"{c['id']} | {c['title']} | {c['updated_at']}")
            continue
        if line.startswith("/use "):
            item = db.get_conversation(line[5:].strip())
            if not item:
                print("Диалог не найден."); continue
            current = item
            print(f"Текущий диалог: {current['title']}"); continue
        if line == "/history":
            for m in db.get_messages(current["id"]):
                print(f"{m['role']}: {m['content']}")
            continue
        if line == "/tasks":
            for t in db.list_tasks():
                print(f"{t['id']} | {t['status']} | {t['user_message'][:70]} | {t.get('error') or ''}")
            continue
        if line.startswith(("/pause ", "/resume ", "/cancel ")):
            command, tid = line.split(maxsplit=1)
            try:
                task = {"pause": agent.pause, "resume": agent.resume, "cancel": agent.cancel}[command[1:]](tid)
                print(f"{task['id']}: {task['status']}")
            except Exception as exc:
                print(f"Ошибка: {exc}")
            continue
        if line == "/status":
            try:
                print(json.dumps(client.status(), ensure_ascii=False, indent=2))
            except Exception as exc:
                print(f"Ошибка: {exc}")
            continue
        if line == "/unload":
            try:
                client.unload()
                print("Запрос на выгрузку отправлен Ollama.")
            except Exception as exc:
                print(f"Ошибка: {exc}")
            continue
        try:
            task = agent.submit(current["id"], line)
            print(f"Задача создана: {task['id']}")
            while True:
                time.sleep(0.5)
                updated = db.get_task(task["id"])
                if updated and updated["status"] in {"completed", "failed", "cancelled", "paused", "interrupted"}:
                    if updated["status"] == "completed":
                        print(f"\n{updated['result']}")
                    else:
                        print(f"\nСтатус: {updated['status']} | {updated.get('error') or ''}")
                    break
        except Exception as exc:
            print(f"Ошибка: {exc}")

if __name__ == "__main__":
    main()
