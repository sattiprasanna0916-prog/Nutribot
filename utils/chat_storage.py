
import json
import uuid
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_DIR = BASE_DIR / "database"
HISTORY_FILE = DATABASE_DIR / "chat_history.json"


def load_all():
    """Load all saved conversations."""
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    if not HISTORY_FILE.exists():
        return {}

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_all(conversations):
    """Save conversations safely to JSON."""
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    temp_file = HISTORY_FILE.with_suffix(".tmp")

    with open(temp_file, "w", encoding="utf-8") as file:
        json.dump(conversations, file, ensure_ascii=False, indent=2)

    temp_file.replace(HISTORY_FILE)


def create_conversation():
    """Create a new conversation."""
    conversations = load_all()

    conversation_id = str(uuid.uuid4())
    now = datetime.now().isoformat(timespec="seconds")

    conversations[conversation_id] = {
        "id": conversation_id,
        "title": "New Conversation",
        "created_at": now,
        "updated_at": now,
        "messages": []
    }

    save_all(conversations)
    return conversation_id


def update_conversation(conversation_id, messages):
    """Update messages and title for a conversation."""
    conversations = load_all()

    if conversation_id not in conversations:
        return False

    conversation = conversations[conversation_id]

    conversation["messages"] = messages
    conversation["updated_at"] = datetime.now().isoformat(
        timespec="seconds"
    )

    # Use the first user question as the conversation title.
    for message in messages:
        if message.get("role") == "user":
            title = message.get("content", "").strip()
            if title:
                conversation["title"] = (
                    title[:45] + "..."
                    if len(title) > 45 else title
                )
                break

    save_all(conversations)
    return True


def get_conversation(conversation_id):
    """Retrieve one conversation."""
    conversations = load_all()
    return conversations.get(conversation_id)


def list_conversations():
    """Return conversations sorted by most recently updated."""
    conversations = load_all()

    return sorted(
        conversations.values(),
        key=lambda item: item.get("updated_at", ""),
        reverse=True
    )


def delete_conversation(conversation_id):
    """Delete one conversation."""
    conversations = load_all()

    if conversation_id in conversations:
        del conversations[conversation_id]
        save_all(conversations)
        return True

    return False


def delete_all_conversations():
    """Delete all saved conversations."""
    save_all({})
    return True
