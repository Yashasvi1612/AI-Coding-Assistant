"""
history.py
----------
Persists chat sessions to a local JSON file (sessions.json) so past
conversations survive app restarts and can be listed/switched between
in the History tab.
"""

import json
import os
import time
import uuid

HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions.json")


def load_sessions() -> dict:
    if not os.path.exists(HISTORY_FILE):
        return {}
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_sessions(sessions: dict) -> None:
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(sessions, f, indent=2)
    except Exception:
        pass  # non-fatal: worst case, history isn't saved this turn


def create_session(sessions: dict) -> str:
    sid = uuid.uuid4().hex[:8]
    sessions[sid] = {
        "title": "New chat",
        "messages": [],
        "created_at": time.strftime("%Y-%m-%d %H:%M"),
    }
    save_sessions(sessions)
    return sid


def update_session_messages(sessions: dict, sid: str, messages: list) -> None:
    if sid not in sessions:
        return
    sessions[sid]["messages"] = messages
    if sessions[sid]["title"] == "New chat" and messages:
        first_user = next((m["content"] for m in messages if m["role"] == "user"), "")
        first_user = first_user.strip().replace("\n", " ")
        sessions[sid]["title"] = (first_user[:40] + "…") if len(first_user) > 40 else (first_user or "New chat")
    save_sessions(sessions)


def delete_session(sessions: dict, sid: str) -> None:
    sessions.pop(sid, None)
    save_sessions(sessions)


def session_labels(sessions: dict):
    """
    Returns (labels, label_to_id) sorted by most recently created first.
    Labels are display strings shown in the History tab's radio list.
    """
    items = sorted(sessions.items(), key=lambda kv: kv[1].get("created_at", ""), reverse=True)
    labels = [f"{v['title']}  ·  {v['created_at']}" for _, v in items]
    label_to_id = {f"{v['title']}  ·  {v['created_at']}": k for k, v in items}
    return labels, label_to_id