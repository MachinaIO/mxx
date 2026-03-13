from __future__ import annotations

import json
from pathlib import Path


def _extract_role_message_from_record(record: dict[str, object], role: str) -> str | None:
    record_type = record.get("type")
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return None
    if record_type == "event_msg" and role == "user" and payload.get("type") == "user_message":
        message = payload.get("message")
        if isinstance(message, str) and message.strip():
            return message
    if record_type == "response_item" and payload.get("type") == "message" and payload.get("role") == role:
        content = payload.get("content")
        if not isinstance(content, list):
            return None
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text)
        if parts:
            return "\n".join(parts)
    return None


def initial_user_message_from_transcript(transcript_path: Path) -> str | None:
    try:
        lines = transcript_path.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, OSError):
        return None

    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        message = _extract_role_message_from_record(record, "user")
        if message:
            return message
    return None


def latest_user_message_from_transcript(transcript_path: Path) -> str | None:
    try:
        lines = transcript_path.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, OSError):
        return None

    latest: str | None = None
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        message = _extract_role_message_from_record(record, "user")
        if message:
            latest = message
    return latest


def latest_assistant_message_from_transcript(transcript_path: Path) -> str | None:
    try:
        lines = transcript_path.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, OSError):
        return None

    latest: str | None = None
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        message = _extract_role_message_from_record(record, "assistant")
        if message:
            latest = message
    return latest
