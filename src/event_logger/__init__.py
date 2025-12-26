# event_logger.py
import json
from pathlib import Path
from datetime import datetime
from google.adk.events import Event

def serialize_event_to_dict(event: Event) -> dict:
    """Safely convert an ADK Event object into a JSON-serializable dict."""

    # ---- Content serialization ----
    content_parts = []
    if getattr(event, "content", None) and getattr(event.content, "parts", None):
        for p in event.content.parts:
            # Text response
            if hasattr(p, "text") and p.text is not None:
                content_parts.append({
                    "type": "text",
                    "value": p.text
                })

            # Function call
            elif hasattr(p, "function_call") and p.function_call is not None:
                fc = p.function_call
                content_parts.append({
                    "type": "function_call",
                    "name": fc.name,
                    "args": fc.args,
                    "id": getattr(fc, "id", None),
                })

            # Function response
            elif hasattr(p, "function_response") and p.function_response is not None:
                fr = p.function_response
                content_parts.append({
                    "type": "function_response",
                    "name": fr.name,
                    "response": fr.response,
                    "id": getattr(fr, "id", None),
                })

            else:
                # Capture anything not covered (important for future ADK versions)
                content_parts.append({"type": "raw_part", "value": str(p)})

    # ---- Token usage ----
    token_usage = None
    if getattr(event, "usage_metadata", None):
        um = event.usage_metadata
        token_usage = {
            "prompt_tokens": getattr(um, "prompt_token_count", None),
            "completion_tokens": getattr(um, "candidates_token_count", None),
            "total_tokens": getattr(um, "total_token_count", None),
        }

    # ---- Main event data ----
    serialized = {
        "timestamp": (
            datetime.fromtimestamp(event.timestamp).isoformat()
            if getattr(event, "timestamp", None)
            else None
        ),
        "event_id": getattr(event, "id", None),
        "invocation_id": getattr(event, "invocation_id", None),
        "author": getattr(event, "author", None),
        "is_final": event.is_final_response() if hasattr(event, "is_final_response") else False,
        "content": content_parts,
        "token_usage": token_usage,
        "finish_reason": (
            getattr(event.finish_reason, "name", None)
            if getattr(event, "finish_reason", None)
            else None
        ),
        "error_code": getattr(event, "error_code", None),
        "error_message": getattr(event, "error_message", None),
    }

    return serialized


class EventLogger:
    """Generic event logger for user/agent interactions."""
    
    def __init__(self, base_dir=".log"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _log_file(self, user_id, session_id):
        return self.base_dir / f"user_{user_id}_session_{session_id}.jsonl"

    def log(self, event: dict, user_id: str, session_id: str):
        """Writes a dictionary event to a JSONL log file."""

        # always add timestamp
        event["timestamp"] = event.get("timestamp") or datetime.utcnow().isoformat()

        # include identity
        event["user_id"] = user_id
        event["session_id"] = session_id

        path = self._log_file(user_id, session_id)
        with open(path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def export_session(self, user_id, session_id, output_file):
        """Combine the full session logs into a single JSON response."""
        path = self._log_file(user_id, session_id)

        if not path.exists():
            print(f"❌ No logs found: {path}")
            return

        events = []
        with open(path, "r") as f:
            for line in f:
                events.append(json.loads(line))

        session_data = {
            "user_id": user_id,
            "session_id": session_id,
            "event_count": len(events),
            "events": events,
        }

        with open(output_file, "w") as out:
            json.dump(session_data, out, indent=2)

        print(f"✅ Session exported to {output_file}")
