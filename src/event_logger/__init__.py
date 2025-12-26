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

    # ---- Extra Metadata (Safety, Grounding, Citations) ----
    # 1. Safety Ratings (pour comprendre les refus)
    safety_ratings = []
    if getattr(event, "safety_ratings", None):
        for rating in event.safety_ratings:
            safety_ratings.append({
                "category": getattr(rating, "category", "UNKNOWN"),
                "probability": getattr(rating, "probability", "UNKNOWN"),
                "blocked": getattr(rating, "blocked", False)
            })

    # 2. Grounding Metadata (Sources Web / Recherche)
    grounding_info = None
    if getattr(event, "grounding_metadata", None):
        # On convertit en dict ou str pour être sûr de le capturer
        gm = event.grounding_metadata
        grounding_info = {
            "web_search_queries": getattr(gm, "web_search_queries", []),
            "search_entry_point": str(getattr(gm, "search_entry_point", "")),
            # Les sources exactes peuvent être complexes, on tente une capture générique
            "retrieval_metadata": str(getattr(gm, "retrieval_metadata", "")) 
        }

    # 3. Citations
    citation_metadata = None
    if getattr(event, "citation_metadata", None):
        cm = event.citation_metadata
        citation_metadata = {
            "citations": [str(c) for c in getattr(cm, "citations", [])]
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
        "author": getattr(event, "author", None), # 'user' ou 'model'
        "is_final": event.is_final_response() if hasattr(event, "is_final_response") else False,
        "content": content_parts,
        "token_usage": token_usage,
        
        # Ajout des nouvelles métadonnées
        "safety_ratings": safety_ratings,
        "grounding_metadata": grounding_info,
        "citation_metadata": citation_metadata,

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

    def _get_log_file_path(self, session_id):
        """
        Returns the path for the log file.
        1. Checks if a file for this session already exists (to append to it).
        2. If not, creates a new filename with the current timestamp.
        Format: YYYYMMDD_HHMMSS_{session_id}.jsonl
        """
        # Look for ANY file ending with _{session_id}.jsonl
        pattern = f"*_{session_id}.jsonl"
        existing_files = sorted(list(self.base_dir.glob(pattern)))
        
        if existing_files:
            # Return the most recent existing file to keep appending
            return existing_files[-1]

        # Generate new filename for the start of the session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{session_id}.jsonl"
        return self.base_dir / filename

    def log(self, event: dict, user_id: str, session_id: str):
        """Writes a dictionary event to a JSONL log file."""

        # always add timestamp
        event["timestamp"] = event.get("timestamp") or datetime.utcnow().isoformat()

        # include identity
        event["user_id"] = user_id
        event["session_id"] = session_id

        # Determine file path
        path = self._get_log_file_path(session_id)
        
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def export_session(self, user_id, session_id, output_file):
        """
        Combine session logs into a single JSON response.
        """
        # Search pattern updated to match new structure (ignores the timestamp prefix)
        pattern = f"*_{session_id}.jsonl"
        found_files = sorted(list(self.base_dir.glob(pattern)))

        if not found_files:
            print(f"❌ No logs found for Session {session_id}")
            return

        events = []
        for log_file in found_files:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        # Optional safety: verify user_id inside the log if provided
                        if data.get("user_id") == user_id:
                            events.append(data)

        # Sort combined events by timestamp
        events.sort(key=lambda x: x.get("timestamp", ""))

        session_data = {
            "user_id": user_id,
            "session_id": session_id,
            "event_count": len(events),
            "files_merged": [f.name for f in found_files],
            "events": events,
        }

        with open(output_file, "w", encoding="utf-8") as out:
            json.dump(session_data, out, indent=2, ensure_ascii=False)

        print(f"✅ Session exported to {output_file} ({len(found_files)} source files merged)")