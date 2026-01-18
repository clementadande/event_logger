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
                # Capture anything not covered
                content_parts.append({"type": "raw_part", "value": str(p)})

    # ---- Token usage with details ----
    token_usage = None
    if getattr(event, "usage_metadata", None):
        um = event.usage_metadata
        token_usage = {
            "prompt_tokens": getattr(um, "prompt_token_count", None),
            "completion_tokens": getattr(um, "candidates_token_count", None),
            "total_tokens": getattr(um, "total_token_count", None),
            "thoughts_tokens": getattr(um, "thoughts_token_count", None),  # NEW: reasoning tokens
            "traffic_type": str(getattr(um, "traffic_type", None)),
        }
        
        # Token details by modality
        if hasattr(um, "prompt_tokens_details"):
            token_usage["prompt_details"] = [
                {"modality": str(d.modality), "count": d.token_count}
                for d in um.prompt_tokens_details
            ]
        if hasattr(um, "candidates_tokens_details"):
            token_usage["completion_details"] = [
                {"modality": str(d.modality), "count": d.token_count}
                for d in um.candidates_tokens_details
            ]

    # ---- Actions / State Changes (CRITICAL for agent behavior) ----
    actions_data = None
    if getattr(event, "actions", None):
        actions = event.actions
        actions_data = {
            "state_delta": getattr(actions, "state_delta", None),  # Agent state updates
            "artifact_delta": getattr(actions, "artifact_delta", None),
            "transfer_to_agent": getattr(actions, "transfer_to_agent", None),
            "escalate": getattr(actions, "escalate", None),
            "end_of_agent": getattr(actions, "end_of_agent", None),
            "skip_summarization": getattr(actions, "skip_summarization", None),
            "requested_auth_configs": getattr(actions, "requested_auth_configs", None),
            "requested_tool_confirmations": getattr(actions, "requested_tool_confirmations", None),
        }

    # ---- Safety Ratings ----
    safety_ratings = []
    if getattr(event, "safety_ratings", None):
        for rating in event.safety_ratings:
            safety_ratings.append({
                "category": str(getattr(rating, "category", "UNKNOWN")),
                "probability": str(getattr(rating, "probability", "UNKNOWN")),
                "blocked": getattr(rating, "blocked", False)
            })

    # ---- Grounding Metadata ----
    grounding_info = None
    if getattr(event, "grounding_metadata", None):
        gm = event.grounding_metadata
        grounding_info = {
            "web_search_queries": getattr(gm, "web_search_queries", []),
            "search_entry_point": str(getattr(gm, "search_entry_point", "")),
            "retrieval_metadata": str(getattr(gm, "retrieval_metadata", ""))
        }

    # ---- Citations ----
    citation_metadata = None
    if getattr(event, "citation_metadata", None):
        cm = event.citation_metadata
        citation_metadata = {
            "citations": [str(c) for c in getattr(cm, "citations", [])]
        }

    # ---- Model Info ----
    model_info = {
        "version": getattr(event, "model_version", None),
        "avg_logprobs": getattr(event, "avg_logprobs", None),
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
        "role": getattr(event.content, "role", None) if getattr(event, "content", None) else None,
        "is_final": event.is_final_response() if hasattr(event, "is_final_response") else False,
        
        # Core content
        "content": content_parts,
        
        # Metadata
        "model": model_info,
        "token_usage": token_usage,
        "actions": actions_data,  # NEW: Critical for understanding agent decisions
        "safety_ratings": safety_ratings if safety_ratings else None,
        "grounding_metadata": grounding_info,
        "citation_metadata": citation_metadata,
        
        # Status
        "finish_reason": (
            getattr(event.finish_reason, "name", None)
            if getattr(event, "finish_reason", None)
            else None
        ),
        "error_code": getattr(event, "error_code", None),
        "error_message": getattr(event, "error_message", None),
        "interrupted": getattr(event, "interrupted", None),
        
        # Tool execution tracking
        "long_running_tool_ids": (
            list(event.long_running_tool_ids) 
            if getattr(event, "long_running_tool_ids", None) 
            else None
        ),
    }

    # Remove None values to reduce size
    return {k: v for k, v in serialized.items() if v is not None}


class EventLogger:
    """Generic event logger for user/agent interactions."""
    
    def __init__(self, base_dir=".log"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_log_file_path(self, session_id):
        """
        Returns the path for the log file.
        Looks for existing file or creates new one with timestamp.
        Format: YYYYMMDD_HHMMSS_{session_id}.jsonl
        """
        pattern = f"*_{session_id}.jsonl"
        existing_files = sorted(list(self.base_dir.glob(pattern)))
        
        if existing_files:
            return existing_files[-1]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{session_id}.jsonl"
        return self.base_dir / filename

    def log(self, event: dict, user_id: str, session_id: str):
        """Writes a dictionary event to a JSONL log file."""
        event["timestamp"] = event.get("timestamp") or datetime.utcnow().isoformat()
        event["user_id"] = user_id
        event["session_id"] = session_id

        path = self._get_log_file_path(session_id)
        
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def export_session(self, user_id, session_id, output_file):
        """Combine session logs into a single JSON response."""
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
                        if data.get("user_id") == user_id:
                            events.append(data)

        events.sort(key=lambda x: x.get("timestamp", ""))

        # Calculate session statistics
        total_tokens = sum(
            e.get("token_usage", {}).get("total_tokens", 0) 
            for e in events 
            if e.get("token_usage")
        )
        tool_calls = sum(
            1 for e in events 
            for part in e.get("content", []) 
            if part.get("type") == "function_call"
        )

        session_data = {
            "user_id": user_id,
            "session_id": session_id,
            "event_count": len(events),
            "files_merged": [f.name for f in found_files],
            "statistics": {
                "total_tokens": total_tokens,
                "tool_calls": tool_calls,
            },
            "events": events,
        }

        with open(output_file, "w", encoding="utf-8") as out:
            json.dump(session_data, out, indent=2, ensure_ascii=False)

        print(f"✅ Session exported to {output_file} ({len(found_files)} source files merged)")