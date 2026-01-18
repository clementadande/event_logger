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

"""
Enhanced EventLogger with explicit start/end session markers
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


class EventLogger:
    """
    Enhanced event logger with explicit session lifecycle markers.
    Provides clear start/end events for better process mining.
    """
    
    def __init__(self, base_dir=".log"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._active_sessions = {}  # Track active sessions
    
    def _get_log_file_path(self, session_id):
        """Returns the path for the log file."""
        pattern = f"*_{session_id}.jsonl"
        existing_files = sorted(list(self.base_dir.glob(pattern)))
        
        if existing_files:
            return existing_files[-1]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{session_id}.jsonl"
        return self.base_dir / filename
    
    def log_session_start(
        self, 
        user_id: str, 
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict:
        """
        Log the start of a new session with optional metadata.
        
        Args:
            user_id: Unique user identifier
            session_id: Unique session identifier
            metadata: Optional metadata (e.g., user_agent, location, experiment_id)
        
        Returns:
            The created start event
        """
        start_event = {
            "event_type": "session_start",
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "session_id": session_id,
            "event_id": f"start_{session_id}",
            "metadata": metadata or {},
        }
        
        # Track session as active
        self._active_sessions[session_id] = {
            "user_id": user_id,
            "start_time": start_event["timestamp"],
            "metadata": metadata or {}
        }
        
        # Write to log
        path = self._get_log_file_path(session_id)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(start_event, ensure_ascii=False) + "\n")
        
        return start_event
    
    def log_session_end(
        self,
        user_id: str,
        session_id: str,
        reason: Optional[str] = None,
        summary: Optional[Dict[str, Any]] = None
    ) -> Dict:
        """
        Log the end of a session with optional reason and summary.
        
        Args:
            user_id: Unique user identifier
            session_id: Unique session identifier
            reason: Optional reason for session end (e.g., "user_quit", "timeout", "completed", "error")
            summary: Optional summary statistics (e.g., total_turns, total_tokens, user_satisfaction)
        
        Returns:
            The created end event
        """
        end_event = {
            "event_type": "session_end",
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "session_id": session_id,
            "event_id": f"end_{session_id}",
            "reason": reason or "normal",
            "summary": summary or {},
        }
        
        # Calculate session duration if we have start time
        if session_id in self._active_sessions:
            start_time = datetime.fromisoformat(self._active_sessions[session_id]["start_time"])
            end_time = datetime.fromisoformat(end_event["timestamp"])
            duration = (end_time - start_time).total_seconds()
            
            end_event["session_duration_seconds"] = duration
            end_event["start_metadata"] = self._active_sessions[session_id]["metadata"]
            
            # Remove from active sessions
            del self._active_sessions[session_id]
        
        # Write to log
        path = self._get_log_file_path(session_id)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(end_event, ensure_ascii=False) + "\n")
        
        return end_event
    
    def log(self, adk_event: Event, user_id: str, session_id: str):
        """
        Standard event logging (unchanged).
        Writes a dictionary event to a JSONL log file.
        """
        event = serialize_event_to_dict(adk_event)
        event["timestamp"] = event.get("timestamp") or datetime.utcnow().isoformat()
        event["user_id"] = user_id
        event["session_id"] = session_id
        
        # Mark as regular event if not already marked
        if "event_type" not in event:
            event["event_type"] = "agent_event"
        
        path = self._get_log_file_path(session_id)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    
    def get_active_sessions(self) -> Dict:
        """Get all currently active sessions."""
        return self._active_sessions.copy()
    
    def is_session_active(self, session_id: str) -> bool:
        """Check if a session is currently active."""
        return session_id in self._active_sessions
    
    def export_session(self, user_id, session_id, output_file):
        """
        Combine session logs into a single JSON response.
        Enhanced to include session lifecycle information.
        """
        pattern = f"*_{session_id}.jsonl"
        found_files = sorted(list(self.base_dir.glob(pattern)))
        
        if not found_files:
            print(f"❌ No logs found for Session {session_id}")
            return
        
        events = []
        session_start = None
        session_end = None
        
        for log_file in found_files:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if data.get("user_id") == user_id:
                            # Capture start/end events separately
                            if data.get("event_type") == "session_start":
                                session_start = data
                            elif data.get("event_type") == "session_end":
                                session_end = data
                            
                            events.append(data)
        
        # Sort by timestamp
        events.sort(key=lambda x: x.get("timestamp", ""))
        
        # Calculate statistics
        agent_events = [e for e in events if e.get("event_type") == "agent_event"]
        
        session_data = {
            "user_id": user_id,
            "session_id": session_id,
            "session_start": session_start,
            "session_end": session_end,
            "session_complete": session_start is not None and session_end is not None,
            "event_count": len(events),
            "agent_event_count": len(agent_events),
            "files_merged": [f.name for f in found_files],
            "events": events,
        }
        
        with open(output_file, "w", encoding="utf-8") as out:
            json.dump(session_data, out, indent=2, ensure_ascii=False)
        
        print(f"✅ Session exported to {output_file} ({len(found_files)} files merged)")


# ========== USAGE EXAMPLES ==========

def example_basic_usage():
    """Basic usage with start/end markers."""
    logger = EventLogger(base_dir=".log")
    
    user_id = "user_123"
    session_id = "session_abc"
    
    # 1. Start session with metadata
    logger.log_session_start(
        user_id=user_id,
        session_id=session_id,
        metadata={
            "user_agent": "Chrome/120.0",
            "location": "Paris, France",
            "experiment_id": "exp_001",
            "user_tier": "premium"
        }
    )
    
    # 2. Log normal events
    logger.log({"content": "User asked about weather"}, user_id, session_id)
    logger.log({"content": "Agent responded"}, user_id, session_id)
    
    # 3. End session with summary
    logger.log_session_end(
        user_id=user_id,
        session_id=session_id,
        reason="completed",
        summary={
            "total_turns": 5,
            "total_tokens": 1250,
            "user_satisfaction": 4.5,
            "tools_used": ["search", "calculator"]
        }
    )


def example_with_agent_loop():
    """Integration with agent event loop."""
    from google.adk.events import Event
    
    logger = EventLogger(base_dir=".log")
    user_id = "user_456"
    session_id = "session_xyz"
    
    # Start session
    logger.log_session_start(
        user_id=user_id,
        session_id=session_id,
        metadata={"source": "mobile_app", "version": "2.1.0"}
    )
    
    try:
        # Agent loop
        for adk_event in agent.stream():
            # Serialize and log ADK events
            from event_logger import serialize_event_to_dict
            event_dict = serialize_event_to_dict(adk_event)
            logger.log(event_dict, user_id, session_id)
            
            # Check for errors
            if adk_event.error_code:
                logger.log_session_end(
                    user_id=user_id,
                    session_id=session_id,
                    reason="error",
                    summary={"error_code": adk_event.error_code}
                )
                break
    
    except KeyboardInterrupt:
        # User interrupted
        logger.log_session_end(
            user_id=user_id,
            session_id=session_id,
            reason="user_interrupted"
        )
    
    except Exception as e:
        # Unexpected error
        logger.log_session_end(
            user_id=user_id,
            session_id=session_id,
            reason="system_error",
            summary={"error": str(e)}
        )
    
    else:
        # Normal completion
        logger.log_session_end(
            user_id=user_id,
            session_id=session_id,
            reason="completed"
        )


def example_context_manager():
    """Use as context manager for automatic start/end."""
    from contextlib import contextmanager
    
    @contextmanager
    def logged_session(logger, user_id, session_id, **metadata):
        """Context manager for automatic session lifecycle."""
        logger.log_session_start(user_id, session_id, metadata=metadata)
        try:
            yield logger
        except Exception as e:
            logger.log_session_end(
                user_id, session_id, 
                reason="error",
                summary={"error": str(e)}
            )
            raise
        else:
            logger.log_session_end(user_id, session_id, reason="completed")
    
    # Usage
    logger = EventLogger()
    with logged_session(logger, "user_789", "session_def", location="Tokyo") as log:
        log.log({"content": "Event 1"}, "user_789", "session_def")
        log.log({"content": "Event 2"}, "user_789", "session_def")


if __name__ == "__main__":
    # Run examples
    example_basic_usage()