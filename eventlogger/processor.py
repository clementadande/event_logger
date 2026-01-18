import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from itertools import groupby
import warnings
warnings.filterwarnings('ignore')

class EventProcessor:
    """
    Processes event logs into various analytical formats.
    Supports CSV export for data analysis and process mining.
    """
    
    def __init__(self, log_dir=".log"):
        self.log_dir = Path(log_dir)
        
    def load_events_from_jsonl(self, file_path: Path) -> List[Dict]:
        """Load all events from a single JSONL file."""
        events = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))
        return events
    
    def load_all_events(self, session_id: Optional[str] = None, 
                       user_id: Optional[str] = None) -> List[Dict]:
        """Load events from all JSONL files, optionally filtered by session/user."""
        pattern = f"*_{session_id}.jsonl" if session_id else "*.jsonl"
        all_events = []
        
        for log_file in self.log_dir.glob(pattern):
            events = self.load_events_from_jsonl(log_file)
            
            if user_id:
                events = [e for e in events if e.get("user_id") == user_id]
            
            all_events.extend(events)
        
        all_events.sort(key=lambda x: x.get("timestamp", ""))
        return all_events
    
    # ========== CORE TRANSFORMATIONS ==========
    
    def flatten_event_for_csv(self, event: Dict) -> Dict:
        """Flatten a complex event structure into a single-level dictionary."""
        flat = {
            "event_id": event.get("event_id"),
            "invocation_id": event.get("invocation_id"),
            "session_id": event.get("session_id"),
            "user_id": event.get("user_id"),
            "timestamp": event.get("timestamp"),
            "author": event.get("author"),
            "role": event.get("role"),
            "is_final": event.get("is_final", False),
            "model_version": event.get("model", {}).get("version"),
            "avg_logprobs": event.get("model", {}).get("avg_logprobs"),
            "content_types": self._extract_content_types(event),
            "text_content": self._extract_text_content(event),
            "has_function_call": self._has_content_type(event, "function_call"),
            "has_function_response": self._has_content_type(event, "function_response"),
            "function_name": self._extract_function_name(event),
            "prompt_tokens": event.get("token_usage", {}).get("prompt_tokens"),
            "completion_tokens": event.get("token_usage", {}).get("completion_tokens"),
            "total_tokens": event.get("token_usage", {}).get("total_tokens"),
            "thoughts_tokens": event.get("token_usage", {}).get("thoughts_tokens"),
            "traffic_type": event.get("token_usage", {}).get("traffic_type"),
            "state_changes": json.dumps(event.get("actions", {}).get("state_delta")) if event.get("actions", {}).get("state_delta") else None,
            "has_transfer": event.get("actions", {}).get("transfer_to_agent") is not None,
            "has_escalation": event.get("actions", {}).get("escalate") is not None,
            "finish_reason": event.get("finish_reason"),
            "error_code": event.get("error_code"),
            "error_message": event.get("error_message"),
            "interrupted": event.get("interrupted"),
            "has_safety_issues": len(event.get("safety_ratings", [])) > 0,
            "safety_blocked": any(r.get("blocked", False) for r in event.get("safety_ratings", [])),
            "active_tools": len(event.get("long_running_tool_ids", [])),
        }
        return flat
    
    def _extract_content_types(self, event: Dict) -> str:
        content = event.get("content", [])
        types = [part.get("type") for part in content if part.get("type")]
        return ",".join(types) if types else None
    
    def _extract_text_content(self, event: Dict) -> str:
        content = event.get("content", [])
        texts = [part.get("value") for part in content if part.get("type") == "text" and part.get("value")]
        full_text = " ".join(texts)
        return full_text[:500] + "..." if len(full_text) > 500 else full_text
    
    def _has_content_type(self, event: Dict, content_type: str) -> bool:
        content = event.get("content", [])
        return any(part.get("type") == content_type for part in content)
    
    def _extract_function_name(self, event: Dict) -> Optional[str]:
        content = event.get("content", [])
        for part in content:
            if part.get("type") in ["function_call", "function_response"]:
                return part.get("name")
        return None
    
    def to_csv(self, output_file: str, session_id: Optional[str] = None, user_id: Optional[str] = None):
        """Export events to CSV format for analysis."""
        events = self.load_all_events(session_id, user_id)
        if not events:
            print(f"❌ No events found")
            return
        
        flattened = [self.flatten_event_for_csv(e) for e in events]
        df = pd.DataFrame(flattened)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.to_csv(output_file, index=False)
        print(f"✅ Exported {len(df)} events to {output_file}")
        return df
    
    def to_process_mining_csv(self, output_file: str, session_id: Optional[str] = None, user_id: Optional[str] = None):
        """Export in XES-compatible CSV format for process mining tools."""
        events = self.load_all_events(session_id, user_id)
        if not events:
            print(f"❌ No events found")
            return
        
        pm_events = []
        for event in events:
            activity = self._determine_activity(event)
            pm_event = {
                "case_id": event.get("session_id"),
                "activity": activity,
                "timestamp": event.get("timestamp"),
                "resource": event.get("author"),
                "lifecycle": self._determine_lifecycle(event),
                "invocation_id": event.get("invocation_id"),
                "user_id": event.get("user_id"),
                "model_version": event.get("model", {}).get("version"),
                "total_tokens": event.get("token_usage", {}).get("total_tokens"),
                "finish_reason": event.get("finish_reason"),
                "has_error": event.get("error_code") is not None,
            }
            pm_events.append(pm_event)
        
        df = pd.DataFrame(pm_events)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['case_id', 'timestamp'])
        df.to_csv(output_file, index=False)
        print(f"✅ Exported {len(df)} process mining events to {output_file}")
        return df
    
    def _determine_activity(self, event: Dict) -> str:
        """Determine the activity name for process mining."""
        content = event.get("content", [])
        author = event.get("author", "unknown")
        
        for part in content:
            if part.get("type") == "function_call":
                return f"tool_{part.get('name')}_call"
            elif part.get("type") == "function_response":
                return f"tool_{part.get('name')}_response"
        
        if any(p.get("type") == "text" for p in content):
            if author == "TestAgent" or event.get("role") == "model":
                return "agent_response"
            elif event.get("role") == "user":
                return "user_message"
        
        if event.get("actions", {}).get("state_delta"):
            return "state_update"
        
        return "unknown_activity"
    
    def _determine_lifecycle(self, event: Dict) -> str:
        """Determine lifecycle state for process mining."""
        if event.get("is_final"):
            return "complete"
        elif event.get("finish_reason") == "STOP":
            return "complete"
        elif event.get("error_code"):
            return "failed"
        elif event.get("interrupted"):
            return "cancelled"
        else:
            return "start"
    
    # ========== EXISTING EXPORTS ==========
    
    def generate_conversation_flow(self, session_id: str) -> pd.DataFrame:
        """Generate conversation flow showing sequence of messages with timing."""
        events = self.load_all_events(session_id=session_id)
        flow = []
        for i, event in enumerate(events):
            if event.get("role") in ["user", "model"]:
                text = self._extract_text_content(event)
                flow_item = {
                    "turn": i + 1,
                    "timestamp": event.get("timestamp"),
                    "speaker": "User" if event.get("role") == "user" else "Agent",
                    "message": text,
                    "tokens": event.get("token_usage", {}).get("total_tokens"),
                    "invocation_id": event.get("invocation_id"),
                }
                if flow and flow[-1].get("timestamp"):
                    prev_time = pd.to_datetime(flow[-1]["timestamp"])
                    curr_time = pd.to_datetime(event.get("timestamp"))
                    flow_item["time_since_last"] = (curr_time - prev_time).total_seconds()
                flow.append(flow_item)
        return pd.DataFrame(flow)
    
    def generate_tool_usage_summary(self, session_id: Optional[str] = None) -> pd.DataFrame:
        """Generate summary statistics for tool usage."""
        events = self.load_all_events(session_id=session_id)
        tool_calls = []
        for event in events:
            content = event.get("content", [])
            for part in content:
                if part.get("type") == "function_call":
                    tool_calls.append({
                        "session_id": event.get("session_id"),
                        "timestamp": event.get("timestamp"),
                        "tool_name": part.get("name"),
                        "invocation_id": event.get("invocation_id"),
                        "args": json.dumps(part.get("args", {})),
                    })
        
        if not tool_calls:
            return pd.DataFrame()
        
        df = pd.DataFrame(tool_calls)
        summary = df.groupby('tool_name').agg({
            'tool_name': 'count',
            'session_id': 'nunique'
        }).rename(columns={'tool_name': 'call_count', 'session_id': 'unique_sessions'})
        return summary
    
    def export_session_statistics(self, output_file: str, session_id: Optional[str] = None):
        """Export aggregated session statistics."""
        events = self.load_all_events(session_id=session_id)
        sessions = {}
        for event in events:
            sid = event.get("session_id")
            if sid not in sessions:
                sessions[sid] = {
                    "session_id": sid,
                    "user_id": event.get("user_id"),
                    "start_time": event.get("timestamp"),
                    "end_time": event.get("timestamp"),
                    "event_count": 0,
                    "total_tokens": 0,
                    "tool_calls": 0,
                    "errors": 0,
                    "user_messages": 0,
                    "agent_responses": 0,
                }
            
            s = sessions[sid]
            s["event_count"] += 1
            s["end_time"] = event.get("timestamp")
            s["total_tokens"] += event.get("token_usage", {}).get("total_tokens", 0)
            
            if self._has_content_type(event, "function_call"):
                s["tool_calls"] += 1
            if event.get("error_code"):
                s["errors"] += 1
            if event.get("role") == "user":
                s["user_messages"] += 1
            elif event.get("role") == "model":
                s["agent_responses"] += 1
        
        df = pd.DataFrame(list(sessions.values()))
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])
        df['duration_seconds'] = (df['end_time'] - df['start_time']).dt.total_seconds()
        df.to_csv(output_file, index=False)
        print(f"✅ Exported statistics for {len(df)} sessions to {output_file}")
        return df

if __name__ == "__main__":
    processor = EventProcessor(log_dir=".log")
    
    # Export all events to CSV
    processor.to_csv("events_analysis.csv")
    
    # Export for process mining
    processor.to_process_mining_csv("events_process_mining.csv")
    
    # Export session statistics
    processor.export_session_statistics("session_statistics.csv")
    
    # Generate tool usage summary
    tool_summary = processor.generate_tool_usage_summary()
    if not tool_summary.empty:
        tool_summary.to_csv("tool_usage_summary.csv")
        print("✅ Tool usage summary exported")