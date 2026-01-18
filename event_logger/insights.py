import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple
from itertools import groupby
import warnings
warnings.filterwarnings('ignore')
from .processor import EventProcessor

class InsightExtractor:
    """
    Advanced analytics processor for Data Science and Process Mining insights.
    Answers specific analytical questions about event logs.
    """
    
    def __init__(self, event_processor: EventProcessor):
        self.ep = event_processor
        self.events = None
        self.sessions = None
        self.traces = None
        
    def load_and_structure(self, session_id: Optional[str] = None, user_id: Optional[str] = None):
        """Load events and structure them into sessions and traces (invocations)."""
        self.events = self.ep.load_all_events(session_id, user_id)
        
        # Group by session
        self.sessions = defaultdict(list)
        for event in self.events:
            sid = event.get("session_id")
            self.sessions[sid].append(event)
        
        # Group by trace (invocation_id within session)
        self.traces = defaultdict(list)
        for event in self.events:
            trace_key = f"{event.get('session_id')}_{event.get('invocation_id')}"
            self.traces[trace_key].append(event)
        
        print(f"📊 Loaded {len(self.events)} events, {len(self.sessions)} sessions, {len(self.traces)} traces")
    
    # ========== A. DATA SCIENCE QUESTIONS ==========
    
    def ds1_semantic_summary(self) -> Dict:
        """A1: Overall semantic summary of the event log."""
        if not self.events:
            return {"error": "No events loaded"}
        
        summary = {
            "total_events": len(self.events),
            "total_sessions": len(self.sessions),
            "total_traces": len(self.traces),
            "date_range": {
                "start": min(e.get("timestamp") for e in self.events if e.get("timestamp")),
                "end": max(e.get("timestamp") for e in self.events if e.get("timestamp"))
            },
            "unique_users": len(set(e.get("user_id") for e in self.events if e.get("user_id"))),
            "unique_agents": len(set(e.get("author") for e in self.events if e.get("author"))),
            "total_tool_calls": sum(1 for e in self.events if self.ep._has_content_type(e, "function_call")),
            "total_tokens": sum(e.get("token_usage", {}).get("total_tokens", 0) for e in self.events),
            "error_count": sum(1 for e in self.events if e.get("error_code")),
        }
        return summary
    
    def ds2_counts(self) -> Dict:
        """A2: Count of sessions, traces, events, and tool calls."""
        tool_calls = sum(1 for e in self.events if self.ep._has_content_type(e, "function_call"))
        return {
            "sessions": len(self.sessions),
            "traces": len(self.traces),
            "events": len(self.events),
            "tool_calls": tool_calls
        }
    
    def ds3_distribution_traces_events(self) -> Dict:
        """A3: Distributions of traces per session and events per trace."""
        traces_per_session = [len([t for t in self.traces.keys() if t.startswith(sid)]) 
                             for sid in self.sessions.keys()]
        events_per_trace = [len(events) for events in self.traces.values()]
        
        return {
            "traces_per_session": {
                "mean": np.mean(traces_per_session) if traces_per_session else 0,
                "median": np.median(traces_per_session) if traces_per_session else 0,
                "min": min(traces_per_session) if traces_per_session else 0,
                "max": max(traces_per_session) if traces_per_session else 0,
                "std": np.std(traces_per_session) if traces_per_session else 0
            },
            "events_per_trace": {
                "mean": np.mean(events_per_trace) if events_per_trace else 0,
                "median": np.median(events_per_trace) if events_per_trace else 0,
                "min": min(events_per_trace) if events_per_trace else 0,
                "max": max(events_per_trace) if events_per_trace else 0,
                "std": np.std(events_per_trace) if events_per_trace else 0
            }
        }
    
    def ds4_duration_distributions(self) -> Dict:
        """A4: Distribution of session and trace durations."""
        session_durations = []
        for sid, events in self.sessions.items():
            timestamps = [pd.to_datetime(e.get("timestamp")) for e in events if e.get("timestamp")]
            if len(timestamps) > 1:
                duration = (max(timestamps) - min(timestamps)).total_seconds()
                session_durations.append(duration)
        
        trace_durations = []
        for trace_events in self.traces.values():
            timestamps = [pd.to_datetime(e.get("timestamp")) for e in trace_events if e.get("timestamp")]
            if len(timestamps) > 1:
                duration = (max(timestamps) - min(timestamps)).total_seconds()
                trace_durations.append(duration)
        
        return {
            "session_duration_seconds": {
                "mean": np.mean(session_durations) if session_durations else 0,
                "median": np.median(session_durations) if session_durations else 0,
                "min": min(session_durations) if session_durations else 0,
                "max": max(session_durations) if session_durations else 0
            },
            "trace_duration_seconds": {
                "mean": np.mean(trace_durations) if trace_durations else 0,
                "median": np.median(trace_durations) if trace_durations else 0,
                "min": min(trace_durations) if trace_durations else 0,
                "max": max(trace_durations) if trace_durations else 0
            }
        }
    
    def ds5_tool_call_distribution(self) -> Dict:
        """A5: Distribution of tool calls per session and per trace."""
        tools_per_session = []
        for events in self.sessions.values():
            count = sum(1 for e in events if self.ep._has_content_type(e, "function_call"))
            tools_per_session.append(count)
        
        tools_per_trace = []
        for events in self.traces.values():
            count = sum(1 for e in events if self.ep._has_content_type(e, "function_call"))
            tools_per_trace.append(count)
        
        return {
            "tools_per_session": {
                "mean": np.mean(tools_per_session) if tools_per_session else 0,
                "median": np.median(tools_per_session) if tools_per_session else 0,
                "min": min(tools_per_session) if tools_per_session else 0,
                "max": max(tools_per_session) if tools_per_session else 0
            },
            "tools_per_trace": {
                "mean": np.mean(tools_per_trace) if tools_per_trace else 0,
                "median": np.median(tools_per_trace) if tools_per_trace else 0,
                "min": min(tools_per_trace) if tools_per_trace else 0,
                "max": max(tools_per_trace) if tools_per_trace else 0
            }
        }
    
    def ds6_tool_frequency(self) -> Dict:
        """A6: Frequency distribution of tools used."""
        tool_names = []
        for event in self.events:
            tool = self.ep._extract_function_name(event)
            if tool:
                tool_names.append(tool)
        
        freq = Counter(tool_names)
        return dict(freq.most_common())
    
    def ds7_token_distribution(self) -> Dict:
        """A7: Distribution of token usage by category and total."""
        prompt_tokens = [e.get("token_usage", {}).get("prompt_tokens", 0) for e in self.events if e.get("token_usage")]
        completion_tokens = [e.get("token_usage", {}).get("completion_tokens", 0) for e in self.events if e.get("token_usage")]
        thoughts_tokens = [e.get("token_usage", {}).get("thoughts_tokens", 0) for e in self.events if e.get("token_usage")]
        total_tokens = [e.get("token_usage", {}).get("total_tokens", 0) for e in self.events if e.get("token_usage")]
        
        return {
            "prompt_tokens": {"mean": np.mean(prompt_tokens), "total": sum(prompt_tokens)},
            "completion_tokens": {"mean": np.mean(completion_tokens), "total": sum(completion_tokens)},
            "thoughts_tokens": {"mean": np.mean(thoughts_tokens), "total": sum(thoughts_tokens)},
            "total_tokens": {"mean": np.mean(total_tokens), "total": sum(total_tokens)}
        }
    
    def ds8_token_consumption(self) -> Dict:
        """A8: Token consumption per session and per trace."""
        tokens_per_session = []
        for events in self.sessions.values():
            total = sum(e.get("token_usage", {}).get("total_tokens", 0) for e in events)
            tokens_per_session.append(total)
        
        tokens_per_trace = []
        for events in self.traces.values():
            total = sum(e.get("token_usage", {}).get("total_tokens", 0) for e in events)
            tokens_per_trace.append(total)
        
        return {
            "tokens_per_session": {
                "mean": np.mean(tokens_per_session) if tokens_per_session else 0,
                "median": np.median(tokens_per_session) if tokens_per_session else 0,
                "total": sum(tokens_per_session)
            },
            "tokens_per_trace": {
                "mean": np.mean(tokens_per_trace) if tokens_per_trace else 0,
                "median": np.median(tokens_per_trace) if tokens_per_trace else 0,
                "total": sum(tokens_per_trace)
            }
        }
    
    def ds9_success_failure_rate(self) -> Dict:
        """A9: Proportion of traces that complete successfully vs fail."""
        success = 0
        failure = 0
        
        for events in self.traces.values():
            # Check last event in trace
            if events:
                last_event = sorted(events, key=lambda e: e.get("timestamp", ""))[-1]
                if last_event.get("error_code") or last_event.get("interrupted"):
                    failure += 1
                elif last_event.get("finish_reason") == "STOP":
                    success += 1
                else:
                    # Ambiguous, count as success if no error
                    success += 1
        
        total = success + failure
        return {
            "successful_traces": success,
            "failed_traces": failure,
            "total_traces": total,
            "success_rate": success / total if total > 0 else 0,
            "failure_rate": failure / total if total > 0 else 0
        }
    
    def ds10_feature_correlation(self) -> pd.DataFrame:
        """A10: Features most correlated with failure or success."""
        trace_features = []
        
        for trace_key, events in self.traces.items():
            # Determine success/failure
            last_event = sorted(events, key=lambda e: e.get("timestamp", ""))[-1]
            failed = bool(last_event.get("error_code") or last_event.get("interrupted"))
            
            # Extract features
            total_tokens = sum(e.get("token_usage", {}).get("total_tokens", 0) for e in events)
            tool_count = sum(1 for e in events if self.ep._has_content_type(e, "function_call"))
            event_count = len(events)
            
            timestamps = [pd.to_datetime(e.get("timestamp")) for e in events if e.get("timestamp")]
            duration = (max(timestamps) - min(timestamps)).total_seconds() if len(timestamps) > 1 else 0
            
            trace_features.append({
                "trace_id": trace_key,
                "failed": failed,
                "total_tokens": total_tokens,
                "tool_calls": tool_count,
                "event_count": event_count,
                "duration_seconds": duration
            })
        
        df = pd.DataFrame(trace_features)
        
        # Calculate correlations
        if len(df) > 1:
            correlations = df[["failed", "total_tokens", "tool_calls", "event_count", "duration_seconds"]].corr()["failed"].sort_values(ascending=False)
            return correlations
        return pd.Series()
    
    # ========== B. PROCESS MINING QUESTIONS ==========
    
    def pm1_start_end_events(self) -> Dict:
        """B1: Valid start and end events."""
        start_activities = []
        end_activities = []
        
        for events in self.traces.values():
            sorted_events = sorted(events, key=lambda e: e.get("timestamp", ""))
            if sorted_events:
                start_activities.append(self.ep._determine_activity(sorted_events[0]))
                end_activities.append(self.ep._determine_activity(sorted_events[-1]))
        
        return {
            "start_events": dict(Counter(start_activities).most_common()),
            "end_events": dict(Counter(end_activities).most_common())
        }
    
    def pm2_process_model_sessions(self) -> Dict:
        """B2: Discovered process model when sessions are cases."""
        sequences = []
        for events in self.sessions.values():
            sorted_events = sorted(events, key=lambda e: e.get("timestamp", ""))
            sequence = [self.ep._determine_activity(e) for e in sorted_events]
            sequences.append(tuple(sequence))
        
        # Get unique process variants
        variants = Counter(sequences)
        return {
            "total_variants": len(variants),
            "top_10_variants": [
                {"sequence": list(seq), "count": count} 
                for seq, count in variants.most_common(10)
            ]
        }
    
    def pm3_process_model_traces(self) -> Dict:
        """B3: Discovered process model when traces are cases."""
        sequences = []
        for events in self.traces.values():
            sorted_events = sorted(events, key=lambda e: e.get("timestamp", ""))
            sequence = [self.ep._determine_activity(e) for e in sorted_events]
            sequences.append(tuple(sequence))
        
        variants = Counter(sequences)
        return {
            "total_variants": len(variants),
            "top_10_variants": [
                {"sequence": list(seq), "count": count} 
                for seq, count in variants.most_common(10)
            ]
        }
    
    def pm4_happy_paths(self) -> List[Dict]:
        """B4: Most frequent activity sequences (happy paths)."""
        sequences = []
        for events in self.traces.values():
            sorted_events = sorted(events, key=lambda e: e.get("timestamp", ""))
            sequence = tuple(self.ep._determine_activity(e) for e in sorted_events)
            sequences.append(sequence)
        
        freq = Counter(sequences)
        return [
            {"sequence": list(seq), "frequency": count, "percentage": count/len(sequences)*100}
            for seq, count in freq.most_common(5)
        ]
    
    def pm5_deviations(self) -> Dict:
        """B5: Where deviations from happy path occur."""
        # Get the most common path
        sequences = []
        for events in self.traces.values():
            sorted_events = sorted(events, key=lambda e: e.get("timestamp", ""))
            sequence = tuple(self.ep._determine_activity(e) for e in sorted_events)
            sequences.append(sequence)
        
        if not sequences:
            return {"error": "No sequences found"}
        
        happy_path = Counter(sequences).most_common(1)[0][0]
        
        # Find deviations
        deviations = []
        for seq in sequences:
            if seq != happy_path:
                # Find where it diverges
                for i, (act1, act2) in enumerate(zip(happy_path, seq)):
                    if act1 != act2:
                        deviations.append({
                            "position": i,
                            "expected": act1,
                            "actual": act2
                        })
                        break
        
        deviation_summary = Counter([(d["position"], d["expected"], d["actual"]) for d in deviations])
        
        return {
            "happy_path": list(happy_path),
            "deviation_count": len(deviations),
            "common_deviations": [
                {"position": pos, "expected": exp, "actual": act, "count": cnt}
                for (pos, exp, act), cnt in deviation_summary.most_common(5)
            ]
        }

    def pm6_loops_repeats(self) -> Dict:
        """B6: Loops or repeated behaviors within traces."""
        loop_patterns = []
        
        for events in self.traces.values():
            sorted_events = sorted(events, key=lambda e: e.get("timestamp", ""))
            activities = [self.ep._determine_activity(e) for e in sorted_events]
            
            # Find consecutive repeats
            for i in range(len(activities) - 1):
                if activities[i] == activities[i+1]:
                    loop_patterns.append(activities[i])
            
            # Find non-consecutive repeats (cycles)
            seen = {}
            for i, act in enumerate(activities):
                if act in seen:
                    loop_patterns.append(f"cycle:{act}")
                seen[act] = i
        
        loop_freq = Counter(loop_patterns)
        return {
            "total_loops": len(loop_patterns),
            "unique_loop_patterns": len(loop_freq),
            "most_common_loops": dict(loop_freq.most_common(10))
        }

    def pm7_bottlenecks(self) -> Dict:
        """B7: Activities or tools that are bottlenecks."""
        activity_durations = defaultdict(list)
        
        for events in self.traces.values():
            sorted_events = sorted(events, key=lambda e: e.get("timestamp", ""))
            
            for i in range(len(sorted_events) - 1):
                current_event = sorted_events[i]
                next_event = sorted_events[i+1]
                
                activity = self.ep._determine_activity(current_event)
                
                t1 = pd.to_datetime(current_event.get("timestamp"))
                t2 = pd.to_datetime(next_event.get("timestamp"))
                duration = (t2 - t1).total_seconds()
                
                activity_durations[activity].append(duration)
        
        # Calculate average duration per activity
        bottlenecks = {}
        for activity, durations in activity_durations.items():
            bottlenecks[activity] = {
                "mean_duration": np.mean(durations),
                "median_duration": np.median(durations),
                "max_duration": max(durations),
                "occurrences": len(durations)
            }
        
        # Sort by mean duration
        sorted_bottlenecks = sorted(
            bottlenecks.items(), 
            key=lambda x: x[1]["mean_duration"], 
            reverse=True
        )
        
        return {
            "top_10_bottlenecks": [
                {"activity": act, **stats} 
                for act, stats in sorted_bottlenecks[:10]
            ]
        }

    def pm8_success_vs_failure_paths(self) -> Dict:
        """B8: Process structure differences between successful and failed traces."""
        success_sequences = []
        failure_sequences = []
        
        for events in self.traces.values():
            sorted_events = sorted(events, key=lambda e: e.get("timestamp", ""))
            last_event = sorted_events[-1]
            
            sequence = tuple(self.ep._determine_activity(e) for e in sorted_events)
            
            if last_event.get("error_code") or last_event.get("interrupted"):
                failure_sequences.append(sequence)
            else:
                success_sequences.append(sequence)
        
        success_variants = Counter(success_sequences)
        failure_variants = Counter(failure_sequences)
        
        return {
            "successful_traces": {
                "count": len(success_sequences),
                "unique_variants": len(success_variants),
                "top_paths": [
                    {"sequence": list(seq), "count": cnt} 
                    for seq, cnt in success_variants.most_common(5)
                ]
            },
            "failed_traces": {
                "count": len(failure_sequences),
                "unique_variants": len(failure_variants),
                "top_paths": [
                    {"sequence": list(seq), "count": cnt} 
                    for seq, cnt in failure_variants.most_common(5)
                ]
            }
        }

    def pm9_no_tool_traces(self) -> Dict:
        """B9: Traces requiring no tool calls and how they differ."""
        with_tools = []
        without_tools = []
        
        for trace_key, events in self.traces.items():
            has_tools = any(self.ep._has_content_type(e, "function_call") for e in events)
            
            sorted_events = sorted(events, key=lambda e: e.get("timestamp", ""))
            sequence = tuple(self.ep._determine_activity(e) for e in sorted_events)
            
            if has_tools:
                with_tools.append(sequence)
            else:
                without_tools.append(sequence)
        
        total = len(with_tools) + len(without_tools)
        
        return {
            "traces_without_tools": len(without_tools),
            "traces_with_tools": len(with_tools),
            "proportion_without_tools": len(without_tools) / total if total > 0 else 0,
            "no_tool_variants": len(Counter(without_tools)),
            "with_tool_variants": len(Counter(with_tools)),
            "top_no_tool_paths": [
                {"sequence": list(seq), "count": cnt}
                for seq, cnt in Counter(without_tools).most_common(5)
            ]
        }

    def pm10_behavioral_clusters(self, n_clusters: int = 3) -> Dict:
        """B10: Cluster traces into behavioral archetypes."""
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.cluster import KMeans
        
        # Convert sequences to strings for vectorization
        trace_data = []
        for trace_key, events in self.traces.items():
            sorted_events = sorted(events, key=lambda e: e.get("timestamp", ""))
            sequence = " ".join([self.ep._determine_activity(e) for e in sorted_events])
            trace_data.append({
                "trace_id": trace_key,
                "sequence": sequence,
                "length": len(sorted_events),
                "has_tools": any(self.ep._has_content_type(e, "function_call") for e in events)
            })
        
        if len(trace_data) < n_clusters:
            return {"error": f"Not enough traces ({len(trace_data)}) for {n_clusters} clusters"}
        
        # Vectorize sequences
        sequences = [t["sequence"] for t in trace_data]
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(sequences)
        
        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        # Analyze clusters
        cluster_info = defaultdict(list)
        for i, trace in enumerate(trace_data):
            trace["cluster"] = int(clusters[i])
            cluster_info[int(clusters[i])].append(trace)
        
        cluster_summary = {}
        for cluster_id, traces in cluster_info.items():
            sequences = [t["sequence"].split() for t in traces]
            avg_length = np.mean([t["length"] for t in traces])
            tool_usage = np.mean([t["has_tools"] for t in traces])
            
            # Most common activities in this cluster
            all_activities = [act for seq in sequences for act in seq]
            common_activities = Counter(all_activities).most_common(5)
            
            cluster_summary[cluster_id] = {
                "trace_count": len(traces),
                "avg_sequence_length": avg_length,
                "tool_usage_rate": tool_usage,
                "top_activities": dict(common_activities),
                "example_sequences": [" -> ".join(seq[:5]) for seq in sequences[:3]]
            }
        
        return {
            "n_clusters": n_clusters,
            "total_traces": len(trace_data),
            "clusters": cluster_summary
        }


    # ========== COMPREHENSIVE REPORT GENERATOR ==========

    def generate_full_report(self, output_dir: str = "insights"):
        """Generate a comprehensive analysis report with all insights."""
        import os
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("🔍 Generating comprehensive insights report...")
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "data_science_insights": {},
            "process_mining_insights": {}
        }
        
        # Data Science Questions
        print("  → DS1: Semantic summary...")
        report["data_science_insights"]["ds1_semantic_summary"] = self.ds1_semantic_summary()
        
        print("  → DS2: Counts...")
        report["data_science_insights"]["ds2_counts"] = self.ds2_counts()
        
        print("  → DS3: Distribution traces/events...")
        report["data_science_insights"]["ds3_distributions"] = self.ds3_distribution_traces_events()
        
        print("  → DS4: Durations...")
        report["data_science_insights"]["ds4_durations"] = self.ds4_duration_distributions()
        
        print("  → DS5: Tool distribution...")
        report["data_science_insights"]["ds5_tool_distribution"] = self.ds5_tool_call_distribution()
        
        print("  → DS6: Tool frequency...")
        report["data_science_insights"]["ds6_tool_frequency"] = self.ds6_tool_frequency()
        
        print("  → DS7: Token distribution...")
        report["data_science_insights"]["ds7_token_distribution"] = self.ds7_token_distribution()
        
        print("  → DS8: Token consumption...")
        report["data_science_insights"]["ds8_token_consumption"] = self.ds8_token_consumption()
        
        print("  → DS9: Success/failure rate...")
        report["data_science_insights"]["ds9_success_failure"] = self.ds9_success_failure_rate()
        
        print("  → DS10: Feature correlation...")
        corr = self.ds10_feature_correlation()
        report["data_science_insights"]["ds10_correlations"] = corr.to_dict() if not corr.empty else {}
        
        # Process Mining Questions
        print("  → PM1: Start/end events...")
        report["process_mining_insights"]["pm1_start_end"] = self.pm1_start_end_events()
        
        print("  → PM2: Process model (sessions)...")
        report["process_mining_insights"]["pm2_model_sessions"] = self.pm2_process_model_sessions()
        
        print("  → PM3: Process model (traces)...")
        report["process_mining_insights"]["pm3_model_traces"] = self.pm3_process_model_traces()
        
        print("  → PM4: Happy paths...")
        report["process_mining_insights"]["pm4_happy_paths"] = self.pm4_happy_paths()
        
        print("  → PM5: Deviations...")
        report["process_mining_insights"]["pm5_deviations"] = self.pm5_deviations()
        
        print("  → PM6: Loops...")
        report["process_mining_insights"]["pm6_loops"] = self.pm6_loops_repeats()
        
        print("  → PM7: Bottlenecks...")
        report["process_mining_insights"]["pm7_bottlenecks"] = self.pm7_bottlenecks()
        
        print("  → PM8: Success vs failure paths...")
        report["process_mining_insights"]["pm8_success_failure_paths"] = self.pm8_success_vs_failure_paths()
        
        print("  → PM9: No-tool traces...")
        report["process_mining_insights"]["pm9_no_tool_traces"] = self.pm9_no_tool_traces()
        
        print("  → PM10: Behavioral clusters...")
        try:
            report["process_mining_insights"]["pm10_clusters"] = self.pm10_behavioral_clusters()
        except Exception as e:
            report["process_mining_insights"]["pm10_clusters"] = {"error": str(e)}
        
        # Save report
        report_file = output_path / "full_insights_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n✅ Full report saved to {report_file}")
        
        # Also save as readable text
        text_file = output_path / "insights_summary.txt"
        with open(text_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("EVENT LOG INSIGHTS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("DATA SCIENCE INSIGHTS\n")
            f.write("-" * 80 + "\n")
            for key, value in report["data_science_insights"].items():
                f.write(f"\n{key}:\n")
                f.write(json.dumps(value, indent=2, default=str) + "\n")
            
            f.write("\n\nPROCESS MINING INSIGHTS\n")
            f.write("-" * 80 + "\n")
            for key, value in report["process_mining_insights"].items():
                f.write(f"\n{key}:\n")
                f.write(json.dumps(value, indent=2, default=str) + "\n")
        
        print(f"✅ Human-readable summary saved to {text_file}")
        
        return report


    # Usage example
    if __name__ == "__main__":
        from collections import defaultdict
        import numpy as np
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.cluster import KMeans
        
        # Initialize processors
        ep = EventProcessor(log_dir=".log")
        ip = InsightProcessor(ep)
        
        # Load and analyze
        ip.load_and_structure()
        
        # Generate full report
        report = ip.generate_full_report(output_dir="insights")
        
        print("\n📊 Report generation complete!")