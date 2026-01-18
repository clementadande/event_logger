# Event Logger for Google Gen AI Agents

A comprehensive Python toolkit for tracking, analyzing, and mining interaction data from Google Gen AI Agents (Gemini/Vertex AI). Goes beyond simple logging to provide **data science insights** and **process mining capabilities**.

It captures not just text content, but also critical metadata: **function calls/responses**, **token usage**, **safety ratings**, **grounding metadata** (Google Search sources), **citations**, and **agent state changes**.

## Features

### 📝 Event Logging
- **Comprehensive serialization**: Converts complex ADK `Event` objects into clean, serializable JSON dictionaries
- **Chronological sorting**: Log files automatically prefixed with timestamps (`YYYYMMDD_HHMMSS_{session_id}.jsonl`)
- **Session management**: Automatically creates new files or appends to existing sessions
- **Rich metadata support**:
  - 🛡️ **Safety Ratings**: Logs probability and categories of blocked content
  - ⚓ **Grounding**: Captures web search queries and entry points
  - 🧩 **Function Calls**: Logs arguments and responses for tool use
  - 🪙 **Token Usage**: Tracks prompt, completion, and thought tokens
  - 🔄 **State Changes**: Captures agent state deltas and actions
- **Session export**: Merge fragmented JSONL logs into single JSON reports

### 📊 Data Analytics (InsightExtractor)
Answers 20+ analytical questions about your agent logs:

|#|Data Science Insights|Process Mining Insights|
|---|-----------------------|-------------------------|
|1.|Semantic summary of event logs|Start and end event identification|
|2.|Session, trace, event, and tool call counts|Process model discovery (session-level)|
|3.|Distribution statistics (traces/events per session)|Process model discovery (trace-level)|
|4.|Duration distributions (sessions and traces)|Happy path identification|
|5.|Tool usage patterns and distributions|Deviation analysis|
|6.|Tool frequency analysis|Loop and repeat detection|
|7.|Token usage by category (prompt/completion/thoughts)|Bottleneck identification|
|8.|Token consumption patterns|Success vs failure path comparison|
|9.|Success vs failure rates|No-tool trace analysis|
|10.|Feature correlations with outcomes|Behavioral clustering (ML-based)|

### 🔄 Export Formats
- **JSONL**: Raw event logs with full metadata
- **CSV**: Flattened format for Excel/pandas analysis
- **Process Mining CSV**: XES-compatible format for ProM, Disco, Celonis, pm4py
- **JSON Reports**: Comprehensive session and insight reports

## Installation

### Option 1: Standard `pip`

Install directly from source:

```bash
pip install .
```

Or from GitHub:

```bash
pip install git+https://github.com/clementadande/eventlogger.git
```

### Option 2: Using `uv` (Fast Python Installer)

Add to your project:

```bash
uv add git+https://github.com/clementadande/eventlogger.git
```

Or install directly:

```bash
uv pip install git+https://github.com/clementadande/eventlogger.git
```

### Optional Features

Install additional dependencies for specific features:

```bash
# Analytics features (statistics, clustering)
pip install "eventlogger[analytics]"

# Rich terminal output
pip install "eventlogger[rich-output]"

# Full feature set (analytics + rich output)
pip install "eventlogger[full]"

# Visualization tools
pip install "eventlogger[viz]"

# Advanced process mining with pm4py
pip install "eventlogger[process-mining]"

# Everything
pip install "eventlogger[all]"
```

## Usage

### 1. Basic Event Logging

Capture agent interactions as they happen:

```python
from eventlogger import EventLogger, serialize_event_to_dict

# Initialize logger (defaults to .log/ directory)
logger = EventLogger(base_dir=".log")

user_id = "user_123"
session_id = "session_abc"

# Inside your agent loop where you receive an 'event' object
# event is a google.adk.events.Event object

# 1. Serialize the ADK event to a dictionary
event_dict = serialize_event_to_dict(adk_event)

# 2. Log it to disk
logger.log(event_dict, user_id, session_id)
```

Creates files like: `.log/20250118_143005_session_abc.jsonl`

### 2. Export Session Data

Merge all logs for a specific session:

```python
logger.export_session(
    user_id="user_123", 
    session_id="session_abc", 
    output_file="session_report.json"
)
```

### 3. Data Analysis & Insights

Transform logs into analytical datasets:

```python
from eventlogger import EventProcessor, InsightExtractor

# Initialize processors
ep = EventProcessor(log_dir=".log")
ip = InsightExtractor(ep)

# Load and structure data
ip.load_and_structure()

# Export to CSV for analysis
ep.to_csv("events_analysis.csv")

# Export for process mining tools (ProM, Disco, pm4py)
ep.to_process_mining_csv("events_process_mining.csv")

# Generate session statistics
ep.export_session_statistics("session_stats.csv")
```

### 4. Generate Comprehensive Insights

Answer all 20 analytical questions automatically:

```python
# Generate full report (20+ insights)
report = ip.generate_full_report(output_dir="insights")

# Outputs:
# - insights/full_insights_report.json (structured data)
# - insights/insights_summary.txt (human-readable)

# Or query specific insights
semantic_summary = ip.ds1_semantic_summary()
tool_frequency = ip.ds6_tool_frequency()
happy_paths = ip.pm4_happy_paths()
bottlenecks = ip.pm7_bottlenecks()
success_rate = ip.ds9_success_failure_rate()
clusters = ip.pm10_behavioral_clusters(n_clusters=3)
```

### 5. Advanced Analytics

```python
# Get conversation flow with timing
flow = ep.generate_conversation_flow(session_id="session_abc")

# Tool usage summary
tool_summary = ep.generate_tool_usage_summary()

# Feature correlation with failures
correlations = ip.ds10_feature_correlation()

# Compare successful vs failed paths
path_comparison = ip.pm8_success_vs_failure_paths()
```

## Output Formats

### Log File Naming

Files use the format: `YYYYMMDD_HHMMSS_{session_id}.jsonl`

- **YYYYMMDD_HHMMSS**: Timestamp of first event in session
- **session_id**: Unique identifier for the session

### JSONL Structure

Each line contains detailed event information:

```json
{
  "timestamp": "2026-01-18T10:00:00.123456",
  "event_id": "abc-123",
  "invocation_id": "e-xyz-789",
  "session_id": "session_abc",
  "user_id": "user_123",
  "author": "TestAgent",
  "role": "model",
  "is_final": true,
  "content": [
    { "type": "text", "value": "The current time in Tokyo is 08:04:44." },
    { "type": "function_call", "name": "get_current_time", "args": {"city": "Tokyo", "gmt": 9}, "id": "adk-f67a" }
  ],
  "model": {
    "version": "gemini-2.5-flash",
    "avg_logprobs": -0.037
  },
  "token_usage": {
    "prompt_tokens": 123,
    "completion_tokens": 10,
    "total_tokens": 133,
    "thoughts_tokens": 67
  },
  "actions": {
    "state_delta": {
      "agent_location": "La Rochelle, France",
      "premium_member_instruction": "Premium members need concise answers."
    }
  },
  "safety_ratings": [
    { "category": "HARM_CATEGORY_HATE_SPEECH", "probability": "NEGLIGIBLE", "blocked": false }
  ],
  "grounding_metadata": {
    "web_search_queries": ["current time Tokyo"],
    "search_entry_point": "..."
  },
  "finish_reason": "STOP"
}
```

### CSV Exports

**Standard CSV** (`to_csv()`):
- Flattened event structure
- One row per event
- All key metrics included
- Ready for pandas/Excel

**Process Mining CSV** (`to_process_mining_csv()`):
- XES-compatible format
- Columns: `case_id`, `activity`, `timestamp`, `resource`, `lifecycle`
- Import directly into ProM, Disco, Celonis, or pm4py
- Automatic activity classification

### Insight Reports

**JSON Report** (`full_insights_report.json`):
```json
{
  "generated_at": "2026-01-18T15:30:00",
  "data_science_insights": {
    "ds1_semantic_summary": { "total_events": 1250, ... },
    "ds6_tool_frequency": { "get_current_time": 45, "get_gmt_offset": 42 },
    ...
  },
  "process_mining_insights": {
    "pm4_happy_paths": [ {"sequence": [...], "frequency": 120} ],
    "pm7_bottlenecks": [ {"activity": "tool_search_call", "mean_duration": 2.3} ],
    ...
  }
}
```

## Requirements

### Minimal Installation
- Python 3.8+
- `google-adk>=0.1.0`
- `pandas>=2.0.0`
- `python-dotenv>=1.0.0`

### Analytics Features (Recommended)
- `numpy>=1.24.0` (statistics)
- `scikit-learn>=1.3.0` (clustering)

### Optional
- `rich>=13.0.0` (pretty terminal output)
- `matplotlib>=3.7.0` (visualization)
- `pm4py>=2.7.0` (advanced process mining)

## Use Cases

### 🔬 Research & Development
- Analyze agent behavior patterns
- Identify failure modes
- Optimize tool usage
- Measure performance metrics

### 📈 Production Monitoring
- Track token consumption
- Monitor success rates
- Detect bottlenecks
- Analyze user interactions

### 🏭 Process Optimization
- Discover actual workflows
- Compare against expected processes
- Identify deviations
- Find optimization opportunities

### 📊 Business Intelligence
- Session statistics
- User engagement metrics
- Cost analysis (token usage)
- Quality assurance

## Process Mining Compatibility

Export data directly to industry-standard process mining tools:

- **ProM**: Open-source process mining framework
- **Disco/Celonis**: Commercial process mining platforms
- **pm4py**: Python process mining library
- **Bupar**: R-based process analytics

The `to_process_mining_csv()` method creates XES-compatible CSV files that can be imported directly into these tools for:
- Process discovery (BPMN, Petri nets)
- Conformance checking
- Performance analysis
- Variant analysis

## License

MIT

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## Support

For issues or questions, please open an issue on GitHub.