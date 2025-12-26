# Event Logger for Google Gen AI Agents

A robust Python logging utility designed specifically for tracking, storing, and exporting interactions with Google Gen AI Agents (Gemini/Vertex AI).

It captures not just the text content, but also critical metadata often missed by standard loggers: **function calls/responses**, **token usage**, **safety ratings**, **grounding metadata** (Google Search sources), and **citations**.

## Features

- **Comprehensive serialization**: Converts complex ADK `Event` objects into clean, serializable JSON dictionaries.
- **Chronological Sorting**: Log files are automatically prefixed with timestamps (`YYYYMMDD_HHMMSS_{session_id}.jsonl`) to ensure easy sorting in file explorers.
- **Session Management**: Automatically manages file creation. It creates a new file for a new session and appends to existing files if the session continues.
- **Rich Metadata Support**:
  - 🛡️ **Safety Ratings**: Logs probability and categories of blocked content.
  - ⚓ **Grounding**: Captures web search queries and entry points.
  - 🧩 **Function Calls**: Logs arguments and responses for tool use.
  - 🪙 **Token Usage**: Tracks prompt and completion token counts.
- **Session Export**: Utility to merge fragmented JSONL logs into a single, clean JSON report for analysis.

## Installation

### Standard `pip`

You can install this package directly from the source:

```bash
pip install .

```

Or directly from GitHub:

```bash
pip install git+[https://github.com/clementadande/event_logger.git](https://github.com/clementadande/event_logger.git)

```

### Using `uv` (Fast Python Installer)

If you are using [uv](https://docs.astral.sh/uv/) for dependency management, you can easily add this package.

To add it to your project (`pyproject.toml`):

```bash
uv add git+[https://github.com/clementadande/event_logger.git](https://github.com/clementadande/event_logger.git)

```

Or to install it directly into the current virtual environment:

```bash
uv pip install git+[https://github.com/clementadande/event_logger.git](https://github.com/clementadande/event_logger.git)

```

## Usage

### 1. Basic Logging

Initialize the logger and log events as they occur in your agent's loop.

```python
from event_logger import EventLogger, serialize_event_to_dict

# Initialize logger (defaults to .log/ directory)
logger = EventLogger(base_dir="logs")

user_id = "user_123"
session_id = "session_abc"

# ... inside your agent loop where you receive an 'event' object ...
# event is typically a google.adk.events.Event object

# 1. Serialize the ADK event to a dictionary
event_dict = serialize_event_to_dict(adk_event)

# 2. Log it to disk
logger.log(event_dict, user_id, session_id)

```

This will create a file named something like: `logs/20250521_143005_session_abc.jsonl`.

### 2. Exporting a Session

If a session spans multiple interactions or days, the logs might be split or appended. The export function gathers everything related to a specific session ID and produces a clean JSON report.

```python
logger.export_session(
    user_id="user_123", 
    session_id="session_abc", 
    output_file="final_report.json"
)

```

## Output Format

### Log File Naming

Files are named using the format: `YYYYMMDD_HHMMSS_{session_id}.jsonl`.

* **YYYYMMDD_HHMMSS**: The timestamp of the *first* event in that file/session start.
* **session_id**: The unique identifier for the session.

### JSON Structure

Each line in the `.jsonl` file (and the objects in the exported JSON) contains detailed info:

```json
{
  "timestamp": "2023-10-27T10:00:00.123456",
  "event_id": "...",
  "author": "model",
  "content": [
    { "type": "text", "value": "Hello! I found some info." },
    { "type": "function_call", "name": "search", "args": {"query": "python"} }
  ],
  "safety_ratings": [
    { "category": "HARM_CATEGORY_HATE_SPEECH", "probability": "NEGLIGIBLE", "blocked": false }
  ],
  "grounding_metadata": {
    "web_search_queries": ["python documentation"],
    "search_entry_point": "..."
  },
  "token_usage": {
    "total_tokens": 150
  }
}

```

## Requirements

* Python 3.8+
* `google-adk`
