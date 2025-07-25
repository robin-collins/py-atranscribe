# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About This Project

This is a Python application project for automated audio transcription with speaker diarization. The project implements a Docker-based solution that continuously monitors folders for audio/video files and processes them using faster-whisper and pyannote.audio for transcription and diarization, generating multi-format outputs (SRT, WebVTT, TXT, LRC, JSON, TSV).

The project follows a comprehensive Software Design Document (SDD) in `py-atranscribe.md` with detailed architectural specifications based on proven patterns from the Whisper-WebUI reference implementation. The system features robust error handling, performance optimization, and production-ready containerization.

## Project Structure

This is a new project with the following planned structure:
- Main application will be `auto_diarize_transcribe.py` - the core monitoring and processing application
- Docker configuration for containerized deployment
- Configuration files for audio processing parameters
- Input/output directories for audio file processing
- Reference implementation in `reference/Whisper-WebUI/` for architectural patterns

## Development Commands

Since this is a new project, typical commands will be:

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies (when requirements.txt exists)
pip install -r requirements.txt
```

### Running the Application
```bash
# Run the main transcription application
python auto_diarize_transcribe.py

# Run with Docker (when Dockerfile exists)
docker build -t py-atranscribe .
docker run -v /host/audio/in:/data/in -v /host/audio/out:/data/out py-atranscribe
```

### Testing
```bash
# Run tests (when test files exist)
python -m pytest tests/

# Run with coverage
python -m pytest --cov=src tests/
```

## Key Requirements

Based on the comprehensive SDD, the application must:

1. **Continuously monitor** mounted folders for new audio/video files (30+ supported formats)
2. **Process with faster-whisper** (~60% faster than openai/whisper) with multiple model sizes
3. **Apply speaker diarization** using pyannote.audio 3.3.2 with speaker-diarization-3.1 model
4. **Generate multi-format outputs** (SRT, WebVTT, TXT, LRC, JSON, TSV) with speaker labels
5. **Handle advanced preprocessing** (VAD filtering, BGM separation via UVR)
6. **Implement robust error handling** with automatic retry, graceful degradation, and state recovery
7. **Optimize performance** with memory management, model offloading, and parallel processing
8. **Support flexible file management** (move/delete/keep with atomic operations)
9. **Run indefinitely** in Docker containers with health monitoring and metrics
10. **Provide comprehensive configuration** via YAML, environment variables, and CLI arguments

## Dependencies

Key dependencies to be installed:
- `faster-whisper==1.1.1` - for speech-to-text transcription (~60% faster than openai/whisper)
- `pyannote.audio==3.3.2` - for speaker diarization with speaker-diarization-3.1 model
- `transformers==4.47.1` - for transformer model support
- `torch` and `torchaudio` - for ML model support and GPU acceleration
- `watchdog` - for file system monitoring and change detection
- `pyyaml` - for configuration management and YAML parsing
- `ffmpeg` - for audio/video format conversion and processing
- Additional preprocessing libraries for VAD (Silero) and BGM separation (UVR)

## Configuration

The application should support:
- `config.yaml` for application settings
- Environment variables for Docker deployment
- Configurable input/output/backup directories
- Model selection and device configuration (CPU/GPU)
- Logging levels and output formats

## HuggingFace Setup Required

For speaker diarization with pyannote.audio:
1. Create HuggingFace account and generate READ token
2. Accept terms at:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
3. Set HF_TOKEN environment variable

## Reference Implementation

The `reference/Whisper-WebUI/` directory contains a complete implementation that can be used as architectural reference for:
- Whisper model factory patterns
- Audio file processing pipelines
- Configuration management
- Error handling approaches
- Docker containerization patterns

## File Processing Pipeline

Based on the detailed SDD and reference implementation, the pipeline consists of:

1. **File Detection & Monitoring** - Watchdog-based monitoring with stability detection
2. **Format Validation** - Support for 30+ audio/video formats with format verification
3. **Preprocessing Stage** - VAD filtering (Silero), BGM separation (UVR), format conversion
4. **Transcription Stage** - faster-whisper inference with model management and optimization
5. **Diarization Stage** - pyannote.audio speaker detection and assignment
6. **Output Generation** - Multi-format subtitle generation (SRT, WebVTT, TXT, LRC, JSON, TSV)
7. **Post-processing** - Atomic file operations (move/delete/keep) with state tracking
8. **Error Handling** - Comprehensive error recovery with retry logic and graceful degradation

## Docker Deployment

The application is designed for production Docker deployment with:
- **Multi-stage builds** for optimized image size
- **Host-mounted directories** (input, output, backup) with proper volume management
- **Comprehensive environment variables** for all configuration options
- **Structured logging** to stdout with JSON formatting for monitoring
- **Health check endpoints** on port 8000 for container orchestration
- **Graceful shutdown** handling with processing queue preservation
- **Resource optimization** with GPU/CPU detection and memory management
- **Security** with non-root user execution and minimal attack surface

<!-- BACKLOG.MD GUIDELINES START -->
# Instructions for the usage of Backlog.md CLI Tool

## 1. Source of Truth

- Tasks live under **`backlog/tasks/`** (drafts under **`backlog/drafts/`**).
- Every implementation decision starts with reading the corresponding Markdown task file.
- Project documentation is in **`backlog/docs/`**.
- Project decisions are in **`backlog/decisions/`**.

## 2. Defining Tasks

### **Title**

Use a clear brief title that summarizes the task.

### **Description**: (The **"why"**)

Provide a concise summary of the task purpose and its goal. Do not add implementation details here. It
should explain the purpose and context of the task. Code snippets should be avoided.

### **Acceptance Criteria**: (The **"what"**)

List specific, measurable outcomes that define what means to reach the goal from the description. Use checkboxes (`- [ ]`) for tracking.
When defining `## Acceptance Criteria` for a task, focus on **outcomes, behaviors, and verifiable requirements** rather
than step-by-step implementation details.
Acceptance Criteria (AC) define *what* conditions must be met for the task to be considered complete.
They should be testable and confirm that the core purpose of the task is achieved.
**Key Principles for Good ACs:**

- **Outcome-Oriented:** Focus on the result, not the method.
- **Testable/Verifiable:** Each criterion should be something that can be objectively tested or verified.
- **Clear and Concise:** Unambiguous language.
- **Complete:** Collectively, ACs should cover the scope of the task.
- **User-Focused (where applicable):** Frame ACs from the perspective of the end-user or the system's external behavior.

    - *Good Example:* "- [ ] User can successfully log in with valid credentials."
    - *Good Example:* "- [ ] System processes 1000 requests per second without errors."
    - *Bad Example (Implementation Step):* "- [ ] Add a new function `handleLogin()` in `auth.ts`."

### Task file

Once a task is created it will be stored in `backlog/tasks/` directory as a Markdown file with the format
`task-<id> - <title>.md` (e.g. `task-42 - Add GraphQL resolver.md`).

### Additional task requirements

- Tasks must be **atomic** and **testable**. If a task is too large, break it down into smaller subtasks.
  Each task should represent a single unit of work that can be completed in a single PR.

- **Never** reference tasks that are to be done in the future or that are not yet created. You can only reference
  previous
  tasks (id < current task id).

- When creating multiple tasks, ensure they are **independent** and they do not depend on future tasks.
  Example of wrong tasks splitting: task 1: "Add API endpoint for user data", task 2: "Define the user model and DB
  schema".
  Example of correct tasks splitting: task 1: "Add system for handling API requests", task 2: "Add user model and DB
  schema", task 3: "Add API endpoint for user data".

## 3. Recommended Task Anatomy

```markdown
# task‑42 - Add GraphQL resolver

## Description (the why)

Short, imperative explanation of the goal of the task and why it is needed.

## Acceptance Criteria (the what)

- [ ] Resolver returns correct data for happy path
- [ ] Error response matches REST
- [ ] P95 latency ≤ 50 ms under 100 RPS

## Implementation Plan (the how) (added after starting work on a task)

1. Research existing GraphQL resolver patterns
2. Implement basic resolver with error handling
3. Add performance monitoring
4. Write unit and integration tests
5. Benchmark performance under load

## Implementation Notes (only added after finishing work on a task)

- Approach taken
- Features implemented or modified
- Technical decisions and trade-offs
- Modified or added files
```

## 6. Implementing Tasks

Mandatory sections for every task:

- **Implementation Plan**: (The **"how"**) Outline the steps to achieve the task. Because the implementation details may
  change after the task is created, **the implementation plan must be added only after putting the task in progress**
  and before starting working on the task.
- **Implementation Notes**: Document your approach, decisions, challenges, and any deviations from the plan. This
  section is added after you are done working on the task. It should summarize what you did and why you did it. Keep it
  concise but informative.

**IMPORTANT**: Do not implement anything else that deviates from the **Acceptance Criteria**. If you need to
implement something that is not in the AC, update the AC first and then implement it or create a new task for it.

## 2. Typical Workflow

```bash
# 1 Identify work
backlog task list -s "To Do" --plain

# 2 Read details & documentation
backlog task 42 --plain
# Read also all documentation files in `backlog/docs/` directory.
# Read also all decision files in `backlog/decisions/` directory.

# 3 Start work: assign yourself & move column
backlog task edit 42 -a @{yourself} -s "In Progress"

# 4 Add implementation plan before starting
backlog task edit 42 --plan "1. Analyze current implementation\n2. Identify bottlenecks\n3. Refactor in phases"

# 5 Break work down if needed by creating subtasks or additional tasks
backlog task create "Refactor DB layer" -p 42 -a @{yourself} -d "Description" --ac "Tests pass,Performance improved"

# 6 Complete and mark Done
backlog task edit 42 -s Done --notes "Implemented GraphQL resolver with error handling and performance monitoring"
```

### 7. Final Steps Before Marking a Task as Done

Always ensure you have:

1. ✅ Marked all acceptance criteria as completed (change `- [ ]` to `- [x]`)
2. ✅ Added an `## Implementation Notes` section documenting your approach
3. ✅ Run all tests and linting checks
4. ✅ Updated relevant documentation

## 8. Definition of Done (DoD)

A task is **Done** only when **ALL** of the following are complete:

1. **Acceptance criteria** checklist in the task file is fully checked (all `- [ ]` changed to `- [x]`).
2. **Implementation plan** was followed or deviations were documented in Implementation Notes.
3. **Automated tests** (unit + integration) cover new logic.
4. **Static analysis**: linter & formatter succeed.
5. **Documentation**:
    - All relevant docs updated (any relevant README file, backlog/docs, backlog/decisions, etc.).
    - Task file **MUST** have an `## Implementation Notes` section added summarising:
        - Approach taken
        - Features implemented or modified
        - Technical decisions and trade-offs
        - Modified or added files
6. **Review**: self review code.
7. **Task hygiene**: status set to **Done** via CLI (`backlog task edit <id> -s Done`).
8. **No regressions**: performance, security and licence checks green.

⚠️ **IMPORTANT**: Never mark a task as Done without completing ALL items above.

## 9. Handy CLI Commands

| Purpose          | Command                                                                |
|------------------|------------------------------------------------------------------------|
| Create task      | `backlog task create "Add OAuth"`                                      |
| Create with desc | `backlog task create "Feature" -d "Enables users to use this feature"` |
| Create with AC   | `backlog task create "Feature" --ac "Must work,Must be tested"`        |
| Create with deps | `backlog task create "Feature" --dep task-1,task-2`                    |
| Create sub task  | `backlog task create -p 14 "Add Google auth"`                          |
| List tasks       | `backlog task list --plain`                                            |
| View detail      | `backlog task 7 --plain`                                               |
| Edit             | `backlog task edit 7 -a @{yourself} -l auth,backend`                   |
| Add plan         | `backlog task edit 7 --plan "Implementation approach"`                 |
| Add AC           | `backlog task edit 7 --ac "New criterion,Another one"`                 |
| Add deps         | `backlog task edit 7 --dep task-1,task-2`                              |
| Add notes        | `backlog task edit 7 --notes "We added this and that feature because"` |
| Mark as done     | `backlog task edit 7 -s "Done"`                                        |
| Archive          | `backlog task archive 7`                                               |
| Draft flow       | `backlog draft create "Spike GraphQL"` → `backlog draft promote 3.1`   |
| Demote to draft  | `backlog task demote <task-id>`                                        |

## 10. Tips for AI Agents

- **Always use `--plain` flag** when listing or viewing tasks for AI-friendly text output instead of using Backlog.md
  interactive UI.
- When users mention to create a task, they mean to create a task using Backlog.md CLI tool.

<!-- BACKLOG.MD GUIDELINES END -->
