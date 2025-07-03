# Project File Structure

```eztree-output.txt
./
├── src/
│  ├── diarization/
│  │  ├── __init__.py
│  │  └── diarizer.py
│  ├── monitoring/
│  │  ├── __init__.py
│  │  ├── file_monitor.py
│  │  └── health_check.py
│  ├── output/
│  │  ├── __init__.py
│  │  └── subtitle_manager.py
│  ├── pipeline/
│  │  ├── __init__.py
│  │  └── batch_transcriber.py
│  ├── transcription/
│  │  ├── __init__.py
│  │  └── whisper_factory.py
│  ├── utils/
│  │  ├── __init__.py
│  │  ├── error_handling.py
│  │  └── file_handler.py
│  ├── __init__.py
│  └── config.py
├── tests/
│  ├── __init__.py
│  ├── test_config.py
│  ├── test_error_handling.py
│  └── test_subtitle_manager.py
├── auto_diarize_transcribe.py
├── CHANGELOG.md
├── CLAUDE.md
├── config.yaml
├── docker-compose.yaml
├── Dockerfile
├── FILETREE.md
├── linting-errors.md
├── main.py
├── py-atranscribe.md
├── pyproject.toml
├── README.md
├── requirements.txt
├── transcribe_diarize.md
└── uv.lock

```


This document provides an overview of the py-atranscribe project file structure and organization.

## Root Directory

```
py-atranscribe/
├── README.md                          # Project documentation and setup instructions
├── CHANGELOG.md                       # Version history and release notes
├── FILETREE.md                        # This file - project structure documentation
├── linting-errors.md                  # Comprehensive Ruff linting analysis report
├── CLAUDE.md                          # Instructions for Claude Code assistant
├── py-atranscribe.md                  # Software Design Document (SDD)
├── transcribe_diarize.md             # Additional technical documentation
├── requirements.txt                   # Python package dependencies
├── config.yaml                        # Application configuration template
├── .env.example                       # Environment variables template
├── Dockerfile                         # Docker container configuration
├── docker-compose.yaml               # Docker compose with monitoring stack
├── pyproject.toml                     # Python project configuration and Ruff linting
├── .gitignore                         # Git ignore patterns
├── auto_diarize_transcribe.py         # Main application entry point with enhanced diagnostics
│   ├── __init__.py
│   └── diarizer.py                    # Speaker diarization using pyannote.audio
├── .cursorignore                      # Cursor editor ignore patterns
├── .cursorindexingignore             # Cursor indexing ignore patterns
├── .dockerignore                      # Docker ignore patterns
├── .python-version                    # Python version specification
└── .ruff_cache/                       # Ruff linter cache directory (generated)
```

## Source Code Structure (`src/`)

```
src/
├── __init__.py                        # Package initialization
├── config.py                          # Configuration management with Pydantic models
├── diarization/                       # Speaker diarization module
│   ├── __init__.py
│   └── diarizer.py                    # Speaker diarization using pyannote.audio
├── monitoring/                        # System monitoring and health checks
│   ├── __init__.py
│   ├── file_monitor.py                # File system monitoring with watchdog
│   └── health_check.py                # Health check endpoints with FastAPI
├── output/                            # Output format generation
│   ├── __init__.py
│   └── subtitle_manager.py            # Multi-format subtitle/transcript generation
├── pipeline/                          # Processing pipeline orchestration
│   ├── __init__.py
│   └── batch_transcriber.py           # Main processing pipeline coordinator
├── transcription/                     # Speech-to-text transcription
│   ├── __init__.py
│   └── whisper_factory.py             # Whisper model factory and inference wrapper
└── utils/                             # Utility modules
    ├── __init__.py
    ├── error_handling.py              # Comprehensive error handling and retry logic
    └── file_handler.py                # File handling utilities with media type detection
```

## Test Structure (`tests/`)

```
tests/
├── __init__.py                        # Test package initialization
├── test_config.py                     # Configuration management tests
├── test_error_handling.py             # Error handling and retry mechanism tests
└── test_subtitle_manager.py           # Subtitle format generation tests
```

## Reference Directory (`reference/`)

```
reference/                             # Reference implementation (not tracked in git)
└── Whisper-WebUI/                     # Reference architecture patterns
    └── [external reference files]     # Used for architectural guidance
```

## Key File Descriptions

### Configuration Files

- **config.yaml**: Comprehensive configuration template with all application settings
- **.env.example**: Environment variables template for Docker deployment
- **requirements.txt**: Python dependencies with pinned versions for reproducibility
- **pyproject.toml**: Python project metadata and Ruff linter configuration

### Core Application Files

- **auto_diarize_transcribe.py**: Main application entry point with enhanced diagnostics
- **src/config.py**: Pydantic-based configuration management with validation
- **src/pipeline/batch_transcriber.py**: Main processing pipeline coordinator

### Processing Modules

- **src/transcription/whisper_factory.py**: Factory pattern for Whisper model management
- **src/diarization/diarizer.py**: Speaker diarization using pyannote.audio (TRY300 warning suppressed)
- **src/output/subtitle_manager.py**: Multi-format output generation (SRT, WebVTT, etc.)

### Monitoring and Infrastructure

- **src/monitoring/file_monitor.py**: Watchdog-based file system monitoring (fully Ruff-compliant as of 2025-07-02: structure, type annotations, logging, docstrings, and all lint rules)
- **src/monitoring/health_check.py**: FastAPI health check endpoints
- **src/utils/error_handling.py**: Comprehensive error handling and retry mechanisms

### Docker Configuration

- **Dockerfile**: Multi-stage build for optimized production deployment
- **docker-compose.yaml**: Complete stack with optional Prometheus/Grafana monitoring

### Documentation

- **py-atranscribe.md**: Comprehensive Software Design Document (SDD)
- **CLAUDE.md**: Instructions and context for AI assistant development
- **CHANGELOG.md**: Version history following Keep a Changelog format
- **linting-errors.md**: Comprehensive Ruff linting analysis report with prioritized action plan
- **transcribe_diarize.md**: Additional technical documentation and specifications

## Architecture Overview

The project follows a modular architecture with clear separation of concerns:

1. **Configuration Layer** (`src/config.py`): Centralized configuration management
2. **Monitoring Layer** (`src/monitoring/`): File system monitoring and health checks
3. **Processing Layer** (`src/transcription/`, `src/diarization/`): Core audio processing
4. **Output Layer** (`src/output/`): Multi-format result generation
5. **Pipeline Layer** (`src/pipeline/`): Processing orchestration and workflow management
6. **Utility Layer** (`src/utils/`): Cross-cutting concerns like error handling

## Development Workflow

1. **Source Code**: All application logic is in `src/` with modular organization
2. **Testing**: Comprehensive unit tests in `tests/` directory
3. **Configuration**: Environment-specific settings via YAML and environment variables
4. **Docker**: Production deployment using multi-stage Docker builds
5. **Documentation**: Maintained in Markdown files with architectural decisions recorded

## File Naming Conventions

- **Python modules**: lowercase with underscores (e.g., `file_monitor.py`)
- **Configuration files**: lowercase with extensions (e.g., `config.yaml`)
- **Documentation**: UPPERCASE for key files (e.g., `README.md`, `CHANGELOG.md`)
- **Docker files**: Standard naming (e.g., `Dockerfile`, `docker-compose.yaml`)

This structure supports maintainable, scalable development while adhering to Python packaging standards and Docker best practices.