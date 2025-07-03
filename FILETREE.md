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
├── scripts/
│  └── bss-pyatranscribe.sh
├── audio/
│  ├── backup/
│  │  └── 2025-07-03/
│  │      ├── audio_20250703_030630_22kHz.flac
│  │      ├── audio_20250703_031133_22kHz.flac
│  │      └── audio_20250703_031635_22kHz.flac
│  └── output/
│      ├── audio_20250703_030630_22kHz.srt
│      ├── audio_20250703_030630_22kHz.tsv
│      ├── audio_20250703_030630_22kHz.txt
│      ├── audio_20250703_031133_22kHz.srt
│      ├── audio_20250703_031133_22kHz.tsv
│      ├── audio_20250703_031133_22kHz.txt
│      ├── audio_20250703_031635_22kHz.srt
│      ├── audio_20250703_031635_22kHz.tsv
│      └── audio_20250703_031635_22kHz.txt
├── logs/
│  └── transcribe.log
├── reference/
│  ├── Whisper-WebUI-concise/
│  │  └── [reference implementation files]
│  └── [documentation and analysis files]
├── auto_diarize_transcribe.py
├── CHANGELOG.md
├── CLAUDE.md
├── config.yaml
├── docker-compose.yaml
├── Dockerfile
├── entrypoint.sh
├── FAIILURELOG.md
├── FILETREE.md
├── pyproject.toml
├── README.md
└── requirements.txt

```


This document provides an overview of the py-atranscribe project file structure and organization.

## Root Directory

```
py-atranscribe/
├── README.md                          # Project documentation and setup instructions
├── CHANGELOG.md                       # Version history and release notes
├── FILETREE.md                        # This file - project structure documentation
├── FAIILURELOG.md                     # Documentation of failed attempts and debugging notes
├── CLAUDE.md                          # Instructions for Claude Code assistant
├── requirements.txt                   # Python package dependencies
├── config.yaml                        # Application configuration template
├── Dockerfile                         # Docker container configuration
├── docker-compose.yaml               # Docker compose with monitoring stack
├── entrypoint.sh                      # Docker container entrypoint script
├── pyproject.toml                     # Python project configuration and dependencies
├── auto_diarize_transcribe.py         # Main application entry point with enhanced diagnostics
├── audio/                             # Audio processing directories
│   ├── backup/                        # Processed audio files backup
│   │   └── 2025-07-03/               # Daily backup folders
│   └── output/                        # Generated transcription outputs
├── logs/                              # Application log files
│   └── transcribe.log                # Main application log
├── scripts/                           # Utility scripts
│   └── bss-pyatranscribe.sh          # Bash script for service management
├── reference/                         # Reference implementation (not tracked in git)
│   └── Whisper-WebUI-concise/        # Reference architecture patterns
└── temp/                              # Temporary files directory
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
reference/                             # Reference implementation and documentation
├── Whisper-WebUI-concise/            # Reference architecture patterns
│   ├── backend/                      # Backend service architecture
│   ├── configs/                      # Configuration examples
│   ├── models/                       # Model storage structure
│   ├── modules/                      # Core processing modules
│   └── requirements.txt              # Reference dependencies
├── Docker/                           # Docker reference configurations
├── [analysis and documentation files] # Technical analysis and debugging notes
└── [external reference files]        # Used for architectural guidance
```

## Runtime Directories

```
audio/                                # Audio processing workspace
├── backup/                           # Processed files backup
│   └── 2025-07-03/                  # Daily organized backups
├── input/                            # Monitored input directory (created at runtime)
└── output/                           # Generated transcription outputs
    ├── *.srt                         # Subtitle files with speaker labels
    ├── *.tsv                         # Tab-separated transcription data
    └── *.txt                         # Plain text transcriptions

logs/                                 # Application logging
└── transcribe.log                    # Main application log file

temp/                                 # Temporary processing files
└── [runtime temporary files]         # Cleaned up automatically
```

## Key File Descriptions

### Configuration Files

- **config.yaml**: Comprehensive configuration template with all application settings
- **.env.example**: Environment variables template for Docker deployment
- **requirements.txt**: Python dependencies with pinned versions for reproducibility
- **pyproject.toml**: Python project metadata and Ruff linter configuration

### Core Application Files

- **auto_diarize_transcribe.py**: Main application entry point with enhanced diagnostics and system monitoring
- **src/config.py**: Pydantic-based configuration management with validation
- **src/pipeline/batch_transcriber.py**: Main processing pipeline coordinator
- **entrypoint.sh**: Docker container initialization script

### Processing Modules

- **src/transcription/whisper_factory.py**: Factory pattern for Whisper model management
- **src/diarization/diarizer.py**: Speaker diarization using pyannote.audio (TRY300 warning suppressed)
- **src/output/subtitle_manager.py**: Multi-format output generation (SRT, WebVTT, etc.)

### Monitoring and Infrastructure

- **src/monitoring/file_monitor.py**: Watchdog-based file system monitoring (fully Ruff-compliant as of 2025-07-02: structure, type annotations, logging, docstrings, and all lint rules)
- **src/monitoring/health_check.py**: FastAPI health check endpoints
- **src/utils/error_handling.py**: Comprehensive error handling and retry mechanisms

### Docker Configuration

- **Dockerfile**: Multi-stage build for optimized production deployment with CUDA support
- **docker-compose.yaml**: Complete stack with volume mounting for audio processing
- **entrypoint.sh**: Container startup script for proper initialization

### Scripts and Utilities

- **scripts/bss-pyatranscribe.sh**: Bash script for service management and automation
- **temp/**: Temporary files directory for processing intermediate files

### Documentation

- **CLAUDE.md**: Instructions and context for AI assistant development
- **CHANGELOG.md**: Version history following Keep a Changelog format
- **FAIILURELOG.md**: Documentation of failed attempts and debugging notes
- **reference/**: Reference implementation analysis and technical documentation

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
4. **Docker**: Production deployment using multi-stage Docker builds with CUDA support
5. **Documentation**: Maintained in Markdown files with architectural decisions recorded
6. **Processing**: Audio files processed through `audio/` directory structure
7. **Logging**: Centralized logging in `logs/` directory for monitoring and debugging
8. **Scripts**: Utility scripts in `scripts/` directory for automation and management

## File Naming Conventions

- **Python modules**: lowercase with underscores (e.g., `file_monitor.py`)
- **Configuration files**: lowercase with extensions (e.g., `config.yaml`)
- **Documentation**: UPPERCASE for key files (e.g., `README.md`, `CHANGELOG.md`)
- **Docker files**: Standard naming (e.g., `Dockerfile`, `docker-compose.yaml`)
- **Audio files**: Timestamped format (e.g., `audio_20250703_030630_22kHz.flac`)
- **Output files**: Match input filename with format extension (e.g., `*.srt`, `*.tsv`, `*.txt`)
- **Log files**: Descriptive naming (e.g., `transcribe.log`)
- **Script files**: Descriptive with extension (e.g., `bss-pyatranscribe.sh`)

This structure supports maintainable, scalable development while adhering to Python packaging standards and Docker best practices. The audio processing workflow is optimized for continuous monitoring and automated transcription with speaker diarization.
