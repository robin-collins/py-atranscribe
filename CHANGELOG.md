# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **Linting & Code Quality Improvements**
  - Refactored `src/diarization/diarizer.py` to fix docstring formatting (D205), justify broad exception usage (BLE001), and improve structure for Ruff compliance.
  - Refactored `src/monitoring/file_monitor.py` to address nested ifs (SIM102), magic values (PLR2004), and logging best practices (G004).
  - Introduced `FILE_STABILITY_TIME_THRESHOLD` constant to replace magic value in file stability checks.
  - Combined nested conditions for file stability and removed f-strings from logger calls.
  - Improved maintainability and reduced Ruff error count in both modules.

- **File Processing Logic Enhancement**
  - Modified file monitoring system to exclude the most recent audio file from processing unless it's older than 24 hours
  - Updated `FileStabilityTracker.get_pending_files()` method in `src/monitoring/file_monitor.py`
  - Prevents processing of files that may still be actively recording or uploading
  - Improves reliability by ensuring only completed audio files are processed

## [1.0.0] - 2025-07-02

### Added
- **Core Application Architecture**
  - Modular Python application for automated audio transcription with speaker diarization
  - Configuration management using Pydantic with YAML and environment variable support
  - Comprehensive error handling with retry mechanisms and graceful degradation
  - Factory pattern for Whisper model management with caching and device optimization

- **Audio Processing Pipeline**
  - faster-whisper integration for high-performance speech-to-text transcription (~60% faster than openai/whisper)
  - pyannote.audio 3.3.2 integration for speaker diarization with speaker-diarization-3.1 model
  - Support for 30+ audio/video formats (MP3, WAV, MP4, etc.)
  - Voice Activity Detection (VAD) and background music separation capabilities
  - Language auto-detection with confidence scoring

- **File System Monitoring**
  - Watchdog-based continuous folder monitoring with stability detection
  - Processing queue with deduplication and priority handling
  - Configurable stability delay to ensure complete file uploads
  - Support for recursive directory monitoring

- **Multi-Format Output Generation**
  - SRT subtitle format with speaker labels and timing
  - WebVTT format for web-based video players
  - Plain text transcripts with speaker identification
  - JSON format with detailed segment and word-level metadata
  - TSV format for spreadsheet analysis
  - LRC format for karaoke-style synchronized lyrics

- **Speaker Diarization System**
  - HuggingFace token authentication for model access
  - Configurable minimum/maximum speaker detection
  - Speaker embedding and confidence scoring
  - Time-based segment assignment with overlap detection
  - Speaker statistics and duration analysis

- **Performance Optimization**
  - GPU/CPU automatic device detection and fallback
  - Memory usage monitoring and limits
  - Model offloading for GPU memory management
  - Concurrent file processing with configurable limits
  - Circuit breaker pattern for failure handling

- **Docker Containerization**
  - Multi-stage Dockerfile for optimized production deployment
  - docker-compose.yaml with optional monitoring stack (Prometheus/Grafana)
  - Volume mounting for input/output directories
  - Health check endpoints on port 8000
  - Non-root user execution for security

- **Monitoring and Health Checks**
  - FastAPI-based health check endpoints
  - System metrics collection (CPU, memory, disk usage)
  - Processing queue status monitoring
  - Error tracking and categorization
  - Prometheus metrics integration support

- **Comprehensive Testing**
  - Unit tests for configuration management and validation
  - Error handling and retry mechanism tests
  - Subtitle format generation and validation tests
  - Mock-based testing for external dependencies
  - Test coverage for critical application components

- **Configuration Management**
  - Flexible YAML-based configuration with sensible defaults
  - Environment variable override support with nested delimiter handling
  - Runtime configuration validation with detailed error messages
  - Directory auto-creation with permission checking
  - HuggingFace token validation for diarization features

- **Error Handling and Resilience**
  - Comprehensive error classification (network, file system, memory, GPU, model, audio)
  - Automatic retry logic with exponential backoff and jitter
  - Graceful degradation for resource-constrained environments
  - Circuit breaker pattern for preventing cascade failures
  - Detailed error logging with structured metadata

### Technical Specifications
- **Dependencies**: faster-whisper 1.1.1, pyannote.audio 3.3.2, transformers 4.47.1
- **Python Version**: 3.11+
- **GPU Support**: CUDA-compatible with automatic CPU fallback
- **Memory Requirements**: Configurable (default 8GB limit)
- **Supported Formats**: 30+ audio/video formats including MP3, WAV, MP4, AVI, MKV
- **Output Formats**: SRT, WebVTT, TXT, JSON, TSV, LRC

### Architecture Highlights
- Factory pattern for model management and caching
- Observer pattern for file system monitoring
- Pipeline pattern for processing orchestration
- Strategy pattern for output format generation
- Repository pattern for configuration management

### Changed
- Updated `pyproject.toml` with modern Ruff configuration following latest best practices

### Fixed
- Unicode character encoding issues in whisper_factory.py punctuation strings
- Resolved TOML schema validation errors in Ruff configuration by simplifying the structure
- Corrected duplicate rule entries and invalid configuration options

### Documentation
- **Comprehensive Linting Report**: Generated detailed analysis of code quality issues
  - Created `linting-errors.md` with 50 identified issues across 8 Python files
  - Categorized issues by type: deprecated imports, exception handling, code simplification, etc.
  - Provided action plan with priority levels for systematic code quality improvement
  - No automatic fixes applied - all issues require manual intervention or review