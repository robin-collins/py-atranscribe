# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Comprehensive Startup Validation System (2025-01-27)**
  - Added `StartupChecker` class to `auto_diarize_transcribe.py` for comprehensive pre-startup validation
  - Implemented visual emoticon-based status reporting for easy Docker log identification (✅ ❌ ⚠️ 🚀 💥 🎉)
  - Added system requirements validation: Python version (3.10+), FFmpeg/FFprobe availability
  - Added critical Python dependencies check: PyTorch, TorchAudio, Faster-Whisper, Pyannote Audio, Transformers, etc.
  - Added optional dependencies validation: Librosa, SoundFile, PyDub, HTTPX, Requests
  - Added ML framework validation: PyTorch functionality test, CUDA availability and GPU memory detection
  - Added AI models validation: Whisper and diarization model import tests
  - Added HuggingFace token validation with masked token display for security
  - Added file system validation: directory permissions, read/write access testing
  - Added resource validation: disk space and memory availability checks
  - Added network connectivity test for model downloads
  - Service now exits with clear error message if critical checks fail
  - Non-critical checks generate warnings but allow service to start with limited functionality

### Changed
- **Enhanced CUDA Support in Docker (2025-01-27)**
  - Modified `Dockerfile` to include CUDA 12.1 runtime libraries and cuDNN support
  - Added NVIDIA package repositories for CUDA runtime installation
  - Updated PyTorch installation to use CUDA 12.1 compatible wheels
  - Enhanced CUDA startup checks to specifically validate cuDNN availability and tensor operations
  - Added comprehensive CUDA environment variables for proper GPU detection
  - Resolved cuDNN library loading issues in Docker containers while maintaining slim base image

### Fixed
- **PyTorch/torchvision Compatibility Issues (2025-01-03)**
  - Fixed `RuntimeError: operator torchvision::nms does not exist` error in Docker container
  - Updated requirements.txt with pinned PyTorch versions for compatibility (torch==2.1.2, torchvision==0.16.2, torchaudio==2.1.2)
  - Modified Dockerfile to uninstall pre-installed PyTorch packages from NVIDIA base image before installing compatible versions
  - Resolved version conflicts between NVIDIA PyTorch base image and pyannote.audio dependencies
- **Docker FFmpeg Build Issues (2025-01-24)**
  - Replaced FFmpeg source compilation with static binary download approach
  - Fixed missing NASM assembler dependency that was causing compilation failures
  - Switched from `ffmpeg-builder` to `ffmpeg-downloader` stage using John Van Sickle's static builds
  - Added fallback commented section for source compilation with proper NASM/YASM dependencies
  - Significantly reduced Docker build time and improved reliability
  - Static binaries include all necessary codecs (x264, x265, libmp3lame, libopus, etc.)

- **Configuration Validation Issues (2025-07-02)**
  - Fixed `AppConfig` model to ignore extra environment variables by adding `extra="ignore"` to `SettingsConfigDict`
  - Resolved Pydantic validation errors where environment variables were being treated as forbidden extra inputs
  - Removed duplicate `monitoring` section in `config.yaml` that was causing YAML parsing conflicts
  - Cleaned up debug print statements from configuration loading process
  - Fixed Docker Compose stack startup failures due to configuration validation errors

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

### Linting Compliance
- **Ruff Lint Fixes (2025-07-02)**
  - Manually resolved all Ruff errors and warnings in `src/monitoring/file_monitor.py`:
    - Moved all imports to the top of the file (E402).
    - Replaced all f-string logging calls with logger formatting (G004).
    - Removed redundant exception objects from `logger.exception` (TRY401).
    - Added missing type annotations for event arguments (ANN001).
    - Refactored nested ifs and combined conditions (SIM102).
    - Moved return statements into else blocks as required (TRY300).
    - Fixed docstring formatting and imperative mood (D205, D401).
    - Reduced return statements in `update_file` to comply with PLR0911.
  - `src/monitoring/file_monitor.py` is now fully Ruff-compliant.

### Fixed
- Suppressed TRY300 warning in `src/diarization/diarizer.py` by adding `# noqa: TRY300` to the return statement within the exception handler.

### Enhanced Diagnostics and Logging
- **Added SystemMonitor class**: Continuous monitoring of system resources with configurable alerting
  - Real-time CPU, memory, GPU, and disk usage monitoring
  - Automated alerts for resource constraints with throttling to prevent spam
  - Processing health monitoring with error rate tracking
- **Enhanced StartupChecker**: Added detailed timing and logging for all startup validation checks
  - Individual check timing and structured logging
  - Comprehensive failure reporting with categorization
  - Startup completion time tracking
  - **NEW: File and folder permission validation**:
    - Configuration file read access and YAML syntax validation
    - Logging file write permissions and directory creation
    - Working directory read/write access verification
    - Container volume mount detection and validation
    - Environment variable validation for containerized deployments
- **Improved TranscriptionService logging**:
  - Service lifecycle tracking with detailed timing metrics
  - Component initialization timing and status reporting
  - Runtime environment logging (Python, PyTorch, GPU, memory, process info)
  - Enhanced worker logging with per-worker statistics and metadata
  - File processing metadata logging (duration, segments, speakers, language)
- **Added comprehensive statistics tracking**:
  - Service-level processing statistics separate from transcriber stats
  - Real-time queue status monitoring
  - Worker performance metrics and final statistics reporting
  - System resource usage in final statistics
- **Enhanced error reporting**:
  - Emoji-enhanced log messages for better visual parsing
  - Structured error logging with timing information
  - Detailed exception context and recovery information
- **Added periodic status reporting**:
  - Configurable status logging interval via command line
  - Comprehensive system health reports including GPU status
  - Processing queue status and error statistics
  - Service uptime and performance metrics
- **Improved shutdown logging**:
  - Detailed shutdown timing for each component
  - Service uptime reporting on graceful shutdown
  - Enhanced error reporting during shutdown process

### Technical Improvements
- Added timing measurements throughout the application lifecycle
- Enhanced file detection logging with file size information
- Improved signal handling and graceful shutdown procedures
- Added command line option for periodic status logging interval
- Comprehensive resource monitoring and alerting system

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