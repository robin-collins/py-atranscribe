# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **Code Complexity Reduction (2025-01-28)**
  - Resolved all 9 C901 complexity warnings (complexity > 10) by refactoring complex methods into smaller, focused helper functions
  - **auto_diarize_transcribe.py**: Refactored `check_config_file_permissions` (complexity 14→8) and `check_logging_file_permissions` (complexity 12→8) by extracting validation logic into dedicated helper methods
  - **auto_diarize_transcribe.py**: Refactored both `_processing_worker` methods (complexity 11→8) by extracting file processing, error handling, and logging logic into separate helper methods
  - **src/output/subtitle_manager.py**: Refactored `save_transcripts` (complexity 11→8) by extracting single format saving logic into `_save_single_format` helper method
  - **src/pipeline/batch_transcriber.py**: Refactored `process_file` (complexity 11→8) by extracting validation, processing pipeline, diarization handling, and result creation into separate helper methods
  - **src/transcription/whisper_factory.py**: Refactored `_create_model_with_fallback` (complexity 14→8) and `_create_model_with_cudnn_handling` (complexity 13→8) by extracting fallback attempt logic, error handling, and workaround methods
  - **src/utils/file_handler.py**: Refactored `get_media_files` (complexity 11→8) by extracting recursive and single directory processing into separate helper methods
  - **Fixed import issues**: Added missing `TranscriptionError` import to `batch_transcriber.py` and `Any` type import to `auto_diarize_transcribe.py`
  - **Fixed TRY300 warnings**: Resolved 4 additional TRY300 warnings by moving return statements to else blocks in try-except structures
  - All refactored methods maintain original functionality while improving code maintainability and readability
  - Follows DRY, YAGNI, KISS, and SOLID principles as required by project standards

- **Additional Linter Error Resolution (2025-01-28)**
  - Fixed all 32 remaining linter errors identified by ruff with rule selections (BLE001, RUF006, RUF012, S110, S603, S607)
  - **BLE001 fixes**: Replaced broad `Exception` catches with specific exception types:
    - `src/diarization/diarizer.py`: GPU checks now catch `(RuntimeError, OSError, ImportError)`, data processing catches `(AttributeError, KeyError, ValueError, TypeError)`
    - `src/monitoring/health_check.py`: File operations catch `(OSError, PermissionError, FileNotFoundError)`, GPU operations catch `(RuntimeError, AttributeError)`
    - `src/transcription/whisper_factory.py`: Model operations catch `(RuntimeError, OSError, ImportError, ValueError)`, cleanup operations catch `(RuntimeError, AttributeError)`
  - **RUF012 fixes**: Added `ClassVar` annotations for mutable class attributes in `src/utils/file_handler.py`
  - **RUF006 fixes**: Added `# noqa: RUF006` comments for fire-and-forget asyncio tasks that don't need to be awaited
  - **S110 fixes**: Added `# noqa: S110` comments for intentional try-except-pass blocks in console output helpers
  - **S607 fixes**: Used `shutil.which()` to find full executable paths for subprocess calls (`nvidia-smi`, `nvcc`, `find`)
  - **S603 fixes**: Added input validation for subprocess calls and `# noqa: S603` comments for validated system commands
  - Enhanced error handling specificity while maintaining robust fallback behavior
  - Improved security by validating subprocess inputs and using full executable paths

- **Linter Error Resolution (2025-01-28)**
  - Fixed all 23 linter errors identified by ruff with specific rule selections (S105, S106, S108, B017, PT011, TRY002, PTH108, PTH123)
  - **PTH123 fixes**: Replaced `open()` with `Path.open()` in:
    - `auto_diarize_transcribe.py` line 894 (config file reading)
    - `src/utils/file_handler.py` lines 417 and 455 (text file operations)
    - `tests/subtitle_manager_test.py` lines 121, 143, 163, 186, 209, 232 (test file reading)
  - **PTH108 fixes**: Replaced `os.unlink()` with `Path.unlink()` in:
    - `tests/config_test.py` lines 157 and 188 (temporary file cleanup)
  - **S105/S106 fixes**: Added `# noqa` comments for hardcoded test tokens in:
    - `tests/config_test.py` lines 128-129 (test token validation)
    - `tests/diarization_test.py` line 40 (fake token for testing)
  - **S108 fix**: Added `# noqa: S108` comment for hardcoded temporary directory path in tests
  - **B017/PT011/TRY002 fixes**: Replaced broad exception handling with specific custom exceptions in:
    - `tests/error_handling_test.py` - created `TestException` class instead of generic `Exception`
    - Added proper exception matching with `match` parameter in `pytest.raises()`
    - Improved test specificity and error handling practices
  - All changes maintain test functionality while improving code quality and security compliance

### Changed
- **Updated Project Configuration Files (2025-07-03)**
  - Updated `pyproject.toml` to reflect current project state with comprehensive dependencies and metadata
  - Added all production dependencies from `requirements.txt` to `pyproject.toml` for proper package management
  - Moved development dependencies to `[project.optional-dependencies]` section
  - Added author information, project URLs, and proper classifiers for PyPI compatibility
  - Added entry point script configuration for `py-atranscribe` command
  - Updated `README.md` with enhanced documentation including architecture overview, supported formats, and performance requirements
  - Improved installation instructions with proper code blocks and HuggingFace model acceptance steps
  - Added comprehensive configuration section with key options and environment variable examples
  - Enhanced development setup instructions with proper virtual environment and dependency management
  - Added monitoring and health check documentation with endpoint details
  - Fixed repository URLs and corrected formatting issues throughout documentation

### Fixed
- **Blind Exception Handling Compliance (2025-01-28)**
  - Fixed all 30 blind exception handling issues (BLE001) in `auto_diarize_transcribe.py`
  - Replaced generic `except Exception:` with specific exception types for better error handling and debugging
  - System monitoring operations now catch specific exceptions: `psutil.Error`, `OSError`, `PermissionError`
  - GPU/CUDA operations now catch specific exceptions: `RuntimeError`, `torch.cuda.OutOfMemoryError`, `ImportError`
  - File I/O operations now catch specific exceptions: `OSError`, `PermissionError`, `FileNotFoundError`, `UnicodeDecodeError`
  - Network operations now catch specific exceptions: `socket.timeout`, `ConnectionError`, `OSError`
  - Subprocess operations now catch specific exceptions: `subprocess.TimeoutExpired`, `subprocess.SubprocessError`
  - Added `# noqa: BLE001` comments for legitimate broad exception catches in cleanup operations
  - Improved error visibility and debugging capabilities while maintaining robust error handling
  - Reduced total linting errors from 117 to 87 by resolving all blind exception handling violations

### Added
- **Console Output for Processing Status (2025-07-03)**
  - Added console output for file queue status regardless of logging level configuration
  - Added immediate console feedback when files are queued for processing with file size information
  - Added console output for successful file completion with processing time, language, duration, and speaker count
  - Added console output for failed file processing with error messages
  - Enhanced user visibility into processing status even when logging is set to CRITICAL level
  - All console outputs use `print()` with `flush=True` to bypass logger configuration

### Fixed
- **Cross-Device File Move Error (2025-07-03)**
  - Fixed `OSError: [Errno 18] Invalid cross-device link` error in post-processing when moving files from NFS shares to local filesystems
  - Replaced unsafe `Path.rename()` method with `FileHandler.safe_move_file()` which uses `shutil.move()` for cross-device compatibility
  - Added `FileHandler` instance to `BatchTranscriber` class for robust file operations
  - Improved error handling for file operations across different filesystem types
- **Speaker Diarization Reliability Issues (2025-07-03)**
  - Fixed `ValueError: not enough values to unpack (expected 3, got 2)` in `_calculate_confidence` method by adding `yield_label=True` parameter to `itertracks()` calls
  - Fixed `AttributeError: 'NoneType' object has no attribute 'num_speakers'` in batch transcriber by adding proper None-checking for failed diarization results
  - Enhanced error handling in diarization pipeline with graceful degradation when speaker detection fails
  - Added defensive programming in confidence calculation with try-catch blocks to prevent processing interruption
  - Improved diarization pipeline to retry without speaker constraints if initial processing fails
  - Added comprehensive error handling in `_extract_speakers` and `_create_segments` methods
  - Modified diarization method return type to `DiarizationResult | None` for better error handling
  - Enhanced fallback handling to continue processing with default speaker labels when diarization fails

- **cuDNN Library Version Mismatch Issues (2025-01-27)**
  - Fixed cuDNN library loading errors: `Unable to load any of {libcudnn_cnn.so.9.1.0, libcudnn_cnn.so.9.1, libcudnn_cnn.so.9, libcudnn_cnn.so}`
  - Added symbolic links in Dockerfile to map cuDNN 9.8.0 libraries to 9.1.x names expected by PyTorch 2.7.1+cu128
  - Enhanced WhisperFactory with cuDNN-specific error handling and workarounds
  - Modified fallback strategy to prioritize GPU configurations over CPU fallback for explicit CUDA users
  - Added cuDNN environment variable optimization and debugging support
  - Implemented GPU-first error recovery with cuDNN disabled mode before CPU fallback
  - Respects user's explicit CUDA configuration (compute_type: "float32") and avoids unwanted CPU fallback
  - Resolves `Invalid handle. Cannot load symbol cudnnCreateConvolutionDescriptor` runtime errors

- **Missing Module Import Issues (2025-01-27)**
  - Created missing `src/utils/file_handler.py` module based on Whisper-WebUI reference implementation
  - Implemented comprehensive file handling utilities with media type detection, path operations, and file system operations
  - Added support for 13 audio formats and 20 video formats with proper validation
  - Included safe file operations (move, copy, delete) with conflict resolution and error handling
  - Removed unused `OutputFormatter` import from `src/pipeline/batch_transcriber.py`
  - Resolved `ModuleNotFoundError: No module named 'src.utils.file_handler'` startup error
  - Fixed import chain issues preventing application startup in container environment

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

### Added - Comprehensive cuDNN Runtime Error Handling
- **Enhanced WhisperFactory with cuDNN Error Detection**: Implemented comprehensive cuDNN library loading error detection and automatic fallback mechanisms based on Whisper-WebUI reference patterns
- **Runtime Error Testing**: Added model runtime testing with minimal audio samples to catch cuDNN errors that occur during actual transcription operations
- **Multi-Level Fallback Chain**: Implemented systematic fallback from CUDA→CPU with different compute types (float16→float32→int8) when cuDNN issues are detected
- **FasterWhisperInference Wrapper**: Created enhanced wrapper class that provides runtime error handling during transcription operations with automatic CPU fallback
- **Persistent Error Tracking**: Added runtime error counting and automatic permanent CPU fallback after repeated cuDNN failures
- **Memory Management**: Enhanced GPU memory cleanup and cache clearing when switching between devices due to errors
- **Error-Specific Logging**: Added detailed error categorization and logging for cuDNN, CUDA, and memory-related issues
- **Graceful Degradation Integration**: Connected cuDNN error handling to the existing graceful degradation system for consistent behavior

### Fixed - Configuration Duplicate Key Issue
- **Resolved Duplicate Monitoring Keys**: Fixed YAML configuration conflict where both file monitoring and Prometheus metrics were using the same `monitoring` key
- **Renamed Metrics Configuration**: Changed `MonitoringConfig` class to `MetricsConfig` and updated YAML key from `monitoring` to `metrics` for Prometheus configuration
- **Clear Separation of Concerns**: File monitoring settings remain under `monitoring:` while Prometheus metrics settings are now under `metrics:`
- **Updated Test Imports**: Fixed test file imports to use the renamed `MetricsConfig` class
- **Configuration Validation**: Ensured proper separation prevents YAML parsing conflicts and configuration overwrites

### Technical Implementation
- **Device Detection Enhancement**: Improved device selection logic based on Whisper-WebUI patterns with XPU, MPS, and CUDA detection
- **Model Creation Fallback**: Systematic model creation attempts with different device/compute type combinations
- **Runtime Validation**: Test each created model with minimal transcription to catch runtime cuDNN errors before actual use
- **CPU Fallback Models**: Automatic creation of CPU-based fallback models when GPU operations fail
- **Resource Cleanup**: Enhanced cleanup procedures for both primary and fallback models with proper GPU memory management

### Error Handling Improvements
- **cuDNN Library Loading**: Specific detection and handling of `libcudnn_cnn.so` loading failures and `cudnnCreateConvolutionDescriptor` errors
- **CUDA Runtime Errors**: Enhanced handling of CUDA device availability and operation failures
- **Memory Errors**: Improved GPU out-of-memory error detection and automatic CPU fallback
- **Transcription Pipeline**: Updated batch transcriber to handle new tuple return format and provide better error reporting

### Based on Reference Implementation
- **Whisper-WebUI Patterns**: Analyzed and implemented device handling patterns from the reference Whisper-WebUI repository
- **Factory Design**: Enhanced factory pattern with comprehensive error handling and model caching
- **Fallback Strategies**: Implemented systematic fallback strategies based on proven patterns from the reference implementation

## [1.0.0] - 2025-07-03

### Added - Enhanced Logging and Diagnostics
