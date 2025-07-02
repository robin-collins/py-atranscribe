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