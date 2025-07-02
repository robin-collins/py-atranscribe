# **Software Design Document (SDD) — Automated Audio Folder Transcription with Speaker Diarization (Docker Edition)**

## **1. Overview**

Develop a Python application, packaged as a Docker image, that **continuously monitors** a designated (mounted) folder for new audio files. Each file is automatically transcribed using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and [pyannote.audio](https://github.com/pyannote/pyannote-audio) for speaker diarization, producing timestamped, speaker-labeled transcripts in multiple formats (SRT, WebVTT, TXT, LRC, JSON, TSV). The application must run **indefinitely** in a container, with robust error handling and configurable post-processing (backup/delete) of audio files.

**Core Architecture**: Based on proven patterns from [Whisper-WebUI](https://github.com/jhj0517/Whisper-WebUI), implementing a factory-based design with modular components for transcription, diarization, and file processing pipelines.

---

## **2. Functional Requirements**

### **2.1. Input Monitoring (Docker-ready)**
- Monitor a **host-mounted input directory** for new audio files (`.wav`, `.mp3`, `.flac`, etc.).
- Start transcription only when files are fully written.
- Designed for operation as a long-running background process inside a container.

### **2.2. Transcription & Diarization**
- Use [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for transcription.
- Integrate an open-source speaker diarization solution (e.g., `pyannote.audio`).
- Output for each audio file must include:
  - **Timestamps**
  - **Speaker tags**
  - **Transcribed text**
  - Output format: `.txt` (plaintext), optionally `.json` or `.srt`.
- Do **not** save transcription if result is empty/inaudible.

### **2.3. File Handling & Post-Processing**
- After successful transcription (non-empty), perform one of:
  - **Move** (backup) the original audio file to a configurable backup directory (host-mounted), or
  - **Delete** the original audio file.
- Configurable via environment variables or config file.
- Fault tolerance: never lose or corrupt audio or transcripts on crash/restart.

### **2.4. Robust Operation**
- Must **run indefinitely** in Docker, gracefully handling:
  - File access errors
  - Network failures (for model downloads)
  - Transcription or diarization failures (log and skip/retry as needed)
- Never process or output the same file twice.

### **2.5. Configuration**
- All options configurable via `config.yaml` and/or environment variables for Docker use:
  - Input/output/backup directory paths
  - File types to watch
  - Post-processing action
  - Model selection, device (CPU/GPU), diarization settings
  - Logging verbosity

### **2.6. Logging & Observability**
- Log all actions and errors to **stdout** (visible via `docker logs`).
- (Optional) Healthcheck endpoint (e.g., simple HTTP server on port 8000) for container orchestration (K8s, Docker Compose).

### **2.7. Resilience**
- Designed for “fire-and-forget” operation: recovers from all transient errors, can be safely stopped/restarted at any time with no loss or duplication.
- Processes files atomically—handles crashes or container restarts cleanly.

---

## **3. Non-Functional Requirements**

- **Docker-native:** All dependencies (including ML models/tools) are installed and configured at build time.
- **Portability:** Works on x86 and ARM Docker hosts.
- **Efficiency:** Minimal idle CPU/memory when not processing files.
- **Security:** Only processes files in mounted directory; runs as non-root user inside container.
- **Observability:** Suitable for unattended headless deployment (docker, compose, k8s).

---

## **4. Architecture/Components**

### **4.1 Core System Architecture**

The system follows a modular, factory-based architecture derived from Whisper-WebUI patterns:

```
┌─────────────────────────────────────────────────────────────────┐
│                     File Monitoring Layer                      │
├─────────────────────────────────────────────────────────────────┤
│                   Batch Processing Controller                   │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  WhisperFactory │  │  BatchProcessor │  │  FileManager    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│           Transcription Pipeline Components                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ FasterWhisper   │  │   Diarizer      │  │ SubtitleManager │ │
│  │   Inference     │  │   (pyannote)    │  │                 │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                   Preprocessing Layer                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   VAD Filter    │  │  BGM Separator  │  │  File Validator │ │
│  │   (Silero)      │  │     (UVR)       │  │                 │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### **4.2 Component Specifications**

#### **File Monitoring Layer**
- **File Watcher:** Uses [`watchdog`](https://github.com/gorakhargosh/watchdog) or efficient polling for new files
- **File Validator:** Validates file formats, completeness, and readiness for processing
- **Input Queue:** Manages processing queue with deduplication and state tracking

#### **Batch Processing Controller**
- **BatchTranscriber Class:** Main orchestrator implementing the processing pipeline
- **Pipeline Parameter Factory:** Creates configuration objects for different processing modes
- **Progress Tracking:** Real-time progress callbacks and status reporting

#### **Transcription Pipeline Components**
- **WhisperFactory:** Factory pattern for creating Whisper inference instances
  - Reference: `modules/whisper/whisper_factory.py:WhisperFactory.create_whisper_inference()`
- **FasterWhisperInference:** Core transcription engine with model management
  - Reference: `modules/whisper/faster_whisper_inference.py:FasterWhisperInference`
- **Diarizer:** Speaker diarization wrapper using pyannote.audio
  - Reference: `modules/diarize/diarizer.py:Diarizer`
- **SubtitleManager:** Multi-format output generation and file management
  - Reference: `modules/utils/subtitle_manager.py:SubtitleManager`

#### **Preprocessing Layer**
- **VAD Filter:** Voice Activity Detection using Silero VAD to remove silence
- **BGM Separator:** Background music separation using UVR (Ultimate Vocal Remover)
- **Audio Processor:** Handles format conversion and audio preprocessing

#### **Data Management Layer**
- **File Manager:** Handles file operations, backup/delete post-processing
- **Configuration Manager:** Loads and validates configuration from YAML/environment
- **Model Manager:** Downloads, caches, and manages ML models
- **Logger:** Structured logging to stdout with configurable levels

## **5. Error Handling & Recovery Mechanisms**

### **5.1 Comprehensive Error Classification**

#### **File System Errors**
- **Permission Errors**: Read/write access to input/output directories
- **Storage Errors**: Disk space exhaustion, I/O failures
- **File Lock Errors**: Files in use by other processes
- **Network Mount Errors**: Temporary disconnection of mounted volumes

#### **Model Loading Errors**
- **Network Failures**: Model download interruptions
- **Memory Errors**: Insufficient VRAM/RAM for model loading
- **CUDA Errors**: GPU driver issues, CUDA version mismatches
- **Authentication Errors**: Invalid HuggingFace tokens

#### **Processing Errors**
- **Audio Format Errors**: Corrupted or unsupported audio files
- **Transcription Failures**: Silent audio, extreme noise, codec issues
- **Diarization Failures**: Single speaker detection issues, embedding failures
- **Memory Exhaustion**: VRAM/RAM overflow during processing

### **5.2 Recovery Strategies**

#### **Automatic Retry Logic**
```python
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    
    retryable_exceptions: List[Type[Exception]] = [
        ConnectionError,
        TimeoutError,
        torch.cuda.OutOfMemoryError,
        FileNotFoundError
    ]
```

#### **Graceful Degradation**
- **Model Fallback**: Automatic fallback to smaller models on memory errors
- **Compute Type Reduction**: `float16` → `float32` → `int8` on GPU memory issues
- **Device Fallback**: CUDA → CPU when GPU unavailable
- **Feature Disable**: Disable diarization/BGM separation on resource constraints

#### **State Recovery**
- **Processing Checkpoints**: Save intermediate results for long-running operations
- **Queue Persistence**: Maintain processing queue across container restarts
- **Partial Results**: Save completed segments even if processing fails
- **Lock File Management**: Prevent duplicate processing with atomic locks

### **5.3 Monitoring & Alerting**

#### **Health Check Implementation**
```python
class HealthChecker:
    def check_disk_space(self, min_free_gb: float = 1.0) -> bool
    def check_model_availability(self) -> bool
    def check_gpu_memory(self) -> bool
    def check_processing_queue(self) -> bool
    def get_system_metrics(self) -> Dict[str, Any]
```

#### **Error Reporting**
- **Structured Error Logs**: JSON-formatted error details with context
- **Error Categories**: Classification for automated monitoring
- **Performance Metrics**: Processing time, success rates, resource usage
- **Alert Thresholds**: Configurable thresholds for error rate monitoring

## **6. Performance Optimization & Resource Management**

### **6.1 Memory Optimization**

#### **Model Management**
- **Lazy Loading**: Load models only when needed
- **Model Offloading**: Automatic GPU memory management with `enable_offload=True`
- **Compute Type Optimization**: Use `int8` quantization for memory-constrained environments
- **Batch Size Tuning**: Dynamic batch size adjustment based on available memory

#### **Memory Monitoring**
```python
class MemoryManager:
    def get_gpu_memory_usage(self) -> Dict[str, float]
    def estimate_model_memory(self, model_size: str) -> float
    def optimize_batch_size(self, available_memory: float) -> int
    def trigger_garbage_collection(self) -> None
```

### **6.2 Processing Optimization**

#### **Performance Features**
- **faster-whisper**: ~60% speed improvement over openai/whisper
- **VAD Filtering**: Skip silent segments to reduce processing time
- **Chunk Processing**: Process long audio files in 30-second segments
- **Parallel Processing**: Concurrent file processing with configurable limits

#### **Optimization Parameters**
```python
class OptimizationConfig:
    chunk_length_s: int = 30          # Audio chunk duration
    batch_size: int = 16              # Batch processing size
    max_concurrent_files: int = 4     # Parallel file processing limit
    vad_threshold: float = 0.5        # Voice activity detection threshold
    enable_model_offload: bool = True # GPU memory optimization
    use_gpu_memory_fraction: float = 0.8  # GPU memory allocation limit
```

---

## **5. Deliverables**

### **5.1 Core Application Files**
1. **Main Application:** `auto_diarize_transcribe.py` - File monitoring and processing orchestrator
2. **Batch Processor:** `batch_transcriber.py` - Core transcription pipeline implementation
3. **Configuration:** `config.yaml` - Comprehensive configuration template
4. **Dependencies:** `requirements.txt` - Python package dependencies

### **5.2 Containerization**
5. **Dockerfile:** Multi-stage build for production-ready image
6. **Docker Compose:** `docker-compose.yaml` - Development and deployment configuration
7. **Environment:** `.env.example` - Environment variable template

### **5.3 Documentation**
8. **README.md:** Comprehensive usage, Docker instructions, configuration guide
9. **API Documentation:** Detailed API specifications for integration
10. **CHANGELOG.md:** Version history and change tracking

### **5.4 Optional Extensions**
11. **Gradio UI:** `gradio_interface.py` - Web interface for manual operations
12. **Health Check:** `health_check.py` - Container health monitoring endpoint
13. **Monitoring:** `metrics.py` - Performance and processing metrics collection

---

## **6. Sample Docker Usage**

```bash
docker run --rm -it \
  -v /host/audio/in:/data/in \
  -v /host/audio/out:/data/out \
  -v /host/audio/backup:/data/backup \
  -e CONFIG_PATH=/config/config.yaml \
  -e LOG_LEVEL=INFO \
  mytranscriber:latest
````

* `/data/in`, `/data/out`, `/data/backup` are host paths mounted into the container.
* All options can be overridden via environment variables.

---

## **7. Reference Implementation**

* Base folder-watcher and batch logic on this SDD.
* Optional UI/UX via [Whisper-WebUI](https://github.com/jhj0517/Whisper-WebUI).

---

*For initial code skeletons or Dockerfile, request specific examples.*

```

**Your canvas document has been updated.**
If you require a full example Dockerfile, config, or initial app skeleton, specify which files you want generated.
```
