# **py-atranscribe: Automated Transcription & Diarization Service**

An automated, containerized Python application that continuously monitors a directory for new audio/video files and processes them through a robust transcription and speaker diarization pipeline.

Built with faster-whisper for high-performance transcription (~60% faster than OpenAI Whisper) and pyannote.audio for accurate speaker diarization, this service is designed for "fire-and-forget" operation in production environments with comprehensive error handling, health monitoring, and multi-format output generation.

## **Key Features**

* **High-Performance Transcription**: Utilizes faster-whisper for speech-to-text processing that is significantly faster than the standard OpenAI Whisper implementation.
* **Accurate Speaker Diarization**: Integrates pyannote.audio to identify and label different speakers in an audio file.
* **Continuous Folder Monitoring**: Uses watchdog to automatically detect and process new files added to a designated input directory.
* **Multi-Format Output**: Generates transcripts in various formats, including SRT, WebVTT, TXT, and detailed JSON.
* **Broad File Support**: Natively handles over 30 common audio and video file formats.
* **Dockerized & Production-Ready**: Comes with a multi-stage Dockerfile and docker-compose.yaml for easy, secure, and efficient deployment.
* **Robust Error Handling**: Features automatic retries with exponential backoff, a circuit breaker pattern, and graceful degradation to ensure resilient operation.
* **Health Monitoring**: Includes a FastAPI-based health check endpoint for container orchestration systems like Kubernetes or Docker Swarm.
* **Flexible Configuration**: Easily configure the application using a config.yaml file, with support for environment variable overrides.

## **Getting Started**

### **Prerequisites**

* [Docker](https://www.docker.com/get-started) and [Docker Compose](https://docs.docker.com/compose/install/)
* Git
* Python 3.11+ (if running locally without Docker)
* HuggingFace account and access token (for speaker diarization models)

### **Installation & Setup**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/robinsmj2015/py-atranscribe.git
   cd py-atranscribe
   ```

2. **Create the environment file:**
   ```bash
   cp .env.example .env
   ```

3. **Set Your Hugging Face Token:**
   Open the `.env` file and add your Hugging Face access token. This is required for speaker diarization. You can get a token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
   ```bash
   # .env
   HF_TOKEN=your_huggingface_token_here
   ```

4. **Accept HuggingFace Model Terms:**
   Accept the terms for the required models:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

5. **Create Input/Output Directories:**
   ```bash
   mkdir -p ./audio/input ./audio/output ./audio/backup
   ```

6. **Build and Run the Service:**
   ```bash
   docker-compose up --build -d
   ```

The service is now running and monitoring the ./audio/input directory for new files.

## **Usage**

1. **Add Files**: Simply copy or move your audio or video files into the ./audio/input directory on your host machine.
2. **Processing**: The service will automatically detect the new file, wait for it to be fully written, and then begin the transcription and diarization process.
3. **Get Transcripts**: The output files (e.g., .srt, .vtt, .txt) will appear in the ./audio/output directory.
4. **Check Logs**: To view the real-time processing logs, run:
   ```bash
   docker-compose logs -f
   ```

## **Configuration**

The application's behavior can be customized through the config.yaml file and overridden by environment variables defined in .env.

* **config.yaml**: Contains detailed settings for transcription models, diarization parameters, performance tuning, and more.
* **.env**: Used to set high-level configuration and secrets, such as the HF_TOKEN. Environment variables follow a PARENT__CHILD structure to override nested YAML keys (e.g., TRANSCRIPTION__WHISPER__MODEL_SIZE=large-v3).

### **Key Configuration Options**

- **Whisper Model Size**: `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`
- **Device Selection**: `auto`, `cpu`, `cuda`
- **Batch Size**: Configurable for performance optimization
- **Language Detection**: Auto-detect or specify language
- **Speaker Count**: Auto-detect or specify number of speakers
- **File Handling**: Move to backup, delete, or keep original files

Refer to config.yaml for a full list of available options.

## **Health Check**

The service exposes a health check endpoint for monitoring, which is used by the Docker health check.

* **URL**: http://localhost:8000/health
* **Description**: Returns a 200 OK status if the service is running correctly and 503 Service Unavailable if there is an issue.

## **Development**

### **Local Setup**

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   # Or install with development dependencies
   pip install -e ".[dev]"
   ```

3. **Set up environment variables:**
   ```bash
   export HF_TOKEN=your_huggingface_token_here
   ```

4. **Run the application:**
   ```bash
   python auto_diarize_transcribe.py --config config.yaml
   ```

### **Running Tests**

To run the comprehensive unit test suite:
```bash
pytest
```

To include coverage reporting:
```bash
pytest --cov=src
```

To run linting and formatting:
```bash
ruff check src/
ruff format src/
```

## **Architecture**

The application follows a modular architecture with the following components:

- **File Monitor** (`src/monitoring/file_monitor.py`): Watches input directories for new files
- **Transcription Engine** (`src/transcription/whisper_factory.py`): Handles faster-whisper integration
- **Speaker Diarization** (`src/diarization/diarizer.py`): Manages pyannote.audio for speaker identification
- **Batch Processing** (`src/pipeline/batch_transcriber.py`): Coordinates the transcription pipeline
- **Output Generation** (`src/output/subtitle_manager.py`): Creates multi-format subtitle files
- **Configuration Management** (`src/config.py`): Handles YAML and environment variable configuration
- **Error Handling** (`src/utils/error_handling.py`): Provides robust error recovery and logging

## **Supported File Formats**

**Input Formats:** MP3, WAV, FLAC, M4A, AAC, OGG, WMA, MP4, AVI, MKV, MOV, WMV, FLV, WEBM, and 20+ others

**Output Formats:**
- **SRT**: Standard subtitle format with speaker labels
- **WebVTT**: Web-compatible subtitle format
- **TXT**: Plain text transcription with timestamps
- **JSON**: Structured data with detailed timing and speaker information
- **TSV**: Tab-separated values for data analysis
- **LRC**: Synchronized lyrics format

## **Performance & Resource Requirements**

- **Memory**: Minimum 4GB RAM, recommended 8GB+ for larger files
- **CPU**: Multi-core recommended for faster processing
- **GPU**: Optional CUDA support for accelerated transcription
- **Storage**: Approximately 2-5GB for models (automatically downloaded)

## **Monitoring & Health Checks**

The service includes comprehensive monitoring:
- **Health Endpoint**: `http://localhost:8000/health`
- **Processing Metrics**: File count, processing time, error rates
- **Resource Monitoring**: CPU, memory, and disk usage
- **Structured Logging**: JSON-formatted logs for analysis

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.