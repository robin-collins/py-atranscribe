# **py-atranscribe: Automated Transcription & Diarization Service**

An automated, containerized Python application that continuously monitors a directory for new audio/video files and processes them through a robust transcription and speaker diarization pipeline.
Built with faster-whisper for high-performance transcription and pyannote.audio for accurate speaker diarization, this service is designed for "fire-and-forget" operation in a production environment.

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

* [Docker](https://www.google.com/search?q=httpss://www.docker.com/get-started) and [Docker Compose](https://www.google.com/search?q=httpss://docs.docker.com/compose/install/)
* Git

### **Installation & Setup**

1. **Clone the repository:**
   git clone \<your-repository-url\>
   cd py-atranscribe

2. Create the environment file:
   Copy the example environment file and edit it with your specific settings.
   cp .env.example .env

3. Set Your Hugging Face Token:
   Open the .env file and add your Hugging Face access token. This is required for speaker diarization. You can get a token from huggingface.co/settings/tokens.
   \# .env
   HF\_TOKEN=your\_huggingface\_token\_here

4. Create Input/Output Directories:
   The docker-compose setup expects a local ./audio directory.
   mkdir \-p ./audio/input ./audio/output ./audio/backup

5. Build and Run the Service:
   Use Docker Compose to build the image and run the container in detached mode.
   docker-compose up \--build \-d

The service is now running and monitoring the ./audio/input directory for new files.

## **Usage**

1. **Add Files**: Simply copy or move your audio or video files into the ./audio/input directory on your host machine.
2. **Processing**: The service will automatically detect the new file, wait for it to be fully written, and then begin the transcription and diarization process.
3. **Get Transcripts**: The output files (e.g., .srt, .vtt, .txt) will appear in the ./audio/output directory.
4. **Check Logs**: To view the real-time processing logs, run:
   docker-compose logs \-f

## **Configuration**

The application's behavior can be customized through the config.yaml file and overridden by environment variables defined in .env.

* **config.yaml**: Contains detailed settings for transcription models, diarization parameters, performance tuning, and more.
* **.env**: Used to set high-level configuration and secrets, such as the HF\_TOKEN. Environment variables follow a PARENT\_\_CHILD structure to override nested YAML keys (e.g., TRANSCRIPTION\_\_WHISPER\_\_MODEL\_SIZE=large-v3).

Refer to config.yaml for a full list of available options.

## **Health Check**

The service exposes a health check endpoint for monitoring, which is used by the Docker health check.

* **URL**: http://localhost:8000/health
* **Description**: Returns a 200 OK status if the service is running correctly and 503 Service Unavailable if there is an issue.

## **Development**

### **Local Setup**

1. **Create a virtual environment:**
   python \-m venv venv
   source venv/bin/activate  \# On Windows, use \`venv\\Scripts\\activate\`

2. **Install dependencies:**
   pip install \-r requirements.txt

3. **Run the application:**
   python auto\_diarize\_transcribe.py \--config config.yaml

### **Running Tests**

To run the comprehensive unit test suite:
pytest

To include coverage reporting:
pytest \--cov=src

## **License**

This project is licensed under the MIT License \- see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.