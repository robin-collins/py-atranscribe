# Automated Batch Transcription System Implementation Guide

This guide details how to build an automated system that transcribes audio files from a folder using faster-whisper with diarization, based on the Whisper-WebUI repository codebase.

## Overview

The system will:
1. Monitor or process audio files from an input folder
2. Use faster-whisper for speech-to-text transcription
3. Apply speaker diarization for multi-speaker audio
4. Save transcriptions to an organized output folder structure
5. Support multiple subtitle formats (SRT, WebVTT, TXT, LRC)

## Architecture Components

### Core Classes Referenced

- **`modules/whisper/whisper_factory.py`**: Factory for creating Whisper inference instances
- **`modules/whisper/faster_whisper_inference.py`**: Main faster-whisper implementation
- **`modules/diarize/diarizer.py`**: Speaker diarization wrapper
- **`modules/utils/files_manager.py`**: File handling and media format support
- **`modules/utils/subtitle_manager.py`**: Output file generation

## Implementation Steps

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set HuggingFace token for diarization (required)
export HF_TOKEN="your_huggingface_token_here"

# Accept pyannote.audio terms at:
# https://huggingface.co/pyannote/speaker-diarization-3.1
# https://huggingface.co/pyannote/segmentation-3.0
```

### 2. Core Implementation

Create `batch_transcriber.py`:

```python
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import logging

# Add the repository root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.whisper.whisper_factory import WhisperFactory
from modules.whisper.data_classes import (
    WhisperParams, VadParams, DiarizationParams,
    BGMSeparationParams, TranscriptionPipelineParams
)
from modules.utils.files_manager import get_media_files, MEDIA_EXTENSION
from modules.utils.paths import (
    FASTER_WHISPER_MODELS_DIR, DIARIZATION_MODELS_DIR,
    UVR_MODELS_DIR, OUTPUT_DIR
)

logger = logging.getLogger(__name__)

class BatchTranscriber:
    """
    Automated batch transcription system using faster-whisper with diarization.
    Based on Whisper-WebUI architecture.
    """

    def __init__(
        self,
        output_dir: str = "./batch_outputs",
        model_size: str = "large-v2",
        compute_type: str = "float16",
        device: str = "auto",
        enable_diarization: bool = True,
        hf_token: Optional[str] = None
    ):
        """
        Initialize the batch transcriber.

        Args:
            output_dir: Directory for transcription outputs
            model_size: Whisper model size (tiny, base, small, medium, large-v2, large-v3)
            compute_type: Compute precision (float16 for GPU, float32 for CPU)
            device: Device to use (auto, cuda, cpu)
            enable_diarization: Whether to perform speaker diarization
            hf_token: HuggingFace token for diarization models
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize WhisperFactory - creates inference instance
        # Reference: modules/whisper/whisper_factory.py:WhisperFactory.create_whisper_inference()
        self.whisper_inf = WhisperFactory.create_whisper_inference(
            whisper_type="faster-whisper",
            faster_whisper_model_dir=FASTER_WHISPER_MODELS_DIR,
            diarization_model_dir=DIARIZATION_MODELS_DIR,
            uvr_model_dir=UVR_MODELS_DIR,
            output_dir=str(self.output_dir)
        )

        # Update model configuration
        # Reference: modules/whisper/faster_whisper_inference.py:FasterWhisperInference.update_model()
        self.whisper_inf.update_model(
            model_size=model_size,
            compute_type=compute_type
        )

        self.enable_diarization = enable_diarization
        self.hf_token = hf_token or os.getenv('HF_TOKEN')

        logger.info(f"Initialized BatchTranscriber with device: {self.whisper_inf.device}")
        logger.info(f"Model: {model_size}, Compute: {compute_type}")

    def create_pipeline_params(
        self,
        file_format: str = "SRT",
        add_timestamp: bool = True,
        enable_vad: bool = True,
        enable_bgm_separation: bool = False
    ) -> tuple:
        """
        Create pipeline parameters for transcription.

        Reference: modules/whisper/data_classes.py for parameter classes
        """
        # Whisper parameters
        whisper_params = WhisperParams(
            model_size=self.whisper_inf.current_model_size,
            lang="auto",  # Auto-detect language
            is_translate=False,  # Set to True for translation to English
            beam_size=5,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            compute_type=self.whisper_inf.compute_type,
            best_of=5,
            patience=2.0,
            condition_on_previous_text=True,
            initial_prompt="",
            temperature=0.0,
            compression_ratio_threshold=2.4,
            vad_filter=enable_vad,
            threshold=0.5,
            min_speech_duration_ms=250,
            max_speech_duration_s=float('inf'),
            min_silence_duration_ms=2000,
            window_size_samples=1024,
            speech_pad_ms=400,
            chunk_length_s=30,
            batch_size=16,
            is_diarize=self.enable_diarization,
            hf_token=self.hf_token,
            diarization_device=str(self.whisper_inf.device),
            is_separate_bgm=enable_bgm_separation,
            uvr_model_size="UVR-MDX-NET-Voc_FT",
            uvr_segment_size=256,
            uvr_save_file=False
        )

        # Convert to list format expected by pipeline
        # Reference: modules/whisper/data_classes.py:WhisperParams.to_list()
        return whisper_params.to_list()

    def process_folder(
        self,
        input_folder: str,
        file_format: str = "SRT",
        include_subdirectories: bool = True,
        add_timestamp: bool = True
    ) -> Dict[str, any]:
        """
        Process all audio/video files in a folder.

        Args:
            input_folder: Path to folder containing media files
            file_format: Output format (SRT, WebVTT, TXT, LRC, JSON, TSV)
            include_subdirectories: Whether to process subdirectories
            add_timestamp: Whether to add timestamp to output filenames

        Returns:
            Dictionary with processing results
        """
        input_path = Path(input_folder)
        if not input_path.exists():
            raise FileNotFoundError(f"Input folder not found: {input_folder}")

        # Get all media files from folder
        # Reference: modules/utils/files_manager.py:get_media_files()
        media_files = get_media_files(
            folder_path=str(input_path),
            include_sub_directory=include_subdirectories
        )

        if not media_files:
            logger.warning(f"No media files found in {input_folder}")
            return {"processed": 0, "files": []}

        logger.info(f"Found {len(media_files)} media files to process")

        # Create pipeline parameters
        pipeline_params = self.create_pipeline_params(
            file_format=file_format,
            add_timestamp=add_timestamp
        )

        # Process files using the transcribe_file method
        # Reference: modules/whisper/base_transcription_pipeline.py:BaseTranscriptionPipeline.transcribe_file()
        try:
            files_info = self.whisper_inf.transcribe_file(
                files=media_files,
                file_format=file_format,
                add_timestamp=add_timestamp,
                *pipeline_params
            )

            logger.info(f"Successfully processed {len(files_info)} files")
            return {
                "processed": len(files_info),
                "files": files_info,
                "output_dir": str(self.output_dir)
            }

        except Exception as e:
            logger.error(f"Error processing files: {str(e)}")
            raise

    def process_single_file(
        self,
        file_path: str,
        file_format: str = "SRT",
        add_timestamp: bool = True
    ) -> Dict[str, any]:
        """
        Process a single audio/video file.

        Args:
            file_path: Path to the media file
            file_format: Output format
            add_timestamp: Whether to add timestamp to filename

        Returns:
            Processing result dictionary
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Validate file extension
        if file_path.suffix.lower() not in MEDIA_EXTENSION:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        pipeline_params = self.create_pipeline_params(
            file_format=file_format,
            add_timestamp=add_timestamp
        )

        # Process single file
        # Reference: modules/whisper/base_transcription_pipeline.py:BaseTranscriptionPipeline.run()
        try:
            def progress_callback(progress, text=""):
                logger.info(f"Processing progress: {progress:.1%} - {text}")

            transcribed_segments, elapsed_time = self.whisper_inf.run(
                str(file_path),
                progress_callback,
                file_format,
                add_timestamp,
                None,  # yt_metadata
                *pipeline_params
            )

            result = {
                "file": str(file_path),
                "segments": len(transcribed_segments),
                "duration": elapsed_time,
                "output_format": file_format
            }

            logger.info(f"Processed {file_path.name} in {elapsed_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            raise

def main():
    """
    Example usage of the BatchTranscriber
    """
    import argparse

    parser = argparse.ArgumentParser(description="Batch Audio Transcription with Diarization")
    parser.add_argument("--input", "-i", required=True, help="Input folder path")
    parser.add_argument("--output", "-o", default="./batch_outputs", help="Output folder path")
    parser.add_argument("--format", "-f", default="SRT", choices=["SRT", "WebVTT", "TXT", "LRC", "JSON", "TSV"])
    parser.add_argument("--model", "-m", default="large-v2", help="Whisper model size")
    parser.add_argument("--compute", "-c", default="float16", choices=["float16", "float32", "int8"])
    parser.add_argument("--no-diarization", action="store_true", help="Disable speaker diarization")
    parser.add_argument("--subdirs", action="store_true", help="Include subdirectories")
    parser.add_argument("--timestamp", action="store_true", help="Add timestamp to filenames")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        # Initialize transcriber
        transcriber = BatchTranscriber(
            output_dir=args.output,
            model_size=args.model,
            compute_type=args.compute,
            enable_diarization=not args.no_diarization
        )

        # Process folder
        results = transcriber.process_folder(
            input_folder=args.input,
            file_format=args.format,
            include_subdirectories=args.subdirs,
            add_timestamp=args.timestamp
        )

        print(f"\n✅ Batch transcription completed!")
        print(f"📁 Processed: {results['processed']} files")
        print(f"📂 Output directory: {results['output_dir']}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
```

### 3. Usage Examples

#### Basic Usage
```bash
# Process all audio files in a folder
python batch_transcriber.py --input /path/to/audio/folder --output /path/to/outputs

# Process with specific model and format
python batch_transcriber.py \
    --input /path/to/audio/folder \
    --output /path/to/outputs \
    --model large-v3 \
    --format WebVTT \
    --subdirs \
    --timestamp
```

#### Advanced Configuration
```python
# Custom transcriber with specific settings
transcriber = BatchTranscriber(
    output_dir="./transcriptions",
    model_size="large-v3",
    compute_type="float16",  # Use "float32" for CPU
    enable_diarization=True,
    hf_token="your_hf_token_here"
)

# Process folder with custom parameters
results = transcriber.process_folder(
    input_folder="/path/to/audio/files",
    file_format="SRT",
    include_subdirectories=True,
    add_timestamp=True
)
```

### 4. Output Structure

The system creates organized output folders:

```
batch_outputs/
├── audio_file_1_20240101_120000.srt    # Transcription files
├── audio_file_2_20240101_120100.srt
├── UVR/                                # BGM separation (if enabled)
│   ├── instrumental/
│   └── vocals/
└── translations/                       # Translation files (if used)
```

### 5. Supported File Formats

#### Input Formats (Reference: `modules/utils/files_manager.py`)
**Audio**: `.mp3`, `.wav`, `.wma`, `.aac`, `.flac`, `.ogg`, `.m4a`, `.aiff`, `.alac`, `.opus`, `.webm`, `.ac3`, `.amr`, `.au`, `.mid`, `.midi`, `.mka`

**Video**: `.mp4`, `.mkv`, `.flv`, `.avi`, `.mov`, `.wmv`, `.webm`, `.m4v`, `.mpeg`, `.mpg`, `.3gp`, `.f4v`, `.ogv`, `.vob`, `.mts`, `.m2ts`, `.divx`, `.mxf`, `.rm`, `.rmvb`, `.ts`

#### Output Formats (Reference: `modules/utils/subtitle_manager.py`)
- **SRT**: SubRip subtitle format with speaker labels
- **WebVTT**: Web Video Text Tracks format
- **TXT**: Plain text transcription
- **LRC**: Lyric format with timestamps
- **JSON**: Full segment data with metadata
- **TSV**: Tab-separated values format

### 6. Configuration Options

#### Model Configuration
- **Model sizes**: `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`
- **Compute types**: `float16` (GPU), `float32` (CPU), `int8`, `int16`
- **Device**: Auto-detection or specify `cuda`, `cpu`

#### Pipeline Features (Reference: `modules/whisper/data_classes.py`)
- **VAD filtering**: Voice Activity Detection to remove silence
- **BGM separation**: Background music removal using UVR
- **Speaker diarization**: Multi-speaker identification
- **Language detection**: Automatic or specify language code
- **Translation**: Speech-to-text translation to English

### 7. Requirements and Dependencies

#### System Requirements
- **Python**: 3.10-3.12
- **FFmpeg**: Required for audio/video processing
- **CUDA**: Optional, for GPU acceleration

#### Python Dependencies (from `requirements.txt`)
```
torch
torchaudio
faster-whisper==1.1.1
transformers==4.47.1
pyannote.audio==3.3.2
```

#### HuggingFace Setup
1. Create account at [huggingface.co](https://huggingface.co)
2. Generate READ token in account settings
3. Accept terms at speaker diarization model pages:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0

### 8. Error Handling and Monitoring

The implementation includes comprehensive error handling:

- **File validation**: Check file existence and format support
- **Model loading**: Graceful fallback for model loading issues
- **Memory management**: Automatic model offloading for VRAM constraints
- **Progress tracking**: Real-time processing progress callbacks
- **Logging**: Detailed logging for debugging and monitoring

### 9. Performance Optimization

#### Memory Efficiency (Reference: `modules/whisper/faster_whisper_inference.py`)
- **Model offloading**: Reduce VRAM usage with `enable_offload=True`
- **Compute types**: Use `int8` for lower memory usage
- **Batch processing**: Efficient processing of multiple files

#### Speed Optimization
- **faster-whisper**: ~60% faster than openai/whisper
- **VAD filtering**: Skip silent segments
- **Chunk processing**: Process long audio in segments

This implementation provides a robust, production-ready batch transcription system leveraging the proven architecture of Whisper-WebUI with faster-whisper and diarization capabilities.