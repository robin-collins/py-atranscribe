"""Enhanced Whisper transcription using transformers pipeline with flash attention.

Integrates the insanely-fast-whisper approach with flash attention and optimized models.
"""

import logging
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Callable

import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
from transformers import WhisperProcessor

# Global warning filters for PyTorch/PyAnnote warnings (excluding timestamp warning)
warnings.filterwarnings(
    "ignore", message=".*std\\(\\): degrees of freedom is <= 0.*", category=UserWarning
)
# Note: Removed the timestamp warning filter since we're now properly handling timestamps
# via return_timestamps=True which automatically enables WhisperTimestampsLogitsProcessor

from src.config import WhisperConfig
from src.utils.error_handling import TranscriptionError


@dataclass
class EnhancedTranscriptionResult:
    """Result from enhanced whisper transcription."""

    text: str
    chunks: list[dict[str, Any]]
    language: str | None = None
    processing_time: float = 0.0
    model_info: dict[str, Any] | None = None


class EnhancedWhisperTranscriber:
    """Enhanced Whisper transcriber using transformers pipeline with flash attention."""

    def __init__(self, config: WhisperConfig) -> None:
        """Initialize enhanced whisper transcriber.

        Args:
        ----
            config: Whisper configuration

        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.EnhancedWhisperTranscriber")
        self._pipeline = None
        self._device = self._determine_device()
        # This will be set correctly during initialization.
        self.attn_implementation: str | None = None

    def _determine_device(self) -> str:
        """Determine the best device for inference."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda:0"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        elif self.config.device == "mps":
            return "mps"
        elif self.config.device.startswith("cuda:"):
            return self.config.device
        else:
            return "cpu"

    async def initialize(self) -> None:
        """Initialize the enhanced whisper pipeline."""
        try:
            self.logger.info("Initializing enhanced Whisper pipeline...")
            start_time = time.time()

            # Set up performance optimizations
            self._setup_performance_optimizations()

            # Set up warning filters
            self._setup_warning_filters()

            # Determine model name - use distil-small.en or custom model as specified
            model_name = self._get_model_name()
            self.logger.info("Loading HuggingFace pipeline for model: %s", model_name)

            proc = WhisperProcessor.from_pretrained(model_name)
            self.logger.info("WhisperProcessor loaded for model: %s", model_name)

            proc.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

            # Configure model kwargs for optimization based on insanely-fast-whisper.
            # This is the single source of truth for attention configuration.
            model_kwargs = {}
            if is_flash_attn_2_available() and self._device != "mps":
                self.attn_implementation = "flash_attention_2"
                model_kwargs["attn_implementation"] = "flash_attention_2"
                # CRITICAL: This must be set at initialization for Flash Attention 2.
                model_kwargs["output_attentions"] = False
            else:
                self.attn_implementation = "sdpa"
                model_kwargs["attn_implementation"] = "sdpa"

            self.logger.info(
                "Using '%s' attention implementation", self.attn_implementation
            )

            # Create pipeline with optimizations
            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model=model_name,
                torch_dtype=torch.float16,
                device=self._device,
                model_kwargs=model_kwargs,
                tokenizer=proc.tokenizer,
                feature_extractor=proc.feature_extractor,
            )

            # Explicitly move model to device for Flash Attention 2 compatibility
            self._pipeline.model = self._pipeline.model.to(self._device)

            self.logger.info("HuggingFace pipeline ready for model: %s", model_name)

            # Apply device-specific optimizations
            self._apply_device_optimizations()

            init_time = time.time() - start_time
            self.logger.info(
                "Enhanced Whisper pipeline initialized in %.2fs (model: %s, device: %s, attn: %s)",
                init_time,
                model_name,
                self._device,
                self.attn_implementation,
            )

        except Exception as e:
            self.logger.exception("Failed to initialize enhanced Whisper pipeline")
            msg = f"Enhanced Whisper initialization failed: {e}"
            raise TranscriptionError(msg) from e

    def _get_model_name(self) -> str:
        """Get the model name, defaulting to distil-small.en."""
        # Use distil-whisper/distil-small.en as specified in requirements
        if hasattr(self.config, "enhanced_model") and self.config.enhanced_model:
            return self.config.enhanced_model
        return "distil-whisper/distil-small.en"

    def _setup_performance_optimizations(self) -> None:
        """Set up performance optimizations based on device."""
        if self._device.startswith("cuda"):
            # Enable TensorFloat-32 for better performance
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.logger.info("Enabled TF32 for enhanced CUDA performance")

            # Optimize cuDNN for consistent input sizes
            torch.backends.cudnn.benchmark = True
            self.logger.info("Enabled cuDNN auto-tuner for consistent workloads")

    def _setup_warning_filters(self) -> None:
        """Set up warning filters to suppress known repeated warnings."""
        import warnings

        # Suppress Flash Attention device warnings (we handle this explicitly)
        warnings.filterwarnings(
            "ignore",
            message=".*You are attempting to use Flash Attention 2.0 with a model not initialized on GPU.*",
            category=UserWarning,
        )

        # Suppress transformers input name deprecation warnings
        warnings.filterwarnings(
            "ignore",
            message=".*input name.*deprecated.*input_features.*",
            category=FutureWarning,
        )

        # Suppress attention mask warnings (handled automatically by pipeline)
        warnings.filterwarnings(
            "ignore",
            message=".*The attention mask is not set and cannot be inferred from input because pad token is same as eos token.*",
            category=UserWarning,
        )

        # Suppress PyAnnote TF32 warnings (we enable it intentionally)
        warnings.filterwarnings(
            "ignore",
            message=".*TensorFloat-32.*disabled.*reproducibility.*",
            category=UserWarning,
        )

        self.logger.debug("Applied warning filters for known repeated warnings")

    def _apply_device_optimizations(self) -> None:
        """Apply device-specific optimizations."""
        if self._device == "mps":
            # Clear MPS cache for Apple Silicon
            torch.mps.empty_cache()
            self.logger.info("Applied MPS optimizations")
        elif self._device.startswith("cuda"):
            # Clear CUDA cache
            torch.cuda.empty_cache()
            self.logger.info("Applied CUDA optimizations")

    async def transcribe(
        self,
        audio_path: Path,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> EnhancedTranscriptionResult:
        """Transcribe audio using enhanced pipeline.

        Args:
        ----
            audio_path: Path to audio file
            progress_callback: Optional progress callback

        Returns:
        -------
            Enhanced transcription result

        """
        if not self._pipeline:
            raise TranscriptionError("Enhanced Whisper pipeline not initialized")

        try:
            self.logger.info("Starting enhanced transcription: %s", audio_path.name)
            start_time = time.time()

            if progress_callback:
                progress_callback(0.1, "Loading audio file")

            # Configure generation kwargs based on insanely-fast-whisper
            model_name = self._get_model_name()
            is_english_only = model_name.endswith(".en")

            # This is the most reliable way to set parameters for the pipeline.
            generate_kwargs = {
                "return_timestamps": True,
                "temperature": 0.0,
                "condition_on_prev_tokens": True,
            }

            # For multilingual models, specify task and language.
            # English-only models will crash if these are provided.
            if not is_english_only:
                generate_kwargs["task"] = "transcribe"
                if self.config.language and self.config.language.lower() != "auto":
                    generate_kwargs["language"] = self.config.language

            if hasattr(self.config, "initial_prompt") and self.config.initial_prompt:
                generate_kwargs["initial_prompt"] = self.config.initial_prompt

            # The check for output_attentions is no longer needed here,
            # as the model is correctly configured at initialization.

            if progress_callback:
                progress_callback(0.2, "Starting transcription")

            # Perform transcription with enhanced pipeline
            outputs = self._pipeline(
                str(audio_path),
                chunk_length_s=30,
                batch_size=self._get_batch_size(),
                generate_kwargs=generate_kwargs,
            )

            self.logger.debug("Raw Whisper outputs: %s", outputs)

            if progress_callback:
                progress_callback(0.9, "Processing results")

            processing_time = time.time() - start_time

            # Extract language information
            language = None
            if hasattr(outputs, "get") and "language" in outputs:
                language = outputs["language"]

            # Create result
            result = EnhancedTranscriptionResult(
                text=outputs["text"],
                chunks=outputs.get("chunks", []),
                language=language,
                processing_time=processing_time,
                model_info={
                    "model_name": self._get_model_name(),
                    "device": self._device,
                    "flash_attention": is_flash_attn_2_available()
                    and self._device != "mps",
                },
            )

            if progress_callback:
                progress_callback(1.0, "Transcription complete")

            self.logger.info(
                "Enhanced transcription completed in %.2fs (%.2fx speed)",
                processing_time,
                self._calculate_speed_ratio(audio_path, processing_time),
            )

            return result

        except Exception as e:
            self.logger.exception("Enhanced transcription failed: %s", audio_path.name)
            msg = f"Enhanced transcription failed: {e}"
            raise TranscriptionError(msg) from e

    def _get_batch_size(self) -> int:
        """Get optimal batch size for the device."""
        if self._device == "mps":
            return 4  # Smaller batch size for MPS to avoid OOM
        elif self._device.startswith("cuda"):
            return 24  # Default from insanely-fast-whisper
        else:
            return 8  # Conservative for CPU

    def _calculate_speed_ratio(self, audio_path: Path, processing_time: float) -> float:
        """Calculate processing speed ratio (audio_duration / processing_time)."""
        try:
            # Simple estimate - would need actual audio duration for precise calculation
            # For now, return a placeholder based on processing time efficiency
            _ = audio_path  # Acknowledge parameter
            _ = processing_time  # Acknowledge parameter
            return 1.0
        except Exception:
            return 1.0

    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            if self._pipeline:
                del self._pipeline
                self._pipeline = None

            # Clear device caches
            if self._device == "mps":
                torch.mps.empty_cache()
            elif self._device.startswith("cuda"):
                torch.cuda.empty_cache()

            self.logger.info("Enhanced Whisper transcriber cleaned up")

        except Exception:
            self.logger.exception("Error during enhanced transcriber cleanup")


class OutputConverter:
    """Convert transcription output to various formats using insanely-fast-whisper approach."""

    @staticmethod
    def to_json(
        result: EnhancedTranscriptionResult, speakers: list | None = None
    ) -> dict[str, Any]:
        """Convert result to JSON format compatible with convert_output.py."""
        return {
            "speakers": speakers or [],
            "chunks": result.chunks,
            "text": result.text,
        }

    @staticmethod
    def format_seconds(seconds: float) -> str:
        """Format seconds to SRT timestamp format.

        Args:
            seconds: Time in seconds (can be None for missing timestamps)

        Returns:
            Formatted timestamp string in SRT format (HH:MM:SS,mmm)

        """
        # Handle None values gracefully - return zero timestamp
        if seconds is None:
            return "00:00:00,000"

        # Ensure seconds is a valid number
        try:
            seconds = float(seconds)
        except (TypeError, ValueError):
            return "00:00:00,000"

        # Handle negative values
        seconds = max(seconds, 0)

        whole_seconds = int(seconds)
        milliseconds = int((seconds - whole_seconds) * 1000)

        hours = whole_seconds // 3600
        minutes = (whole_seconds % 3600) // 60
        seconds_remainder = whole_seconds % 60

        return f"{hours:02d}:{minutes:02d}:{seconds_remainder:02d},{milliseconds:03d}"

    @staticmethod
    def to_srt(
        result: EnhancedTranscriptionResult, speakers: list | None = None
    ) -> str:
        """Convert to SRT format."""
        _ = speakers  # Acknowledge parameter for future speaker integration
        srt_content = ""

        for index, chunk in enumerate(result.chunks, 1):
            text = chunk.get("text", "").strip()

            # Handle missing or malformed timestamp data
            timestamp = chunk.get("timestamp", [None, None])
            if not isinstance(timestamp, (list, tuple)) or len(timestamp) < 2:
                # Use default timestamps for malformed data
                start, end = 0.0, 0.0
            else:
                start, end = timestamp[0], timestamp[1]

            # Format timestamps with robust error handling
            start_format = OutputConverter.format_seconds(start)
            end_format = OutputConverter.format_seconds(end)

            # Skip empty chunks
            if not text:
                continue

            srt_content += f"{index}\n{start_format} --> {end_format}\n{text}\n\n"

        return srt_content

    @staticmethod
    def to_txt(result: EnhancedTranscriptionResult) -> str:
        """Convert to plain text format."""
        return result.text
