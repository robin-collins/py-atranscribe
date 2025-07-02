"""
Factory for creating Whisper inference instances.
Provides centralized model management and configuration.
"""

import gc
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from faster_whisper import WhisperModel

from ..config import WhisperConfig
from ..utils.error_handling import ModelError, graceful_degradation, retry_on_error


class WhisperFactory:
    """
    Factory class for creating and managing Whisper model instances.
    Implements model caching, device detection, and graceful degradation.
    """

    _instances: dict[str, WhisperModel] = {}
    _device_cache: str | None = None

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached model instances and free memory."""
        for model in cls._instances.values():
            try:
                # Force garbage collection to free GPU memory
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                logging.getLogger(__name__).warning(f"Error clearing model cache: {e}")

        cls._instances.clear()
        logging.getLogger(__name__).info("Whisper model cache cleared")

    @classmethod
    def get_optimal_device(cls) -> str:
        """
        Determine the optimal device for inference.

        Returns:
            str: Device string ("cuda", "cpu")
        """
        if cls._device_cache is not None:
            return cls._device_cache

        # Check CUDA availability and memory
        if torch.cuda.is_available():
            try:
                # Check if we have sufficient GPU memory (at least 2GB)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_gb = gpu_memory / (1024**3)

                if gpu_memory_gb >= 2.0:
                    cls._device_cache = "cuda"
                    logging.getLogger(__name__).info(
                        f"Using CUDA device with {gpu_memory_gb:.1f}GB memory"
                    )
                    return "cuda"
                else:
                    logging.getLogger(__name__).warning(
                        f"GPU has insufficient memory ({gpu_memory_gb:.1f}GB), using CPU"
                    )
            except Exception as e:
                logging.getLogger(__name__).warning(
                    f"Error checking GPU memory, using CPU: {e}"
                )

        cls._device_cache = "cpu"
        logging.getLogger(__name__).info("Using CPU device for inference")
        return "cpu"

    @classmethod
    def get_optimal_compute_type(cls, device: str, requested_type: str = "auto") -> str:
        """
        Determine the optimal compute type for the given device.

        Args:
            device: Target device ("cuda", "cpu")
            requested_type: Requested compute type ("auto", "int8", "int16", "float16", "float32")

        Returns:
            str: Optimal compute type
        """
        if requested_type != "auto":
            # Apply graceful degradation if needed
            return graceful_degradation.get_compute_type_fallback(requested_type)

        if device == "cuda":
            # Use float16 for GPU to save memory
            return graceful_degradation.get_compute_type_fallback("float16")
        else:
            # Use int8 for CPU for better performance
            return graceful_degradation.get_compute_type_fallback("int8")

    @retry_on_error()
    def create_whisper_inference(
        self, config: WhisperConfig
    ) -> "FasterWhisperInference":
        """
        Create a FasterWhisperInference instance with the given configuration.

        Args:
            config: Whisper configuration

        Returns:
            FasterWhisperInference: Configured inference instance

        Raises:
            ModelError: If model creation fails
        """
        try:
            # Apply graceful degradation to model size
            model_size = graceful_degradation.get_model_fallback(config.model_size)

            # Determine device
            if config.device == "auto":
                device = WhisperFactory.get_optimal_device()
            else:
                device = config.device

            # Determine compute type
            compute_type = WhisperFactory.get_optimal_compute_type(
                device, config.compute_type
            )

            # Create cache key
            cache_key = f"{model_size}_{device}_{compute_type}"

            # Check if model is already cached
            if cache_key in self._instances:
                self.logger.debug(f"Using cached Whisper model: {cache_key}")
                model = self._instances[cache_key]
            else:
                self.logger.info(f"Creating new Whisper model: {cache_key}")

                # Create model with optimal settings
                model_kwargs = {
                    "model_size_or_path": model_size,
                    "device": device,
                    "compute_type": compute_type,
                    "cpu_threads": config.cpu_threads
                    if config.cpu_threads > 0
                    else os.cpu_count(),
                    "num_workers": config.num_workers,
                }

                # Add GPU-specific optimizations
                if device == "cuda":
                    model_kwargs.update(
                        {
                            "device_index": 0,  # Use first GPU
                        }
                    )

                try:
                    model = WhisperModel(**model_kwargs)
                    self._instances[cache_key] = model
                    self.logger.info(f"Successfully created Whisper model: {cache_key}")

                except Exception as e:
                    # Try fallback options on failure
                    if device == "cuda" and "out of memory" in str(e).lower():
                        self.logger.warning("GPU out of memory, falling back to CPU")
                        graceful_degradation.increase_degradation()

                        # Retry with CPU
                        model_kwargs["device"] = "cpu"
                        model_kwargs["compute_type"] = "int8"
                        cache_key = f"{model_size}_cpu_int8"

                        if cache_key not in self._instances:
                            model = WhisperModel(**model_kwargs)
                            self._instances[cache_key] = model
                            self.logger.info(
                                f"Successfully created fallback model: {cache_key}"
                            )
                        else:
                            model = self._instances[cache_key]
                    else:
                        raise ModelError(
                            f"Failed to create Whisper model: {e}", model_size
                        )

            return FasterWhisperInference(model, config)

        except Exception as e:
            self.logger.error(f"Error creating Whisper inference: {e}")
            raise ModelError(f"Failed to create Whisper inference: {e}")


class FasterWhisperInference:
    """
    Wrapper class for faster-whisper model with optimized inference settings.
    Provides transcription functionality with performance optimizations.
    """

    def __init__(self, model: WhisperModel, config: WhisperConfig):
        """
        Initialize FasterWhisperInference.

        Args:
            model: WhisperModel instance
            config: Whisper configuration
        """
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)

    @retry_on_error()
    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        initial_prompt: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Transcribe audio file to text with timestamps.

        Args:
            audio_path: Path to audio file
            language: Language code (None for auto-detection)
            initial_prompt: Initial prompt to guide transcription
            **kwargs: Additional transcription parameters

        Returns:
            Dict containing transcription results with segments and metadata

        Raises:
            AudioProcessingError: If transcription fails
        """
        try:
            self.logger.info(f"Starting transcription: {audio_path}")

            # Set default transcription parameters
            transcribe_params = {
                "language": language,
                "initial_prompt": initial_prompt,
                "beam_size": 5,  # Good balance of quality and speed
                "best_of": 5,  # Number of candidates for beam search
                "temperature": 0.0,  # Deterministic output
                "condition_on_previous_text": True,  # Use context from previous segments
                "compression_ratio_threshold": 2.4,  # Filter out segments with low compression
                "log_prob_threshold": -1.0,  # Filter out segments with low probability
                "no_speech_threshold": 0.6,  # Filter out segments without speech
                "word_timestamps": True,  # Enable word-level timestamps
                "prepend_punctuations": "\"'([{-",
                "append_punctuations": "\"'.,:)]}",
            }

            # Override with any provided parameters
            transcribe_params.update(kwargs)

            # Perform transcription
            segments, info = self.model.transcribe(audio_path, **transcribe_params)

            # Convert segments generator to list and extract information
            segments_list = []
            total_duration = 0.0

            for segment in segments:
                segment_dict = {
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "words": [],
                    "avg_logprob": segment.avg_logprob,
                    "no_speech_prob": segment.no_speech_prob,
                    "compression_ratio": segment.compression_ratio,
                }

                # Add word-level information if available
                if hasattr(segment, "words") and segment.words:
                    for word in segment.words:
                        word_dict = {
                            "start": word.start,
                            "end": word.end,
                            "word": word.word,
                            "probability": word.probability,
                        }
                        segment_dict["words"].append(word_dict)

                segments_list.append(segment_dict)
                total_duration = max(total_duration, segment.end)

            # Prepare result dictionary
            result = {
                "segments": segments_list,
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "duration_after_vad": getattr(
                    info, "duration_after_vad", info.duration
                ),
                "transcription_info": {
                    "model_size": self.config.model_size,
                    "compute_type": transcribe_params.get("compute_type", "auto"),
                    "device": self.model.device,
                    "beam_size": transcribe_params["beam_size"],
                    "temperature": transcribe_params["temperature"],
                },
                "text": " ".join(segment["text"] for segment in segments_list),
            }

            self.logger.info(
                f"Transcription completed: {len(segments_list)} segments, "
                f"{total_duration:.2f}s duration, language: {info.language}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Transcription failed for {audio_path}: {e}")
            from ..utils.error_handling import AudioProcessingError

            raise AudioProcessingError(f"Transcription failed: {e}", audio_path)

    def detect_language(self, audio_path: str) -> dict[str, Any]:
        """
        Detect the language of an audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Dict containing detected language and confidence
        """
        try:
            self.logger.debug(f"Detecting language for: {audio_path}")

            # Use the model's language detection
            segments, info = self.model.transcribe(
                audio_path,
                language=None,  # Auto-detect
                beam_size=1,  # Fast detection
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
                word_timestamps=False,
            )

            # Consume first segment to trigger language detection
            try:
                next(segments)
            except StopIteration:
                pass

            result = {
                "language": info.language,
                "language_probability": info.language_probability,
                "all_language_probs": getattr(info, "all_language_probs", {}),
            }

            self.logger.debug(
                f"Language detected: {info.language} "
                f"(confidence: {info.language_probability:.3f})"
            )

            return result

        except Exception as e:
            self.logger.error(f"Language detection failed for {audio_path}: {e}")
            from ..utils.error_handling import AudioProcessingError

            raise AudioProcessingError(f"Language detection failed: {e}", audio_path)

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_size": self.config.model_size,
            "device": self.model.device,
            "compute_type": getattr(self.model, "compute_type", "unknown"),
            "cpu_threads": self.config.cpu_threads,
            "num_workers": self.config.num_workers,
        }
