"""Factory for creating Whisper inference instances.

Provides centralized model management and configuration.

"""

import contextlib
import gc
import logging
import os
from typing import Any, ClassVar

import torch
from faster_whisper import WhisperModel

from src.config import WhisperConfig
from src.utils.error_handling import (
    AudioProcessingError,
    ModelError,
    graceful_degradation,
    retry_on_error,
)


class WhisperFactory:
    """Factory class for creating and managing Whisper model instances.

    Implements model caching, device detection, and graceful degradation.

    """

    _instances: ClassVar[dict[str, WhisperModel]] = {}
    _device_cache: ClassVar[str | None] = None

    def __init__(self) -> None:
        """Initialize WhisperFactory."""
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
            except (RuntimeError, AttributeError) as e:
                logging.getLogger(__name__).warning("Error clearing model cache: %s", e)

        cls._instances.clear()
        logging.getLogger(__name__).info("Whisper model cache cleared")

    @classmethod
    def get_optimal_device(cls) -> str:
        """Determine the optimal device for inference.

        Returns
        -------
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

                gpu_memory_threshold = 2.0

                if gpu_memory_gb >= gpu_memory_threshold:
                    cls._device_cache = "cuda"
                    logging.getLogger(__name__).info(
                        "Using CUDA device with %.1fGB memory",
                        gpu_memory_gb,
                    )
                    return "cuda"
                logging.getLogger(__name__).warning(
                    "GPU has insufficient memory (%.1fGB), using CPU",
                    gpu_memory_gb,
                )
            except RuntimeError as e:
                logging.getLogger(__name__).warning(
                    "Error checking GPU memory, using CPU: %s",
                    e,
                )
            else:
                return "cpu"

        cls._device_cache = "cpu"
        logging.getLogger(__name__).info("Using CPU device for inference")
        return "cpu"

    @classmethod
    def get_optimal_compute_type(cls, device: str, requested_type: str = "auto") -> str:
        """Determine the optimal compute type for the given device.

        Args:
        ----
            device: Target device ("cuda", "cpu")
            requested_type: Requested compute type ("auto", "int8", "int16", "float16", "float32")

        Returns:
        -------
            str: Optimal compute type

        """
        if requested_type != "auto":
            # Apply graceful degradation if needed
            return graceful_degradation.get_compute_type_fallback(requested_type)

        if device == "cuda":
            # Use float16 for GPU to save memory
            return graceful_degradation.get_compute_type_fallback("float16")
        # Use int8 for CPU for better performance
        return graceful_degradation.get_compute_type_fallback("int8")

    @retry_on_error()
    def create_whisper_inference(
        self,
        config: WhisperConfig,
    ) -> "FasterWhisperInference":
        """Create a FasterWhisperInference instance with the given configuration.

        Args:
        ----
            config: Whisper configuration

        Returns:
        -------
            FasterWhisperInference: Configured inference instance

        Raises:
        ------
            ModelError: If model creation fails

        """
        try:
            # Apply graceful degradation to model size
            model_size = graceful_degradation.get_model_fallback(config.model_size)

            # Determine device with enhanced error handling
            if config.device == "auto":
                device = WhisperFactory.get_optimal_device()
            else:
                device = config.device

            # Determine compute type
            compute_type = WhisperFactory.get_optimal_compute_type(
                device,
                config.compute_type,
            )

            # Create cache key
            cache_key = f"{model_size}_{device}_{compute_type}"

            # Check if model is already cached
            if cache_key in self._instances:
                self.logger.debug("Using cached Whisper model: %s", cache_key)
                model = self._instances[cache_key]
            else:
                self.logger.info("Creating new Whisper model: %s", cache_key)

                # Create model with comprehensive error handling
                model = self._create_model_with_fallback(
                    model_size,
                    device,
                    compute_type,
                    config,
                )

                if model:
                    self._instances[cache_key] = model
                    self.logger.info(
                        "Successfully created Whisper model: %s",
                        cache_key,
                    )
                else:
                    self._raise_model_creation_error()

            # Wrap in inference class with error handling
            return FasterWhisperInference(model, config, self.logger)

        except Exception as e:
            self.logger.exception("Failed to create Whisper inference: %s", e)
            msg = f"Whisper inference creation failed: {e}"
            raise ModelError(msg) from e

    def _raise_model_creation_error(self) -> None:
        """Raise ModelError for failed model creation."""
        msg = "Failed to create Whisper model after all fallback attempts"
        raise ModelError(msg)

    def _create_model_with_fallback(
        self,
        model_size: str,
        device: str,
        compute_type: str,
        config: WhisperConfig,
    ) -> WhisperModel | None:
        """Create WhisperModel with comprehensive fallback handling.

        Args:
        ----
            model_size: Model size to load
            device: Target device
            compute_type: Compute type
            config: Whisper configuration

        Returns:
        -------
            WhisperModel instance or None if all attempts fail

        """
        # Build fallback attempt configurations
        fallback_attempts = self._build_fallback_attempts(device, compute_type, config)

        # Try each fallback configuration
        for attempt_device, attempt_compute_type in fallback_attempts:
            model = self._attempt_model_creation(
                model_size, attempt_device, attempt_compute_type, config
            )
            if model:
                # Check if we had to fallback and update degradation
                if (attempt_device, attempt_compute_type) != (device, compute_type):
                    graceful_degradation.increase_degradation()
                    self.logger.warning(
                        "Model creation required fallback from %s/%s to %s/%s",
                        device,
                        compute_type,
                        attempt_device,
                        attempt_compute_type,
                    )
                return model

        # All attempts failed
        self.logger.error("All model creation attempts failed")
        return None

    def _build_fallback_attempts(
        self, device: str, compute_type: str, config: WhisperConfig
    ) -> list[tuple[str, str]]:
        """Build list of fallback attempts based on device type."""
        # Define fallback chain - prioritize GPU configurations for CUDA users
        fallback_attempts = [
            (device, compute_type),  # Original request
        ]

        # Add fallbacks based on device type - be more persistent with GPU
        if device == "cuda":
            # Try multiple CUDA configurations before falling back to CPU
            fallback_attempts.extend(
                [
                    ("cuda", "float32"),  # Try float32 (often more stable than float16)
                    ("cuda", "float16"),  # Try float16 for memory efficiency
                    ("cuda", "int8"),  # Try int8 for memory-constrained GPUs
                ],
            )

            # Only add CPU fallback if user didn't explicitly configure CUDA
            if config.device == "auto" and config.compute_type == "auto":
                fallback_attempts.extend(
                    [
                        ("cpu", "int8"),  # CPU fallback only for auto config
                        ("cpu", "float32"),  # CPU with float32
                    ],
                )
            else:
                self.logger.info(
                    "User explicitly configured CUDA - skipping CPU fallback",
                )

        elif device == "cpu":
            fallback_attempts.extend(
                [
                    ("cpu", "float32"),  # Try float32 if int8 fails
                    ("cpu", "int16"),  # Try int16
                ],
            )
        else:
            # For other devices (mps, xpu), add CPU fallback
            fallback_attempts.append(("cpu", "int8"))

        # Remove duplicates while preserving order
        seen = set()
        unique_attempts = []
        for attempt in fallback_attempts:
            if attempt not in seen:
                seen.add(attempt)
                unique_attempts.append(attempt)

        return unique_attempts

    def _attempt_model_creation(
        self, model_size: str, device: str, compute_type: str, config: WhisperConfig
    ) -> WhisperModel | None:
        """Attempt to create model with specific configuration."""
        try:
            self.logger.info(
                "Attempting to create model with device=%s, compute_type=%s",
                device,
                compute_type,
            )

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
                model_kwargs.update({"device_index": 0})  # Use first GPU

            # Create the model with cuDNN error handling
            model = self._create_model_with_cudnn_handling(
                model_kwargs, device, compute_type
            )

            # Test the model with a simple operation to catch runtime cuDNN errors
            self._test_model_runtime(model, device)

            self.logger.info(
                "Successfully created and tested model with device=%s, compute_type=%s",
                device,
                compute_type,
            )

        except (RuntimeError, OSError, ImportError, ValueError) as e:
            self._log_model_creation_error(e, device, compute_type)
            return None
        else:
            return model

    def _log_model_creation_error(
        self, error: Exception, device: str, compute_type: str
    ) -> None:
        """Log model creation error with specific error type information."""
        error_msg = str(error).lower()

        # Log specific error types for debugging
        if "libcudnn" in error_msg or "cudnn" in error_msg:
            self.logger.warning(
                "cuDNN library error with device=%s, compute_type=%s: %s",
                device,
                compute_type,
                error,
            )
        elif "out of memory" in error_msg:
            self.logger.warning(
                "GPU memory error with device=%s, compute_type=%s: %s",
                device,
                compute_type,
                error,
            )
        elif "cuda" in error_msg and "not available" in error_msg:
            self.logger.warning(
                "CUDA not available with device=%s, compute_type=%s: %s",
                device,
                compute_type,
                error,
            )
        else:
            self.logger.warning(
                "Model creation failed with device=%s, compute_type=%s: %s",
                device,
                compute_type,
                error,
            )

    def _create_model_with_cudnn_handling(
        self,
        model_kwargs: dict,
        device: str,
        compute_type: str,
    ) -> WhisperModel:
        """Create WhisperModel with specific cuDNN error handling.

        Args:
        ----
            model_kwargs: Model creation arguments
            device: Target device
            compute_type: Compute type

        Returns:
        -------
            WhisperModel instance

        Raises:
        ------
            Exception: If model creation fails

        """
        try:
            # Announce model download start
            model_name = model_kwargs.get("model_size_or_path", "unknown")
            self.logger.info(
                "🔽 Downloading Whisper model '%s' (device: %s, compute: %s)...",
                model_name,
                device,
                compute_type,
            )
            print(
                f"🔽 Downloading Whisper model '{model_name}' (device: {device}, compute: {compute_type})...",
                flush=True,
            )

            # First attempt: Normal model creation
            model = WhisperModel(**model_kwargs)

        except Exception as e:
            error_msg = str(e).lower()

            # Handle cuDNN-specific errors
            if "libcudnn" in error_msg or "cudnn" in error_msg:
                self.logger.warning(
                    "cuDNN error detected, trying cuDNN workarounds: %s",
                    e,
                )

                # Try cuDNN workarounds in sequence
                model = self._try_cudnn_workarounds(model_kwargs)
                if model:
                    return model

                model = self._try_cudnn_disabled(model_kwargs)
                if model:
                    return model

                model = self._try_cpu_fallback_for_cudnn(model_kwargs)
                if model:
                    return model

            # Re-raise the original exception if cuDNN workarounds didn't help
            raise
        else:
            # Announce completion
            self.logger.info(
                "✅ Whisper model '%s' download completed successfully", model_name
            )
            print(
                f"✅ Whisper model '{model_name}' download completed successfully",
                flush=True,
            )

            return model

    def _try_cudnn_workarounds(self, model_kwargs: dict) -> WhisperModel | None:
        """Try cuDNN workarounds with environment variables."""
        import os

        original_env = {}

        try:
            # Save original environment
            cudnn_env_vars = [
                "CUDNN_LOGINFO_DBG",
                "CUDNN_LOGERR_DBG",
                "CUDNN_LOGWARN_DBG",
                "CUDA_LAUNCH_BLOCKING",
                "PYTORCH_CUDA_ALLOC_CONF",
            ]
            for var in cudnn_env_vars:
                if var in os.environ:
                    original_env[var] = os.environ[var]

            # Set cuDNN debugging and optimization environment variables
            os.environ["CUDNN_LOGINFO_DBG"] = "0"
            os.environ["CUDNN_LOGERR_DBG"] = "1"
            os.environ["CUDNN_LOGWARN_DBG"] = "1"
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
                "max_split_size_mb:512,expandable_segments:True"
            )

            self.logger.info(
                "Retrying model creation with cuDNN environment variables",
            )

            # Announce retry attempt
            model_name = model_kwargs.get("model_size_or_path", "unknown")
            print(
                f"🔄 Retrying Whisper model '{model_name}' download with cuDNN workarounds...",
                flush=True,
            )

            model = WhisperModel(**model_kwargs)

            self.logger.info(
                "Successfully created model with cuDNN workarounds",
            )
            print(
                f"✅ Whisper model '{model_name}' download completed with cuDNN workarounds",
                flush=True,
            )

        except (RuntimeError, OSError, ImportError, ValueError) as cudnn_retry_error:
            self.logger.warning(
                "cuDNN workaround failed: %s",
                cudnn_retry_error,
            )
            return None
        else:
            return model

        finally:
            # Restore original environment
            cudnn_env_vars = [
                "CUDNN_LOGINFO_DBG",
                "CUDNN_LOGERR_DBG",
                "CUDNN_LOGWARN_DBG",
                "CUDA_LAUNCH_BLOCKING",
                "PYTORCH_CUDA_ALLOC_CONF",
            ]
            for var in cudnn_env_vars:
                if var in original_env:
                    os.environ[var] = original_env[var]
                elif var in os.environ:
                    del os.environ[var]

    def _try_cudnn_disabled(self, model_kwargs: dict) -> WhisperModel | None:
        """Try creating model with cuDNN disabled."""
        try:
            self.logger.info("Trying to create model with cuDNN disabled")

            # Temporarily disable cuDNN
            import torch

            original_cudnn_enabled = torch.backends.cudnn.enabled
            torch.backends.cudnn.enabled = False

            try:
                # Announce attempt
                model_name = model_kwargs.get("model_size_or_path", "unknown")
                print(
                    f"🔄 Retrying Whisper model '{model_name}' download with cuDNN disabled...",
                    flush=True,
                )

                model = WhisperModel(**model_kwargs)
                self.logger.warning(
                    "Successfully created model with cuDNN disabled - "
                    "performance may be reduced",
                )
                print(
                    f"✅ Whisper model '{model_name}' download completed with cuDNN disabled",
                    flush=True,
                )
                return model
            finally:
                # Restore cuDNN setting
                torch.backends.cudnn.enabled = original_cudnn_enabled

        except (RuntimeError, OSError, ImportError, ValueError) as no_cudnn_error:
            self.logger.warning(
                "Model creation failed even with cuDNN disabled: %s",
                no_cudnn_error,
            )
            return None

    def _try_cpu_fallback_for_cudnn(self, model_kwargs: dict) -> WhisperModel | None:
        """Try CPU fallback as last resort for cuDNN errors."""
        try:
            self.logger.info(
                "cuDNN issues detected, trying CPU fallback as last resort",
            )
            cpu_kwargs = model_kwargs.copy()
            cpu_kwargs.update(
                {
                    "device": "cpu",
                    "compute_type": "int8",  # Use int8 for better CPU performance
                },
            )

            # Announce CPU fallback attempt
            model_name = model_kwargs.get("model_size_or_path", "unknown")
            print(
                f"🔄 Falling back to CPU for Whisper model '{model_name}' due to cuDNN issues...",
                flush=True,
            )

            model = WhisperModel(**cpu_kwargs)
            self.logger.warning(
                "Successfully created model on CPU due to cuDNN issues - "
                "performance will be significantly reduced",
            )
            print(
                f"✅ Whisper model '{model_name}' download completed on CPU fallback",
                flush=True,
            )

        except Exception as cpu_fallback_error:
            self.logger.exception(
                "Even CPU fallback failed for cuDNN error: %s",
                cpu_fallback_error,
            )
            return None
        else:
            return model

    def _test_model_runtime(self, model: WhisperModel, device: str) -> None:
        """Test model with a simple operation to catch runtime errors.

        Args:
        ----
            model: WhisperModel to test
            device: Device the model is running on

        Raises:
        ------
            RuntimeError: If model runtime test fails

        """
        try:
            # Create a minimal test audio (1 second of silence at 16kHz)
            import numpy as np

            test_audio = np.zeros(16000, dtype=np.float32)

            # Try to transcribe the test audio - this will trigger cuDNN operations
            # Use minimal parameters to speed up the test
            segments, _ = model.transcribe(
                test_audio,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
                word_timestamps=False,
                vad_filter=False,
            )

            # Consume the generator to actually trigger the operations
            list(segments)

            self.logger.debug("Model runtime test passed for device: %s", device)

        except Exception as e:
            error_msg = str(e).lower()
            if "libcudnn" in error_msg or "cudnn" in error_msg:
                msg = f"cuDNN runtime error: {e}"
                raise RuntimeError(msg) from e
            if "cuda" in error_msg:
                msg = f"CUDA runtime error: {e}"
                raise RuntimeError(msg) from e
            msg = f"Model runtime test failed: {e}"
            raise RuntimeError(msg) from e


class FasterWhisperInference:
    """Wrapper for WhisperModel with enhanced error handling and graceful degradation.

    This class provides runtime error detection and automatic fallback mechanisms
    for cuDNN and CUDA errors that may occur during transcription operations.
    """

    def __init__(
        self,
        model: WhisperModel,
        config: WhisperConfig,
        logger: logging.Logger,
    ) -> None:
        """Initialize FasterWhisperInference.

        Args:
        ----
            model: WhisperModel instance
            config: Whisper configuration
            logger: Logger for logging messages

        """
        self.model = model
        self.config = config
        self.logger = logger
        self._fallback_model = None
        self._runtime_errors = 0
        self._max_runtime_errors = 3  # Max errors before forcing CPU fallback

    @retry_on_error()
    def transcribe(self, audio_path: str, **kwargs: Any) -> tuple:
        """Transcribe audio with runtime error handling.

        Args:
        ----
            audio_path: Path to audio file
            **kwargs: Additional transcription parameters

        Returns:
        -------
            Tuple of (segments, transcription_info)

        Raises:
        ------
            AudioProcessingError: If transcription fails after all fallback attempts

        """
        try:
            # Try with primary model first
            return self._transcribe_with_model(self.model, audio_path, **kwargs)

        except RuntimeError as e:
            error_msg = str(e).lower()

            # Handle cuDNN runtime errors
            if "libcudnn" in error_msg or "cudnn" in error_msg:
                self.logger.warning("cuDNN runtime error during transcription: %s", e)
                return self._handle_cudnn_runtime_error(audio_path, **kwargs)

            # Handle CUDA runtime errors
            if "cuda" in error_msg:
                self.logger.warning("CUDA runtime error during transcription: %s", e)
                return self._handle_cuda_runtime_error(audio_path, **kwargs)

            # Handle memory errors
            if "out of memory" in error_msg:
                self.logger.warning("GPU memory error during transcription: %s", e)
                return self._handle_memory_error(audio_path, **kwargs)

            # Re-raise other runtime errors
            msg = f"Transcription runtime error: {e}"
            raise AudioProcessingError(msg) from e

        except Exception as e:
            self.logger.exception("Unexpected error during transcription: %s", e)
            msg = f"Transcription failed: {e}"
            raise AudioProcessingError(msg) from e

    def _transcribe_with_model(
        self,
        model: WhisperModel,
        audio_path: str,
        **kwargs: Any,
    ) -> tuple:
        """Perform transcription with a specific model.

        Args:
        ----
            model: WhisperModel to use
            audio_path: Path to audio file
            **kwargs: Transcription parameters

        Returns:
        -------
            Tuple of (segments, transcription_info)

        """
        try:
            # Perform transcription
            segments, info = model.transcribe(audio_path, **kwargs)

            # Convert segments to list to trigger any lazy evaluation errors
            segments_list = list(segments)

            # Create transcription info
            transcription_info = {
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "all_language_probs": info.all_language_probs,
            }

        except Exception as e:
            # Log the specific error for debugging
            self.logger.debug("Transcription error with model: %s", e)
            raise
        else:
            return segments_list, transcription_info

    def _handle_cudnn_runtime_error(self, audio_path: str, **kwargs: Any) -> tuple:
        """Handle cuDNN runtime errors with GPU-first approach.

        Args:
        ----
            audio_path: Path to audio file
            **kwargs: Transcription parameters

        Returns:
        -------
            Tuple of (segments, transcription_info)

        """
        self._runtime_errors += 1

        # Try cuDNN workarounds first before falling back to CPU
        try:
            self.logger.info("Attempting cuDNN workarounds for runtime error")

            # Try with cuDNN disabled but still on GPU
            import torch

            original_cudnn_enabled = torch.backends.cudnn.enabled

            try:
                torch.backends.cudnn.enabled = False
                self.logger.info("Retrying transcription with cuDNN disabled")
                result = self._transcribe_with_model(self.model, audio_path, **kwargs)

                self.logger.warning(
                    "Successfully transcribed with cuDNN disabled - "
                    "GPU performance may be reduced but still using GPU",
                )
                return result

            finally:
                torch.backends.cudnn.enabled = original_cudnn_enabled

        except (RuntimeError, OSError, ImportError, ValueError) as gpu_retry_error:
            self.logger.warning(
                "GPU retry with cuDNN disabled failed: %s",
                gpu_retry_error,
            )

        # Only fall back to CPU after multiple failures and if not explicitly configured for CUDA
        if self._runtime_errors >= self._max_runtime_errors:
            self.logger.warning(
                "Multiple cuDNN runtime errors (%d), considering CPU fallback",
                self._runtime_errors,
            )

            # Check if user explicitly wants CUDA - if so, raise error instead of CPU fallback
            if (hasattr(self.config, "device") and self.config.device == "cuda") or (
                hasattr(self.config, "compute_type")
                and self.config.compute_type != "auto"
            ):
                self.logger.error(
                    "User explicitly configured CUDA but cuDNN errors persist. "
                    "Please fix cuDNN installation or adjust configuration.",
                )
                msg = (
                    "cuDNN runtime errors with explicit CUDA configuration - "
                    "check cuDNN installation or use CPU explicitly"
                )
                raise AudioProcessingError(
                    msg,
                )

            # Create CPU fallback model if user didn't explicitly configure CUDA
            if not self._fallback_model:
                self.logger.info("Creating CPU fallback model for cuDNN runtime errors")
                self._fallback_model = self._create_cpu_fallback_model()

            if self._fallback_model:
                self.logger.info("Using CPU fallback model due to cuDNN runtime error")
                return self._transcribe_with_model(
                    self._fallback_model,
                    audio_path,
                    **kwargs,
                )
            msg = "Failed to create CPU fallback model for cuDNN error"
            raise AudioProcessingError(
                msg,
            )
        # For early errors, just re-raise to try other GPU configurations
        msg = f"cuDNN runtime error (attempt {self._runtime_errors})"
        raise AudioProcessingError(
            msg,
        )

    def _handle_cuda_runtime_error(self, audio_path: str, **kwargs: Any) -> tuple:
        """Handle CUDA runtime errors with CPU fallback.

        Args:
        ----
            audio_path: Path to audio file
            **kwargs: Transcription parameters

        Returns:
        -------
            Tuple of (segments, transcription_info)

        """
        self._runtime_errors += 1

        # Create CPU fallback model if needed
        if not self._fallback_model:
            self.logger.info("Creating CPU fallback model for CUDA runtime errors")
            self._fallback_model = self._create_cpu_fallback_model()

        if self._fallback_model:
            self.logger.info("Using CPU fallback model due to CUDA runtime error")
            return self._transcribe_with_model(
                self._fallback_model,
                audio_path,
                **kwargs,
            )
        msg = "Failed to create CPU fallback model for CUDA error"
        raise AudioProcessingError(
            msg,
        )

    def _handle_memory_error(self, audio_path: str, **kwargs: Any) -> tuple:
        """Handle GPU memory errors with CPU fallback.

        Args:
        ----
            audio_path: Path to audio file
            **kwargs: Transcription parameters

        Returns:
        -------
            Tuple of (segments, transcription_info)

        """
        self._runtime_errors += 1

        # Try to free GPU memory
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("Cleared GPU memory cache")
        except (RuntimeError, AttributeError):  # noqa: S110
            pass

        # Create CPU fallback model if needed
        if not self._fallback_model:
            self.logger.info("Creating CPU fallback model for memory errors")
            self._fallback_model = self._create_cpu_fallback_model()

        if self._fallback_model:
            self.logger.info("Using CPU fallback model due to GPU memory error")
            return self._transcribe_with_model(
                self._fallback_model,
                audio_path,
                **kwargs,
            )
        msg = "Failed to create CPU fallback model for memory error"
        raise AudioProcessingError(
            msg,
        )

    def _create_cpu_fallback_model(self) -> WhisperModel | None:
        """Create a CPU fallback model.

        Returns
        -------
            WhisperModel instance or None if creation fails

        """
        try:
            # Use the same model size but force CPU
            model_size = graceful_degradation.get_model_fallback(self.config.model_size)

            model_kwargs = {
                "model_size_or_path": model_size,
                "device": "cpu",
                "compute_type": "int8",  # Use int8 for better CPU performance
                "cpu_threads": self.config.cpu_threads
                if self.config.cpu_threads > 0
                else os.cpu_count(),
                "num_workers": self.config.num_workers,
            }

            # Announce CPU fallback model creation
            print(
                f"🔄 Creating CPU fallback Whisper model '{model_size}'...", flush=True
            )

            fallback_model = WhisperModel(**model_kwargs)
            self.logger.info("Successfully created CPU fallback model")
            print(
                f"✅ CPU fallback Whisper model '{model_size}' created successfully",
                flush=True,
            )
        except Exception as e:
            self.logger.exception("Failed to create CPU fallback model: %s", e)
            return None
        else:
            return fallback_model

    def cleanup(self) -> None:
        """Clean up model resources."""
        try:
            if self._fallback_model:
                del self._fallback_model
                self._fallback_model = None

            if self.model:
                del self.model
                self.model = None

            # Clear GPU memory if available
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except (RuntimeError, AttributeError):  # noqa: S110
                pass

            self.logger.debug("Cleaned up WhisperModel resources")

        except (RuntimeError, AttributeError) as e:
            self.logger.warning("Error during model cleanup: %s", e)

    def detect_language(self, audio_path: str) -> dict[str, Any]:
        """Detect the language of an audio file.

        Args:
        ----
            audio_path: Path to audio file

        Returns:
        -------
            dict containing detected language and confidence

        """
        try:
            self.logger.debug("Detecting language for: %s", audio_path)

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
            with contextlib.suppress(StopIteration):
                next(segments)

            result = {
                "language": info.language,
                "language_probability": info.language_probability,
                "all_language_probs": getattr(info, "all_language_probs", {}),
            }

            self.logger.debug(
                "Language detected: %s (confidence: %.3f)",
                info.language,
                info.language_probability,
            )

        except AudioProcessingError as e:
            self.logger.exception("Language detection failed for %s", audio_path)
            msg = f"Language detection failed: {e}"
            raise AudioProcessingError(msg, audio_path) from e
        else:
            return result

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_size": self.config.model_size,
            "device": self.model.device,
            "compute_type": getattr(self.model, "compute_type", "unknown"),
            "cpu_threads": self.config.cpu_threads,
            "num_workers": self.config.num_workers,
        }
