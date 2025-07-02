"""Comprehensive error handling and retry mechanisms for the transcription application.
Provides automatic retry logic, graceful degradation, and error classification.
"""

import asyncio
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, TypeVar

T = TypeVar("T")


class ErrorCategory(Enum):
    """Classification of errors for monitoring and handling."""

    NETWORK = "network"
    FILE_SYSTEM = "file_system"
    MEMORY = "memory"
    GPU = "gpu"
    MODEL = "model"
    AUDIO = "audio"
    CONFIGURATION = "configuration"
    PROCESSING = "processing"
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorInfo:
    """Detailed error information for tracking and monitoring."""

    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    exception: Exception
    timestamp: float
    context: dict[str, Any]
    retry_count: int = 0
    recoverable: bool = True


class TranscriptionError(Exception):
    """Base exception for transcription-related errors."""

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        recoverable: bool = True,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.recoverable = recoverable


class FileSystemError(TranscriptionError):
    """File system related errors."""

    def __init__(self, message: str, path: str | None = None) -> None:
        super().__init__(message, ErrorCategory.FILE_SYSTEM, ErrorSeverity.HIGH)
        self.path = path


class ModelError(TranscriptionError):
    """Model loading or inference errors."""

    def __init__(
        self,
        message: str,
        model_name: str | None = None,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
    ) -> None:
        super().__init__(message, ErrorCategory.MODEL, severity)
        self.model_name = model_name


class MemoryError(TranscriptionError):
    """Memory-related errors."""

    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.HIGH) -> None:
        super().__init__(message, ErrorCategory.MEMORY, severity)


class AudioProcessingError(TranscriptionError):
    """Audio processing errors."""

    def __init__(self, message: str, file_path: str | None = None) -> None:
        super().__init__(message, ErrorCategory.AUDIO, ErrorSeverity.MEDIUM)
        self.file_path = file_path


class GPUError(TranscriptionError):
    """GPU-related errors."""

    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.HIGH) -> None:
        super().__init__(message, ErrorCategory.GPU, severity)


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: list[type[Exception]] | None = None,
    ) -> None:
        """Initialize retry configuration.

        Args:
            max_attempts: Maximum number of retry attempts
            base_delay: Base delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            exponential_base: Base for exponential backoff
            jitter: Whether to add random jitter to delays
            retryable_exceptions: List of exception types that should trigger retries

        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

        if retryable_exceptions is None:
            self.retryable_exceptions = [
                ConnectionError,
                TimeoutError,
                FileSystemError,
                ModelError,
                MemoryError,
                GPUError,
                OSError,
                IOError,
            ]
        else:
            self.retryable_exceptions = retryable_exceptions

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a given retry attempt."""
        delay = self.base_delay * (self.exponential_base ** (attempt - 1))
        delay = min(delay, self.max_delay)

        if self.jitter:
            # Add random jitter (±25% of delay)
            jitter_amount = delay * 0.25
            delay += random.uniform(-jitter_amount, jitter_amount)

        return max(0, delay)

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if an exception should trigger a retry."""
        if attempt >= self.max_attempts:
            return False

        # Check if exception type is retryable
        for retryable_type in self.retryable_exceptions:
            if isinstance(exception, retryable_type):
                # Additional checks for specific exceptions
                if isinstance(exception, TranscriptionError):
                    return exception.recoverable
                return True

        return False


class ErrorTracker:
    """Tracks errors for monitoring and analysis."""

    def __init__(self, max_errors: int = 1000) -> None:
        """Initialize error tracker.

        Args:
            max_errors: Maximum number of errors to keep in memory

        """
        self.max_errors = max_errors
        self.errors: list[ErrorInfo] = []
        self.error_counts: dict[ErrorCategory, int] = dict.fromkeys(ErrorCategory, 0)
        self.logger = logging.getLogger(__name__)

    def record_error(self, error_info: ErrorInfo) -> None:
        """Record an error for tracking."""
        self.errors.append(error_info)
        self.error_counts[error_info.category] += 1

        # Maintain maximum size
        if len(self.errors) > self.max_errors:
            old_error = self.errors.pop(0)
            self.error_counts[old_error.category] -= 1

        # Log the error
        self.logger.error(
            f"Error recorded: {error_info.category.value} - {error_info.message}",
            extra={
                "error_category": error_info.category.value,
                "error_severity": error_info.severity.value,
                "retry_count": error_info.retry_count,
                "context": error_info.context,
            },
        )

    def get_error_stats(self) -> dict[str, Any]:
        """Get error statistics."""
        total_errors = len(self.errors)
        recent_errors = [
            e for e in self.errors if time.time() - e.timestamp < 3600
        ]  # Last hour

        return {
            "total_errors": total_errors,
            "recent_errors": len(recent_errors),
            "error_counts": dict(self.error_counts),
            "error_rate": len(recent_errors) / 60
            if recent_errors
            else 0,  # errors per minute
        }

    def get_recent_errors(self, limit: int = 10) -> list[ErrorInfo]:
        """Get most recent errors."""
        return sorted(self.errors, key=lambda e: e.timestamp, reverse=True)[:limit]


# Global error tracker instance
error_tracker = ErrorTracker()


def classify_error(exception: Exception) -> ErrorInfo:
    """Classify an exception into error categories and create ErrorInfo.

    Args:
        exception: Exception to classify

    Returns:
        ErrorInfo: Classified error information

    """
    category = ErrorCategory.UNKNOWN
    severity = ErrorSeverity.MEDIUM
    recoverable = True

    # Classify based on exception type and message
    if isinstance(exception, FileNotFoundError | PermissionError | OSError | IOError):
        category = ErrorCategory.FILE_SYSTEM
        severity = ErrorSeverity.HIGH
    elif isinstance(exception, ConnectionError | TimeoutError):
        category = ErrorCategory.NETWORK
        severity = ErrorSeverity.MEDIUM
    elif isinstance(exception, MemoryError) or "memory" in str(exception).lower():
        category = ErrorCategory.MEMORY
        severity = ErrorSeverity.HIGH
    elif "cuda" in str(exception).lower() or "gpu" in str(exception).lower():
        category = ErrorCategory.GPU
        severity = ErrorSeverity.HIGH
    elif isinstance(exception, TranscriptionError):
        category = exception.category
        severity = exception.severity
        recoverable = exception.recoverable

    # Special handling for specific error messages
    error_msg = str(exception).lower()
    if "out of memory" in error_msg or "oom" in error_msg:
        category = ErrorCategory.MEMORY
        severity = ErrorSeverity.HIGH
    elif "model" in error_msg and ("load" in error_msg or "download" in error_msg):
        category = ErrorCategory.MODEL
        severity = ErrorSeverity.HIGH
    elif "audio" in error_msg or "codec" in error_msg:
        category = ErrorCategory.AUDIO
        severity = ErrorSeverity.MEDIUM

    return ErrorInfo(
        category=category,
        severity=severity,
        message=str(exception),
        exception=exception,
        timestamp=time.time(),
        context={},
        recoverable=recoverable,
    )


def retry_on_error(config: RetryConfig | None = None):
    """Decorator for automatic retry on errors.

    Args:
        config: Retry configuration. If None, uses default configuration.

    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_info = classify_error(e)
                    error_info.retry_count = attempt - 1

                    if not config.should_retry(e, attempt):
                        error_tracker.record_error(error_info)
                        raise

                    if attempt < config.max_attempts:
                        delay = config.calculate_delay(attempt)
                        logging.getLogger(__name__).warning(
                            f"Attempt {attempt} failed, retrying in {delay:.2f}s: {e}",
                        )
                        time.sleep(delay)
                    else:
                        error_tracker.record_error(error_info)

            # If we get here, all attempts failed
            raise last_exception

        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(1, config.max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_info = classify_error(e)
                    error_info.retry_count = attempt - 1

                    if not config.should_retry(e, attempt):
                        error_tracker.record_error(error_info)
                        raise

                    if attempt < config.max_attempts:
                        delay = config.calculate_delay(attempt)
                        logging.getLogger(__name__).warning(
                            f"Attempt {attempt} failed, retrying in {delay:.2f}s: {e}",
                        )
                        await asyncio.sleep(delay)
                    else:
                        error_tracker.record_error(error_info)

            # If we get here, all attempts failed
            raise last_exception

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


class CircuitBreaker:
    """Circuit breaker pattern implementation for handling repeated failures.
    Prevents cascading failures by temporarily disabling failing operations.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exception: type[Exception] = Exception,
    ) -> None:
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Time to wait before attempting to close circuit
            expected_exception: Exception type that triggers the circuit breaker

        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open
        self.logger = logging.getLogger(__name__)

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Call a function through the circuit breaker.

        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If circuit is open or function fails

        """
        if self.state == "open":
            if time.time() - self.last_failure_time >= self.timeout:
                self.state = "half-open"
                self.logger.info("Circuit breaker moving to half-open state")
            else:
                msg = "Circuit breaker is open"
                raise Exception(msg)

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        """Handle successful operation."""
        self.failure_count = 0
        if self.state == "half-open":
            self.state = "closed"
            self.logger.info("Circuit breaker closed")

    def _on_failure(self) -> None:
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            self.logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures",
            )


class GracefulDegradation:
    """Handles graceful degradation of functionality when resources are constrained.
    Provides fallback mechanisms for model loading and processing.
    """

    def __init__(self) -> None:
        self.degradation_level = 0  # 0 = full functionality, higher = more degraded
        self.logger = logging.getLogger(__name__)

    def get_model_fallback(self, requested_model: str) -> str:
        """Get a fallback model based on current degradation level.

        Args:
            requested_model: Originally requested model

        Returns:
            str: Fallback model to use

        """
        model_hierarchy = {
            "large-v3": ["large-v2", "large-v1", "medium", "small", "base", "tiny"],
            "large-v2": ["large-v1", "medium", "small", "base", "tiny"],
            "large-v1": ["medium", "small", "base", "tiny"],
            "medium": ["small", "base", "tiny"],
            "small": ["base", "tiny"],
            "base": ["tiny"],
            "tiny": ["tiny"],
        }

        fallbacks = model_hierarchy.get(requested_model, [requested_model])
        fallback_index = min(self.degradation_level, len(fallbacks) - 1)

        fallback_model = fallbacks[fallback_index]
        if fallback_model != requested_model:
            self.logger.warning(
                f"Using fallback model {fallback_model} instead of {requested_model}",
            )

        return fallback_model

    def get_compute_type_fallback(self, requested_type: str) -> str:
        """Get a fallback compute type based on current degradation level.

        Args:
            requested_type: Originally requested compute type

        Returns:
            str: Fallback compute type to use

        """
        type_hierarchy = {
            "float16": ["float32", "int8"],
            "float32": ["int8"],
            "int8": ["int8"],
        }

        fallbacks = type_hierarchy.get(requested_type, [requested_type])
        fallback_index = min(self.degradation_level, len(fallbacks) - 1)

        fallback_type = fallbacks[fallback_index]
        if fallback_type != requested_type:
            self.logger.warning(
                f"Using fallback compute type {fallback_type} instead of {requested_type}",
            )

        return fallback_type

    def should_disable_feature(self, feature: str) -> bool:
        """Determine if a feature should be disabled based on degradation level.

        Args:
            feature: Feature name to check

        Returns:
            bool: True if feature should be disabled

        """
        feature_thresholds = {
            "bgm_separation": 1,
            "diarization": 2,
            "vad": 3,
        }

        threshold = feature_thresholds.get(feature, 999)
        should_disable = self.degradation_level >= threshold

        if should_disable:
            self.logger.warning(
                f"Disabling feature '{feature}' due to resource constraints",
            )

        return should_disable

    def increase_degradation(self) -> None:
        """Increase degradation level due to resource constraints."""
        self.degradation_level += 1
        self.logger.warning("Degradation level increased to %s", self.degradation_level)

    def reset_degradation(self) -> None:
        """Reset degradation level when resources are available."""
        if self.degradation_level > 0:
            self.logger.info("Resetting degradation level to 0")
            self.degradation_level = 0


# Global graceful degradation instance
graceful_degradation = GracefulDegradation()
