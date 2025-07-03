"""Unit tests for error handling and retry mechanisms."""

import time
from typing import Never

import pytest

from src.utils.error_handling import (
    AudioProcessingError,
    CircuitBreaker,
    ErrorCategory,
    ErrorInfo,
    ErrorSeverity,
    ErrorTracker,
    FileSystemError,
    GracefulDegradation,
    ModelError,
    RetryConfig,
    TranscriptionError,
    classify_error,
    retry_on_error,
)

# Expected test values for retry configuration
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BASE_DELAY = 1.0
DEFAULT_EXPONENTIAL_BASE = 2.0
DELAY_ATTEMPT_1 = 1.0
DELAY_ATTEMPT_2 = 2.0
DELAY_ATTEMPT_3 = 4.0
MAX_DELAY_CAPPED = 3.0
MAX_ERRORS_LIMIT = 2
TOTAL_ERRORS_COUNT = 3
NETWORK_ERRORS_COUNT = 2
MODEL_ERRORS_COUNT = 1
RETRY_SUCCESS_ATTEMPTS = 3
RETRY_FAILURE_ATTEMPTS = 2
ASYNC_SUCCESS_ATTEMPTS = 2
DEGRADATION_LEVEL = 2


class TestErrorClasses:
    """Test custom error classes."""

    def test_transcription_error(self) -> None:
        """Test base TranscriptionError."""
        error = TranscriptionError("Test message")
        assert str(error) == "Test message"
        assert error.category == ErrorCategory.UNKNOWN
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.recoverable is True

    def test_file_system_error(self) -> None:
        """Test FileSystemError."""
        error = FileSystemError("File not found", "/path/to/file")
        assert error.category == ErrorCategory.FILE_SYSTEM
        assert error.severity == ErrorSeverity.HIGH
        assert error.path == "/path/to/file"

    def test_model_error(self) -> None:
        """Test ModelError."""
        error = ModelError("Model loading failed", "whisper-large")
        assert error.category == ErrorCategory.MODEL
        assert error.severity == ErrorSeverity.HIGH
        assert error.model_name == "whisper-large"

    def test_audio_processing_error(self) -> None:
        """Test AudioProcessingError."""
        error = AudioProcessingError("Audio processing failed", "/path/to/audio.wav")
        assert error.category == ErrorCategory.AUDIO
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.file_path == "/path/to/audio.wav"


class TestErrorClassification:
    """Test error classification functionality."""

    def test_classify_file_not_found(self) -> None:
        """Test classification of FileNotFoundError."""
        error = FileNotFoundError("File not found")
        info = classify_error(error)
        assert info.category == ErrorCategory.FILE_SYSTEM
        assert info.severity == ErrorSeverity.HIGH

    def test_classify_connection_error(self) -> None:
        """Test classification of ConnectionError."""
        error = ConnectionError("Connection failed")
        info = classify_error(error)
        assert info.category == ErrorCategory.NETWORK
        assert info.severity == ErrorSeverity.MEDIUM

    def test_classify_memory_error(self) -> None:
        """Test classification of MemoryError."""
        error = MemoryError("Out of memory")
        info = classify_error(error)
        assert info.category == ErrorCategory.MEMORY
        assert info.severity == ErrorSeverity.HIGH

    def test_classify_cuda_error(self) -> None:
        """Test classification of CUDA errors."""
        error = RuntimeError("CUDA out of memory")
        info = classify_error(error)
        assert info.category == ErrorCategory.GPU
        assert info.severity == ErrorSeverity.HIGH

    def test_classify_transcription_error(self) -> None:
        """Test classification of custom TranscriptionError."""
        error = AudioProcessingError("Processing failed")
        info = classify_error(error)
        assert info.category == ErrorCategory.AUDIO
        assert info.severity == ErrorSeverity.MEDIUM


class TestRetryConfig:
    """Test retry configuration functionality."""

    def test_default_config(self) -> None:
        """Test default retry configuration."""
        config = RetryConfig()
        assert config.max_attempts == DEFAULT_MAX_ATTEMPTS
        assert config.base_delay == DEFAULT_BASE_DELAY
        assert config.exponential_base == DEFAULT_EXPONENTIAL_BASE
        assert config.jitter is True

    def test_calculate_delay(self) -> None:
        """Test delay calculation."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)
        assert config.calculate_delay(1) == DELAY_ATTEMPT_1
        assert config.calculate_delay(2) == DELAY_ATTEMPT_2
        assert config.calculate_delay(3) == DELAY_ATTEMPT_3

    def test_calculate_delay_with_max(self) -> None:
        """Test delay calculation with maximum."""
        config = RetryConfig(base_delay=1.0, max_delay=3.0, jitter=False)
        assert config.calculate_delay(1) == DELAY_ATTEMPT_1
        assert config.calculate_delay(2) == DELAY_ATTEMPT_2
        assert config.calculate_delay(3) == MAX_DELAY_CAPPED  # Capped at max_delay
        assert config.calculate_delay(4) == MAX_DELAY_CAPPED  # Still capped

    def test_should_retry(self) -> None:
        """Test retry decision logic."""
        config = RetryConfig(max_attempts=3)

        # Should retry retryable exceptions
        assert config.should_retry(ConnectionError(), 1) is True
        assert config.should_retry(FileSystemError("test"), 2) is True

        # Should not retry after max attempts
        assert config.should_retry(ConnectionError(), 3) is False

        # Should not retry non-retryable exceptions
        assert config.should_retry(ValueError(), 1) is False


class TestErrorTracker:
    """Test error tracking functionality."""

    def test_record_error(self) -> None:
        """Test error recording."""
        tracker = ErrorTracker(max_errors=10)
        error_info = ErrorInfo(
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            message="Test error",
            exception=ConnectionError("Test"),
            timestamp=time.time(),
            context={},
        )

        tracker.record_error(error_info)
        assert len(tracker.errors) == 1
        assert tracker.error_counts[ErrorCategory.NETWORK] == 1

    def test_max_errors_limit(self) -> None:
        """Test maximum errors limit."""
        tracker = ErrorTracker(max_errors=2)

        for i in range(3):
            error_info = ErrorInfo(
                category=ErrorCategory.NETWORK,
                severity=ErrorSeverity.MEDIUM,
                message=f"Error {i}",
                exception=ConnectionError(f"Error {i}"),
                timestamp=time.time(),
                context={},
            )
            tracker.record_error(error_info)

        assert len(tracker.errors) == MAX_ERRORS_LIMIT  # Limited to max_errors

    def test_get_error_stats(self) -> None:
        """Test error statistics."""
        tracker = ErrorTracker()

        # Add some errors
        for category in [
            ErrorCategory.NETWORK,
            ErrorCategory.NETWORK,
            ErrorCategory.MODEL,
        ]:
            error_info = ErrorInfo(
                category=category,
                severity=ErrorSeverity.MEDIUM,
                message="Test",
                exception=Exception("Test"),
                timestamp=time.time(),
                context={},
            )
            tracker.record_error(error_info)

        stats = tracker.get_error_stats()
        assert stats["total_errors"] == TOTAL_ERRORS_COUNT
        assert (
            stats["error_counts"][ErrorCategory.NETWORK.value] == NETWORK_ERRORS_COUNT
        )
        assert stats["error_counts"][ErrorCategory.MODEL.value] == MODEL_ERRORS_COUNT


class TestRetryDecorator:
    """Test retry decorator functionality."""

    def test_successful_function(self) -> None:
        """Test retry decorator with successful function."""
        call_count = 0

        @retry_on_error(RetryConfig(max_attempts=3))
        def successful_function() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_function()
        assert result == "success"
        assert call_count == 1

    def test_function_with_retries(self) -> None:
        """Test retry decorator with failing then successful function."""
        call_count = 0

        @retry_on_error(RetryConfig(max_attempts=3, base_delay=0.1))
        def sometimes_failing_function() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < RETRY_SUCCESS_ATTEMPTS:
                msg = "Temporary failure"
                raise ConnectionError(msg)
            return "success"

        result = sometimes_failing_function()
        assert result == "success"
        assert call_count == RETRY_SUCCESS_ATTEMPTS

    def test_function_exceeds_max_attempts(self) -> None:
        """Test retry decorator exceeding maximum attempts."""
        call_count = 0

        @retry_on_error(RetryConfig(max_attempts=2, base_delay=0.1))
        def always_failing_function() -> Never:
            nonlocal call_count
            call_count += 1
            msg = "Always fails"
            raise ConnectionError(msg)

        with pytest.raises(ConnectionError):
            always_failing_function()
        assert call_count == RETRY_FAILURE_ATTEMPTS

    @pytest.mark.asyncio
    async def test_async_function_with_retries(self) -> None:
        """Test retry decorator with async function."""
        call_count = 0

        @retry_on_error(RetryConfig(max_attempts=3, base_delay=0.1))
        async def async_sometimes_failing_function() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < ASYNC_SUCCESS_ATTEMPTS:
                msg = "Temporary failure"
                raise ConnectionError(msg)
            return "success"

        result = await async_sometimes_failing_function()
        assert result == "success"
        assert call_count == ASYNC_SUCCESS_ATTEMPTS


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_circuit_breaker_normal_operation(self) -> None:
        """Test circuit breaker in normal operation."""
        breaker = CircuitBreaker(failure_threshold=3, timeout=1.0)

        def successful_function() -> str:
            return "success"

        # Should work normally
        result = breaker.call(successful_function)
        assert result == "success"
        assert breaker.state == "closed"

    def test_circuit_breaker_opens_on_failures(self) -> None:
        """Test circuit breaker opening after failures."""
        breaker = CircuitBreaker(failure_threshold=2, timeout=1.0)

        class TestException(Exception):
            """Custom exception for testing."""

            pass

        def failing_function() -> Never:
            msg = "Always fails"
            raise TestException(msg)

        # First failure
        with pytest.raises(TestException, match="Always fails"):
            breaker.call(failing_function)
        assert breaker.state == "closed"

        # Second failure - should open circuit
        with pytest.raises(TestException, match="Always fails"):
            breaker.call(failing_function)
        assert breaker.state == "open"

        # Third call should fail due to open circuit
        with pytest.raises(TestException, match="Circuit breaker is open"):
            breaker.call(failing_function)

    def test_circuit_breaker_half_open_recovery(self) -> None:
        """Test circuit breaker half-open and recovery."""
        breaker = CircuitBreaker(failure_threshold=1, timeout=0.1)

        class TestException(Exception):
            """Custom exception for testing."""

            pass

        def failing_function() -> Never:
            msg = "Fails"
            raise TestException(msg)

        def successful_function() -> str:
            return "success"

        # Trigger failure to open circuit
        with pytest.raises(TestException, match="Fails"):
            breaker.call(failing_function)
        assert breaker.state == "open"

        # Wait for timeout
        time.sleep(0.2)

        # Should move to half-open and succeed
        result = breaker.call(successful_function)
        assert result == "success"
        assert breaker.state == "closed"


class TestGracefulDegradation:
    """Test graceful degradation functionality."""

    def test_model_fallback(self) -> None:
        """Test model fallback functionality."""
        degradation = GracefulDegradation()

        # No degradation
        assert degradation.get_model_fallback("large-v3") == "large-v3"

        # With degradation
        degradation.increase_degradation()
        assert degradation.get_model_fallback("large-v3") == "large-v2"

        degradation.increase_degradation()
        assert degradation.get_model_fallback("large-v3") == "large-v1"

    def test_compute_type_fallback(self) -> None:
        """Test compute type fallback functionality."""
        degradation = GracefulDegradation()

        # No degradation
        assert degradation.get_compute_type_fallback("float16") == "float16"

        # With degradation
        degradation.increase_degradation()
        assert degradation.get_compute_type_fallback("float16") == "float32"

        degradation.increase_degradation()
        assert degradation.get_compute_type_fallback("float16") == "int8"

    def test_feature_disabling(self) -> None:
        """Test feature disabling based on degradation level."""
        degradation = GracefulDegradation()

        # No degradation - all features enabled
        assert degradation.should_disable_feature("bgm_separation") is False
        assert degradation.should_disable_feature("diarization") is False

        # Degradation level 1 - disable BGM separation
        degradation.increase_degradation()
        assert degradation.should_disable_feature("bgm_separation") is True
        assert degradation.should_disable_feature("diarization") is False

        # Degradation level 2 - disable diarization too
        degradation.increase_degradation()
        assert degradation.should_disable_feature("diarization") is True

    def test_reset_degradation(self) -> None:
        """Test resetting degradation level."""
        degradation = GracefulDegradation()

        # Increase degradation
        degradation.increase_degradation()
        degradation.increase_degradation()
        assert degradation.degradation_level == DEGRADATION_LEVEL

        # Reset
        degradation.reset_degradation()
        assert degradation.degradation_level == 0
