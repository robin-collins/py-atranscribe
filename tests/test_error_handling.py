"""
Unit tests for error handling and retry mechanisms.
"""

import asyncio
import time
from unittest.mock import Mock, patch

import pytest

from src.utils.error_handling import (
    ErrorCategory, ErrorSeverity, ErrorInfo, TranscriptionError,
    FileSystemError, ModelError, AudioProcessingError, GPUError,
    RetryConfig, ErrorTracker, classify_error, retry_on_error,
    CircuitBreaker, GracefulDegradation, graceful_degradation
)


class TestErrorClasses:
    """Test custom error classes."""

    def test_transcription_error(self):
        """Test base TranscriptionError."""
        error = TranscriptionError("Test message")
        assert str(error) == "Test message"
        assert error.category == ErrorCategory.UNKNOWN
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.recoverable is True

    def test_file_system_error(self):
        """Test FileSystemError."""
        error = FileSystemError("File not found", "/path/to/file")
        assert error.category == ErrorCategory.FILE_SYSTEM
        assert error.severity == ErrorSeverity.HIGH
        assert error.path == "/path/to/file"

    def test_model_error(self):
        """Test ModelError."""
        error = ModelError("Model loading failed", "whisper-large")
        assert error.category == ErrorCategory.MODEL
        assert error.severity == ErrorSeverity.HIGH
        assert error.model_name == "whisper-large"

    def test_audio_processing_error(self):
        """Test AudioProcessingError."""
        error = AudioProcessingError("Audio processing failed", "/path/to/audio.wav")
        assert error.category == ErrorCategory.AUDIO
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.file_path == "/path/to/audio.wav"


class TestErrorClassification:
    """Test error classification functionality."""

    def test_classify_file_not_found(self):
        """Test classification of FileNotFoundError."""
        error = FileNotFoundError("File not found")
        info = classify_error(error)
        assert info.category == ErrorCategory.FILE_SYSTEM
        assert info.severity == ErrorSeverity.HIGH

    def test_classify_connection_error(self):
        """Test classification of ConnectionError."""
        error = ConnectionError("Connection failed")
        info = classify_error(error)
        assert info.category == ErrorCategory.NETWORK
        assert info.severity == ErrorSeverity.MEDIUM

    def test_classify_memory_error(self):
        """Test classification of MemoryError."""
        error = MemoryError("Out of memory")
        info = classify_error(error)
        assert info.category == ErrorCategory.MEMORY
        assert info.severity == ErrorSeverity.HIGH

    def test_classify_cuda_error(self):
        """Test classification of CUDA errors."""
        error = RuntimeError("CUDA out of memory")
        info = classify_error(error)
        assert info.category == ErrorCategory.GPU
        assert info.severity == ErrorSeverity.HIGH

    def test_classify_transcription_error(self):
        """Test classification of custom TranscriptionError."""
        error = AudioProcessingError("Processing failed")
        info = classify_error(error)
        assert info.category == ErrorCategory.AUDIO
        assert info.severity == ErrorSeverity.MEDIUM


class TestRetryConfig:
    """Test retry configuration functionality."""

    def test_default_config(self):
        """Test default retry configuration."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.exponential_base == 2.0
        assert config.jitter is True

    def test_calculate_delay(self):
        """Test delay calculation."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)
        assert config.calculate_delay(1) == 1.0
        assert config.calculate_delay(2) == 2.0
        assert config.calculate_delay(3) == 4.0

    def test_calculate_delay_with_max(self):
        """Test delay calculation with maximum."""
        config = RetryConfig(base_delay=1.0, max_delay=3.0, jitter=False)
        assert config.calculate_delay(1) == 1.0
        assert config.calculate_delay(2) == 2.0
        assert config.calculate_delay(3) == 3.0  # Capped at max_delay
        assert config.calculate_delay(4) == 3.0  # Still capped

    def test_should_retry(self):
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

    def test_record_error(self):
        """Test error recording."""
        tracker = ErrorTracker(max_errors=10)
        error_info = ErrorInfo(
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            message="Test error",
            exception=ConnectionError("Test"),
            timestamp=time.time(),
            context={}
        )

        tracker.record_error(error_info)
        assert len(tracker.errors) == 1
        assert tracker.error_counts[ErrorCategory.NETWORK] == 1

    def test_max_errors_limit(self):
        """Test maximum errors limit."""
        tracker = ErrorTracker(max_errors=2)

        for i in range(3):
            error_info = ErrorInfo(
                category=ErrorCategory.NETWORK,
                severity=ErrorSeverity.MEDIUM,
                message=f"Error {i}",
                exception=ConnectionError(f"Error {i}"),
                timestamp=time.time(),
                context={}
            )
            tracker.record_error(error_info)

        assert len(tracker.errors) == 2  # Limited to max_errors

    def test_get_error_stats(self):
        """Test error statistics."""
        tracker = ErrorTracker()

        # Add some errors
        for category in [ErrorCategory.NETWORK, ErrorCategory.NETWORK, ErrorCategory.MODEL]:
            error_info = ErrorInfo(
                category=category,
                severity=ErrorSeverity.MEDIUM,
                message="Test",
                exception=Exception("Test"),
                timestamp=time.time(),
                context={}
            )
            tracker.record_error(error_info)

        stats = tracker.get_error_stats()
        assert stats["total_errors"] == 3
        assert stats["error_counts"][ErrorCategory.NETWORK.value] == 2
        assert stats["error_counts"][ErrorCategory.MODEL.value] == 1


class TestRetryDecorator:
    """Test retry decorator functionality."""

    def test_successful_function(self):
        """Test retry decorator with successful function."""
        call_count = 0

        @retry_on_error(RetryConfig(max_attempts=3))
        def successful_function():
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_function()
        assert result == "success"
        assert call_count == 1

    def test_function_with_retries(self):
        """Test retry decorator with failing then successful function."""
        call_count = 0

        @retry_on_error(RetryConfig(max_attempts=3, base_delay=0.1))
        def sometimes_failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"

        result = sometimes_failing_function()
        assert result == "success"
        assert call_count == 3

    def test_function_exceeds_max_attempts(self):
        """Test retry decorator exceeding maximum attempts."""
        call_count = 0

        @retry_on_error(RetryConfig(max_attempts=2, base_delay=0.1))
        def always_failing_function():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Always fails")

        with pytest.raises(ConnectionError):
            always_failing_function()
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_function_with_retries(self):
        """Test retry decorator with async function."""
        call_count = 0

        @retry_on_error(RetryConfig(max_attempts=3, base_delay=0.1))
        async def async_sometimes_failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Temporary failure")
            return "success"

        result = await async_sometimes_failing_function()
        assert result == "success"
        assert call_count == 2


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_circuit_breaker_normal_operation(self):
        """Test circuit breaker in normal operation."""
        breaker = CircuitBreaker(failure_threshold=3, timeout=1.0)

        def successful_function():
            return "success"

        # Should work normally
        result = breaker.call(successful_function)
        assert result == "success"
        assert breaker.state == "closed"

    def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opening after failures."""
        breaker = CircuitBreaker(failure_threshold=2, timeout=1.0)

        def failing_function():
            raise Exception("Always fails")

        # First failure
        with pytest.raises(Exception):
            breaker.call(failing_function)
        assert breaker.state == "closed"

        # Second failure - should open circuit
        with pytest.raises(Exception):
            breaker.call(failing_function)
        assert breaker.state == "open"

        # Third call should fail due to open circuit
        with pytest.raises(Exception, match="Circuit breaker is open"):
            breaker.call(failing_function)

    def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker half-open and recovery."""
        breaker = CircuitBreaker(failure_threshold=1, timeout=0.1)

        def failing_function():
            raise Exception("Fails")

        def successful_function():
            return "success"

        # Trigger failure to open circuit
        with pytest.raises(Exception):
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

    def test_model_fallback(self):
        """Test model fallback functionality."""
        degradation = GracefulDegradation()

        # No degradation
        assert degradation.get_model_fallback("large-v3") == "large-v3"

        # With degradation
        degradation.increase_degradation()
        assert degradation.get_model_fallback("large-v3") == "large-v2"

        degradation.increase_degradation()
        assert degradation.get_model_fallback("large-v3") == "large-v1"

    def test_compute_type_fallback(self):
        """Test compute type fallback functionality."""
        degradation = GracefulDegradation()

        # No degradation
        assert degradation.get_compute_type_fallback("float16") == "float16"

        # With degradation
        degradation.increase_degradation()
        assert degradation.get_compute_type_fallback("float16") == "float32"

        degradation.increase_degradation()
        assert degradation.get_compute_type_fallback("float16") == "int8"

    def test_feature_disabling(self):
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

    def test_reset_degradation(self):
        """Test resetting degradation level."""
        degradation = GracefulDegradation()

        # Increase degradation
        degradation.increase_degradation()
        degradation.increase_degradation()
        assert degradation.degradation_level == 2

        # Reset
        degradation.reset_degradation()
        assert degradation.degradation_level == 0