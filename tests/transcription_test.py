"""Unit tests for Whisper transcription factory and inference wrapper."""

import logging
from unittest.mock import MagicMock, Mock, patch
from typing import Any

import pytest

from src.config import WhisperConfig
from src.transcription.whisper_factory import FasterWhisperInference, WhisperFactory
from src.utils.error_handling import (
    AudioProcessingError,
    ModelError,
    graceful_degradation,
)


# Mock the faster_whisper.WhisperModel class
class MockWhisperModel:
    """Mock implementation of faster_whisper.WhisperModel for testing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize mock Whisper model."""
        self.device = kwargs.get("device", "cpu")
        self.compute_type = kwargs.get("compute_type", "int8")

    def transcribe(self, audio: Any, **kwargs: Any) -> tuple[list[Any], Any]:
        """Mock transcription processing."""
        # Simulate a successful transcription
        mock_segment = MagicMock()
        mock_segment.start = 0
        mock_segment.end = 1
        mock_segment.text = "test"
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.99
        mock_info.duration = 1.0
        return [mock_segment], mock_info


@patch("faster_whisper.WhisperModel", MockWhisperModel)
class TestWhisperFactory:
    """Test the WhisperFactory class."""

    def setup_method(self) -> None:
        """Set up the factory and clear caches."""
        WhisperFactory.clear_cache()
        graceful_degradation.reset_degradation()
        self.factory = WhisperFactory()

    def test_create_whisper_inference(self) -> None:
        """Test creating a standard WhisperInference instance."""
        config = WhisperConfig(model_size="tiny", device="cpu", compute_type="int8")
        inference = self.factory.create_whisper_inference(config)
        assert isinstance(inference, FasterWhisperInference)
        assert inference.model.device == "cpu"
        assert inference.model.compute_type == "int8"

    def test_model_caching(self) -> None:
        """Test that models are cached to avoid reloading."""
        config = WhisperConfig(model_size="tiny", device="cpu", compute_type="int8")

        inference1 = self.factory.create_whisper_inference(config)
        inference2 = self.factory.create_whisper_inference(config)

        assert inference1.model is inference2.model

    @patch(
        "src.transcription.whisper_factory.WhisperFactory.get_optimal_device",
        return_value="cuda",
    )
    @patch(
        "src.transcription.whisper_factory.WhisperFactory.get_optimal_compute_type",
        return_value="float16",
    )
    def test_auto_device_and_compute_type(
        self, mock_get_compute: MagicMock, mock_get_device: MagicMock
    ) -> None:
        """Test automatic detection of device and compute type."""
        config = WhisperConfig(model_size="medium", device="auto", compute_type="auto")
        inference = self.factory.create_whisper_inference(config)

        mock_get_device.assert_called_once()
        mock_get_compute.assert_called_once_with("cuda", "auto")
        assert inference.model.device == "cuda"
        assert inference.model.compute_type == "float16"

    @patch(
        "faster_whisper.WhisperModel",
        side_effect=[
            ModelError("Load failed"),
            MockWhisperModel(device="cpu", compute_type="int8"),
        ],
    )
    def test_creation_fallback(self, mock_create_model: MagicMock) -> None:
        """Test that the factory falls back to a safer configuration on error."""
        # This test is conceptual as the mock setup is complex.
        # The goal is to verify the fallback logic is invoked.
        config = WhisperConfig(
            model_size="large-v3", device="cuda", compute_type="float16"
        )

        # We expect a warning about the fallback
        with self.assertLogs(
            "src.transcription.whisper_factory", level="WARNING"
        ) as cm:
            inference = self.factory.create_whisper_inference(config)
            assert "Model creation failed" in cm.output[0]
            assert "Successfully created model with device=cpu" in cm.output[1]

        assert inference.model.device == "cpu"
        assert inference.model.compute_type == "int8"
        assert mock_create_model.call_count > 1


@patch("faster_whisper.WhisperModel", MockWhisperModel)
class TestFasterWhisperInference:
    """Test the FasterWhisperInference wrapper."""

    def setup_method(self) -> None:
        """Set up the inference wrapper."""
        self.config = WhisperConfig(model_size="tiny")
        self.model = MockWhisperModel()
        self.logger = logging.getLogger(__name__)
        self.inference = FasterWhisperInference(self.model, self.config, self.logger)

    def test_successful_transcription(self) -> None:
        """Test a successful transcription call."""
        segments, info = self.inference.transcribe("dummy_path.wav")
        assert len(segments) == 1
        assert segments[0].text == "test"
        assert info["language"] == "en"

    @patch.object(
        MockWhisperModel, "transcribe", side_effect=RuntimeError("CUDA out of memory")
    )
    def test_runtime_error_fallback(self, mock_transcribe: MagicMock) -> None:
        """Test that a runtime error triggers a fallback to a CPU model."""
        # Mock the creation of the fallback model
        with patch.object(
            self.inference,
            "_create_cpu_fallback_model",
            return_value=MockWhisperModel(device="cpu"),
        ) as mock_create_fallback:
            segments, info = self.inference.transcribe("dummy_path.wav")

            mock_transcribe.assert_called_once()
            mock_create_fallback.assert_called_once()

            # Check that the transcription was successful with the fallback model
            assert len(segments) == 1
            assert info["language"] == "en"

    @patch.object(
        MockWhisperModel, "transcribe", side_effect=AudioProcessingError("Failed")
    )
    def test_non_runtime_error_raises(self, mock_transcribe: MagicMock) -> None:
        """Test that non-runtime errors are raised without fallback."""
        with pytest.raises(AudioProcessingError):
            self.inference.transcribe("dummy_path.wav")
