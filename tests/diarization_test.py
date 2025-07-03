"""Unit tests for the Diarizer class."""

from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.config import DiarizationConfig
from src.diarization.diarizer import DiarizationResult, Diarizer
from src.utils.error_handling import ModelError

# Expected test values
EXPECTED_NUM_SPEAKERS = 2
EXPECTED_NUM_SEGMENTS = 2
EXPECTED_NUM_LABELED_SEGMENTS = 3


# Mock the pyannote.audio classes
class MockPyannotePipeline:
    """Mock implementation of pyannote.audio Pipeline for testing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize mock pipeline."""
        pass

    def __call__(self, audio_path: str, **kwargs: Any) -> MagicMock:
        """Mock diarization processing."""
        # Simulate a successful diarization
        annotation = MagicMock()
        track1 = MagicMock()
        track1.start = 0.0
        track1.end = 2.0
        track2 = MagicMock()
        track2.start = 2.5
        track2.end = 4.0
        annotation.itertracks.return_value = [
            (track1, None, "SPEAKER_00"),
            (track2, None, "SPEAKER_01"),
        ]
        annotation.get_timeline.return_value.duration.return_value = 4.0
        return annotation


@patch("pyannote.audio.Pipeline.from_pretrained", return_value=MockPyannotePipeline())
class TestDiarizer:
    """Test the Diarizer class."""

    def setup_method(self) -> None:
        """Set up the diarizer."""
        self.config = DiarizationConfig(enabled=True, hf_token="fake_token")  # noqa: S106
        # We patch torch since it's a heavy dependency
        with patch("torch.cuda.is_available", return_value=False):
            self.diarizer = Diarizer(self.config)

    @patch(
        "torchaudio.info",
        return_value=MagicMock(num_frames=16000 * 5, sample_rate=16000),
    )
    def test_successful_diarization(
        self, mock_torchaudio_info: MagicMock, mock_pipeline_from_pretrained: MagicMock
    ) -> None:
        """Test a successful diarization call."""
        result = self.diarizer.diarize("dummy_audio.wav")
        assert isinstance(result, DiarizationResult)
        assert result.num_speakers == EXPECTED_NUM_SPEAKERS
        assert len(result.segments) == EXPECTED_NUM_SEGMENTS
        assert result.speakers[0].label == "SPEAKER_00"

    def test_diarization_disabled(
        self, mock_pipeline_from_pretrained: MagicMock
    ) -> None:
        """Test that diarization returns an empty result when disabled."""
        config = DiarizationConfig(enabled=False)
        diarizer = Diarizer(config)
        result = diarizer.diarize("dummy_audio.wav")
        assert result.num_speakers == 0
        assert len(result.speakers) == 0

    def test_assign_speakers_to_segments(
        self, mock_pipeline_from_pretrained: MagicMock
    ) -> None:
        """Test the logic for assigning speaker labels to transcription segments."""
        transcription_segments = [
            {"start": 0.1, "end": 1.8, "text": "Hello"},
            {"start": 2.6, "end": 3.9, "text": "world"},
            {"start": 4.1, "end": 5.0, "text": "..."},  # No overlap
        ]

        diarization_result = MagicMock()
        diarization_result.segments = [
            {"start": 0.0, "end": 2.0, "speaker_label": "SPEAKER_00"},
            {"start": 2.5, "end": 4.0, "speaker_label": "SPEAKER_01"},
        ]

        labeled_segments = self.diarizer.assign_speakers_to_segments(
            transcription_segments, diarization_result
        )

        assert len(labeled_segments) == EXPECTED_NUM_LABELED_SEGMENTS
        assert labeled_segments[0]["speaker"] == "SPEAKER_00"
        assert labeled_segments[1]["speaker"] == "SPEAKER_01"
        # Falls back to default when no overlap
        assert labeled_segments[2]["speaker"] == "SPEAKER_00"

    def test_assign_speakers_when_diarization_disabled(
        self, mock_pipeline_from_pretrained: MagicMock
    ) -> None:
        """Test speaker assignment when diarization is disabled."""
        self.diarizer.config.enabled = False
        transcription_segments = [{"start": 0, "end": 1, "text": "test"}]
        diarization_result = MagicMock()
        diarization_result.segments = []

        labeled = self.diarizer.assign_speakers_to_segments(
            transcription_segments, diarization_result
        )
        assert labeled[0]["speaker"] == "SPEAKER_00"

    @patch(
        "pyannote.audio.Pipeline.from_pretrained", side_effect=ModelError("Auth failed")
    )
    def test_initialization_failure(
        self,
        mock_from_pretrained_fail: MagicMock,
        mock_pipeline_from_pretrained_ok: MagicMock,
    ) -> None:
        """Test that a ModelError is raised if the pipeline fails to load."""
        with pytest.raises(ModelError):
            Diarizer(self.config)
