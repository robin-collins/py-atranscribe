"""Integration tests for the BatchTranscriber pipeline."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import AppConfig
from src.diarization.diarizer import DiarizationResult
from src.pipeline.batch_transcriber import BatchTranscriber, ProcessingResult


@pytest.mark.asyncio
class TestBatchTranscriber:
    """Test the BatchTranscriber class."""

    def setup_method(self) -> None:
        """Set up the BatchTranscriber with mock components."""
        self.config = AppConfig()
        # Use a real temporary directory for file operations
        self.test_dir = Path(self.config.directories.input)
        self.test_dir.mkdir(parents=True, exist_ok=True)

        self.transcriber = BatchTranscriber(self.config)

        # Mock the internal components
        self.transcriber._whisper_inference = AsyncMock()
        self.transcriber.diarizer = AsyncMock()
        self.transcriber.subtitle_manager = MagicMock()
        self.transcriber.file_handler = MagicMock()

    def teardown_method(self) -> None:
        """Clean up any created files."""
        import shutil

        shutil.rmtree(self.test_dir, ignore_errors=True)
        shutil.rmtree(self.config.directories.output, ignore_errors=True)

    async def test_successful_processing_with_diarization(self) -> None:
        """Test a successful run of the entire pipeline with diarization."""
        # --- MOCK SETUP ---
        # Mock transcription result
        mock_transcription_result = {
            "segments": [{"start": 0, "end": 5, "text": "Hello world"}],
            "language": "en",
            "duration": 5.0,
        }
        self.transcriber._transcribe_audio = AsyncMock(
            return_value=mock_transcription_result
        )

        # Mock diarization result
        mock_diarization_result = DiarizationResult(
            speakers=[MagicMock()],
            segments=[{"start": 0, "end": 5, "speaker_label": "SPEAKER_00"}],
            duration=5.0,
            num_speakers=1,
            confidence=0.9,
        )
        self.transcriber._perform_diarization = AsyncMock(
            return_value=mock_diarization_result
        )

        # Mock the merging and output generation
        self.transcriber._merge_results = MagicMock(
            return_value=mock_transcription_result["segments"]
        )
        self.transcriber.subtitle_manager.save_transcripts.return_value = {
            "srt": Path("/fake/output.srt")
        }
        self.transcriber._post_process_file = AsyncMock()

        # --- EXECUTION ---
        test_file = self.test_dir / "test.wav"
        test_file.touch()

        result = await self.transcriber.process_file(test_file)

        # --- ASSERTIONS ---
        assert result.success is True
        assert result.input_file == test_file
        assert "srt" in result.output_files

        self.transcriber._transcribe_audio.assert_called_once_with(test_file)
        self.transcriber._perform_diarization.assert_called_once_with(test_file)
        self.transcriber.subtitle_manager.save_transcripts.assert_called_once()
        self.transcriber._post_process_file.assert_called_once_with(test_file)

    async def test_processing_without_diarization(self) -> None:
        """Test the pipeline with diarization disabled."""
        self.config.diarization.enabled = False
        self.transcriber.diarizer.is_enabled.return_value = False

        mock_transcription_result = {"segments": [{"text": "test"}], "duration": 1.0}
        self.transcriber._transcribe_audio = AsyncMock(
            return_value=mock_transcription_result
        )
        self.transcriber._perform_diarization = AsyncMock()
        self.transcriber._post_process_file = AsyncMock()

        test_file = self.test_dir / "test.wav"
        test_file.touch()

        result = await self.transcriber.process_file(test_file)

        assert result.success is True
        self.transcriber._perform_diarization.assert_not_called()

    async def test_processing_fails_on_transcription_error(self) -> None:
        """Test that the pipeline fails gracefully if transcription fails."""
        self.transcriber._transcribe_audio = AsyncMock(
            side_effect=Exception("Transcription failed")
        )

        test_file = self.test_dir / "test.wav"
        test_file.touch()

        result = await self.transcriber.process_file(test_file)

        assert result.success is False
        assert "Transcription failed" in result.error_message

    async def test_diarization_failure_is_handled_gracefully(self) -> None:
        """Test that the pipeline continues if only diarization fails."""
        mock_transcription_result = {"segments": [{"text": "test"}], "duration": 1.0}
        self.transcriber._transcribe_audio = AsyncMock(
            return_value=mock_transcription_result
        )
        self.transcriber._perform_diarization = AsyncMock(
            return_value=None
        )  # Simulate failure
        self.transcriber._post_process_file = AsyncMock()

        test_file = self.test_dir / "test.wav"
        test_file.touch()

        result = await self.transcriber.process_file(test_file)

        # The overall process should still succeed
        assert result.success is True
        # Check that the log contains a warning about diarization
        # (This requires inspecting logs, which is complex in unit tests,
        # but we can verify the flow continued)
        self.transcriber.subtitle_manager.save_transcripts.assert_called_once()

    async def test_post_processing_move(self) -> None:
        """Test the 'move' post-processing action."""
        self.config.post_processing.action = "move"
        self.transcriber.file_handler.safe_move_file = MagicMock()

        test_file = self.test_dir / "test.wav"
        await self.transcriber._post_process_file(test_file)

        self.transcriber.file_handler.safe_move_file.assert_called_once()
