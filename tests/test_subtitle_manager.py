"""
Unit tests for subtitle manager functionality.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

import pytest

from src.output.subtitle_manager import SubtitleManager, TranscriptSegment


class TestTranscriptSegment:
    """Test TranscriptSegment dataclass."""
    
    def test_create_segment(self):
        """Test creating a TranscriptSegment."""
        segment = TranscriptSegment(
            start=0.0,
            end=5.0,
            text="Hello world",
            speaker="SPEAKER_00"
        )
        assert segment.start == 0.0
        assert segment.end == 5.0
        assert segment.text == "Hello world"
        assert segment.speaker == "SPEAKER_00"


class TestSubtitleManager:
    """Test SubtitleManager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = SubtitleManager()
        self.sample_segments = [
            {
                "start": 0.0,
                "end": 2.5,
                "text": "Hello world",
                "speaker": "SPEAKER_00",
                "speaker_confidence": 0.95,
                "avg_logprob": -0.2
            },
            {
                "start": 2.5,
                "end": 5.0,
                "text": "How are you?",
                "speaker": "SPEAKER_01",
                "speaker_confidence": 0.88,
                "avg_logprob": -0.3
            }
        ]
    
    def test_convert_segments(self):
        """Test converting raw segments to TranscriptSegment objects."""
        segments = self.manager._convert_segments(self.sample_segments)
        
        assert len(segments) == 2
        assert isinstance(segments[0], TranscriptSegment)
        assert segments[0].start == 0.0
        assert segments[0].text == "Hello world"
        assert segments[0].speaker == "SPEAKER_00"
    
    def test_convert_segments_filters_empty_text(self):
        """Test that segments with empty text are filtered out."""
        segments_with_empty = self.sample_segments + [
            {"start": 5.0, "end": 6.0, "text": "", "speaker": "SPEAKER_00"},
            {"start": 6.0, "end": 7.0, "text": "   ", "speaker": "SPEAKER_00"}
        ]
        
        segments = self.manager._convert_segments(segments_with_empty)
        assert len(segments) == 2  # Empty segments filtered out
    
    def test_format_text_with_speaker(self):
        """Test formatting text with speaker labels."""
        segment = TranscriptSegment(0.0, 2.0, "Hello", speaker="SPEAKER_00")
        
        # With speaker
        result = self.manager._format_text_with_speaker(segment)
        assert result == "[SPEAKER_00] Hello"
        
        # Without speaker
        segment_no_speaker = TranscriptSegment(0.0, 2.0, "Hello")
        result = self.manager._format_text_with_speaker(segment_no_speaker)
        assert result == "Hello"
    
    def test_format_timestamp(self):
        """Test timestamp formatting."""
        # Test various timestamps
        assert self.manager._format_timestamp(0.0) == "[00:00:00.00]"
        assert self.manager._format_timestamp(65.5) == "[00:01:05.50]"
        assert self.manager._format_timestamp(3661.25) == "[01:01:01.25]"
    
    def test_seconds_to_webvtt_time(self):
        """Test WebVTT time formatting."""
        assert self.manager._seconds_to_webvtt_time(0.0) == "00:00:00.000"
        assert self.manager._seconds_to_webvtt_time(65.5) == "00:01:05.500"
        assert self.manager._seconds_to_webvtt_time(3661.25) == "01:01:01.250"


class TestSubtitleFormats:
    """Test different subtitle format generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = SubtitleManager()
        self.segments = [
            TranscriptSegment(0.0, 2.5, "Hello world", "SPEAKER_00"),
            TranscriptSegment(2.5, 5.0, "How are you?", "SPEAKER_01")
        ]
    
    def test_save_srt(self):
        """Test SRT format generation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as f:
            output_path = Path(f.name)
        
        try:
            result_path = self.manager._save_srt(self.segments, output_path)
            assert result_path == output_path
            
            # Check file content
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert "1" in content  # Subtitle index
            assert "00:00:00,000 --> 00:00:02,500" in content
            assert "[SPEAKER_00] Hello world" in content
            assert "00:00:02,500 --> 00:00:05,000" in content
            assert "[SPEAKER_01] How are you?" in content
            
        finally:
            output_path.unlink()
    
    def test_save_webvtt(self):
        """Test WebVTT format generation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vtt', delete=False) as f:
            output_path = Path(f.name)
        
        try:
            result_path = self.manager._save_webvtt(self.segments, output_path)
            assert result_path == output_path
            
            # Check file content
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert "WEBVTT" in content
            assert "00:00:00.000 --> 00:00:02.500" in content
            assert "[SPEAKER_00] Hello world" in content
            
        finally:
            output_path.unlink()
    
    def test_save_txt(self):
        """Test plain text format generation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            output_path = Path(f.name)
        
        try:
            result_path = self.manager._save_txt(self.segments, output_path)
            assert result_path == output_path
            
            # Check file content
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert "[SPEAKER_00]" in content
            assert "[00:00:00.00] Hello world" in content
            assert "[SPEAKER_01]" in content
            assert "[00:00:02.50] How are you?" in content
            
        finally:
            output_path.unlink()
    
    def test_save_json(self):
        """Test JSON format generation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = Path(f.name)
        
        metadata = {"test_key": "test_value"}
        
        try:
            result_path = self.manager._save_json(self.segments, output_path, metadata)
            assert result_path == output_path
            
            # Check file content
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert "segments" in data
            assert "metadata" in data
            assert "statistics" in data
            assert len(data["segments"]) == 2
            assert data["metadata"]["test_key"] == "test_value"
            assert data["statistics"]["total_segments"] == 2
            
        finally:
            output_path.unlink()
    
    def test_save_tsv(self):
        """Test TSV format generation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            output_path = Path(f.name)
        
        try:
            result_path = self.manager._save_tsv(self.segments, output_path)
            assert result_path == output_path
            
            # Check file content
            with open(output_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Check header
            assert "start\tend\tduration\tspeaker\ttext\tconfidence" in lines[0]
            
            # Check data rows
            assert "0.000\t2.500\t2.500\tSPEAKER_00\tHello world" in lines[1]
            assert "2.500\t5.000\t2.500\tSPEAKER_01\tHow are you?" in lines[2]
            
        finally:
            output_path.unlink()
    
    def test_save_lrc(self):
        """Test LRC format generation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lrc', delete=False) as f:
            output_path = Path(f.name)
        
        try:
            result_path = self.manager._save_lrc(self.segments, output_path)
            assert result_path == output_path
            
            # Check file content
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert "[ti:Transcription]" in content
            assert "[ar:Generated by py-atranscribe]" in content
            assert "[00:00.00][SPEAKER_00] Hello world" in content
            assert "[00:02.50][SPEAKER_01] How are you?" in content
            
        finally:
            output_path.unlink()


class TestMultipleFormats:
    """Test saving multiple formats simultaneously."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = SubtitleManager()
    
    def test_save_transcripts_multiple_formats(self):
        """Test saving transcripts in multiple formats."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_output"
            
            segments = [
                {"start": 0.0, "end": 2.0, "text": "Hello", "speaker": "SPEAKER_00"}
            ]
            
            formats = ["srt", "txt", "json"]
            result = self.manager.save_transcripts(
                segments=segments,
                output_path=output_path,
                formats=formats
            )
            
            assert len(result) == 3
            assert "srt" in result
            assert "txt" in result
            assert "json" in result
            
            # Check that files were created
            assert result["srt"].exists()
            assert result["txt"].exists()
            assert result["json"].exists()
    
    def test_save_transcripts_empty_segments(self):
        """Test saving with empty segments."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_output"
            
            segments = []
            result = self.manager.save_transcripts(
                segments=segments,
                output_path=output_path
            )
            
            assert result == {}
    
    def test_save_transcripts_unsupported_format(self):
        """Test saving with unsupported format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_output"
            
            segments = [
                {"start": 0.0, "end": 2.0, "text": "Hello", "speaker": "SPEAKER_00"}
            ]
            
            formats = ["srt", "unsupported"]
            result = self.manager.save_transcripts(
                segments=segments,
                output_path=output_path,
                formats=formats
            )
            
            # Should save supported format, skip unsupported
            assert len(result) == 1
            assert "srt" in result
            assert "unsupported" not in result


class TestTranscriptSummary:
    """Test transcript summary generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = SubtitleManager()
    
    def test_get_transcript_summary(self):
        """Test transcript summary generation."""
        segments = [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "Hello world this is a test",
                "speaker": "SPEAKER_00"
            },
            {
                "start": 2.0,
                "end": 4.0,
                "text": "How are you doing today",
                "speaker": "SPEAKER_01"
            },
            {
                "start": 4.0,
                "end": 6.0,
                "text": "I am doing well",
                "speaker": "SPEAKER_00"
            }
        ]
        
        summary = self.manager.get_transcript_summary(segments)
        
        assert summary["total_segments"] == 3
        assert summary["total_duration"] == 6.0
        assert summary["word_count"] == 13  # Total words across all segments
        assert summary["num_speakers"] == 2
        assert "SPEAKER_00" in summary["speakers"]
        assert "SPEAKER_01" in summary["speakers"]
        assert summary["average_segment_length"] == 2.0
        assert summary["speech_rate"] == 130.0  # 13 words / (6 seconds / 60)
    
    def test_get_transcript_summary_empty(self):
        """Test transcript summary with empty segments."""
        summary = self.manager.get_transcript_summary([])
        
        assert summary["total_segments"] == 0
        assert summary["total_duration"] == 0.0
        assert summary["word_count"] == 0
        assert summary["speakers"] == []
        assert summary["num_speakers"] == 0
        assert summary["speech_rate"] == 0.0


class TestSegmentProcessing:
    """Test segment processing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = SubtitleManager()
    
    def test_merge_short_segments(self):
        """Test merging short segments."""
        segments = [
            {"start": 0.0, "end": 0.5, "text": "Hi", "speaker": "SPEAKER_00"},  # Short
            {"start": 0.5, "end": 2.0, "text": "How are you?", "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 2.3, "text": "Good", "speaker": "SPEAKER_01"},  # Short
            {"start": 2.3, "end": 4.0, "text": "Thanks for asking", "speaker": "SPEAKER_01"}
        ]
        
        merged = self.manager.merge_short_segments(segments, min_duration=1.0)
        
        # Should merge the short segments with their neighbors
        assert len(merged) == 2
        assert merged[0]["text"] == "Hi How are you?"
        assert merged[0]["start"] == 0.0
        assert merged[0]["end"] == 2.0
        assert merged[1]["text"] == "Good Thanks for asking"
        assert merged[1]["start"] == 2.0
        assert merged[1]["end"] == 4.0
    
    def test_merge_short_segments_different_speakers(self):
        """Test that segments with different speakers are not merged."""
        segments = [
            {"start": 0.0, "end": 0.5, "text": "Hi", "speaker": "SPEAKER_00"},  # Short
            {"start": 0.5, "end": 2.0, "text": "How are you?", "speaker": "SPEAKER_01"}  # Different speaker
        ]
        
        merged = self.manager.merge_short_segments(segments, min_duration=1.0)
        
        # Should not merge due to different speakers
        assert len(merged) == 2
    
    def test_filter_low_confidence_segments(self):
        """Test filtering low confidence segments."""
        segments = [
            {"start": 0.0, "end": 2.0, "text": "Good quality", "avg_logprob": -0.2},
            {"start": 2.0, "end": 4.0, "text": "Poor quality", "avg_logprob": -2.5},
            {"start": 4.0, "end": 6.0, "text": "Medium quality", "avg_logprob": -1.0}
        ]
        
        filtered = self.manager.filter_low_confidence_segments(segments, min_confidence=-1.5)
        
        # Should keep only segments above threshold
        assert len(filtered) == 2
        assert filtered[0]["text"] == "Good quality"
        assert filtered[1]["text"] == "Medium quality"