"""Multi-format subtitle and transcript generation manager.
Supports SRT, WebVTT, TXT, JSON, TSV, and LRC output formats.

"""

import json
import logging
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import srt
import webvtt

from src.utils.error_handling import FileSystemError, retry_on_error


@dataclass
class TranscriptSegment:
    """Represents a segment of transcribed text with timing and speaker information."""

    start: float
    end: float
    text: str
    speaker: str | None = None
    speaker_confidence: float = 0.0
    words: list[dict[str, Any]] | None = None
    confidence: float = 0.0


class SubtitleManager:
    """Manages the creation and output of transcripts in multiple formats.
    Supports SRT, WebVTT, TXT, JSON, TSV, and LRC formats.

    """

    def __init__(self) -> None:
        """Initialize SubtitleManager."""
        self.logger = logging.getLogger(__name__)

    @retry_on_error()
    def save_transcripts(
        self,
        segments: list[dict[str, Any]],
        output_path: Path,
        formats: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Path]:
        """Save transcripts in multiple formats.

        Args:
        ----
            segments: List of transcript segments with timing and speaker info
            output_path: Base output path (without extension)
            formats: List of formats to generate (default: all supported)
            metadata: Optional metadata to include in output

        Returns:
        -------
            Dict mapping format names to output file paths

        Raises:
        ------
            FileSystemError: If file writing fails

        """
        if formats is None:
            formats = ["srt", "webvtt", "txt", "json", "tsv", "lrc"]

        # Convert segments to internal format
        transcript_segments = self._convert_segments(segments)

        if not transcript_segments:
            self.logger.warning("No segments to save")
            return {}

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        for format_name in formats:
            try:
                file_path = self._save_single_format(
                    transcript_segments, output_path, format_name, metadata
                )
                if file_path:
                    saved_files[format_name] = file_path
                    self.logger.debug(
                        "Saved %s transcript: %s",
                        format_name.upper(),
                        file_path,
                    )

            except Exception as e:
                self.logger.exception("Failed to save %s format: %s", format_name, e)
                msg = f"Failed to save {format_name} format: {e}"
                raise FileSystemError(msg, str(output_path)) from e

        self.logger.info(
            "Saved transcripts in %d formats: %s",
            len(saved_files),
            list(saved_files.keys()),
        )
        return saved_files

    def _save_single_format(
        self,
        transcript_segments: list[TranscriptSegment],
        output_path: Path,
        format_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> Path | None:
        """Save transcript in a single format.

        Args:
        ----
            transcript_segments: List of transcript segments
            output_path: Base output path (without extension)
            format_name: Format to save (srt, webvtt, txt, json, tsv, lrc)
            metadata: Optional metadata to include

        Returns:
        -------
            Path to saved file or None if format not supported

        """
        format_lower = format_name.lower()

        # Create format-specific subdirectory
        format_dir = output_path.parent / format_lower
        format_dir.mkdir(parents=True, exist_ok=True)

        # Create output path in format subdirectory
        format_output_path = format_dir / output_path.name

        if format_lower == "srt":
            return self._save_srt(
                transcript_segments, format_output_path.with_suffix(".srt")
            )
        elif format_lower == "webvtt":
            return self._save_webvtt(
                transcript_segments, format_output_path.with_suffix(".vtt")
            )
        elif format_lower == "txt":
            return self._save_txt(
                transcript_segments, format_output_path.with_suffix(".txt")
            )
        elif format_lower == "json":
            return self._save_json(
                transcript_segments, format_output_path.with_suffix(".json"), metadata
            )
        elif format_lower == "tsv":
            return self._save_tsv(
                transcript_segments, format_output_path.with_suffix(".tsv")
            )
        elif format_lower == "lrc":
            return self._save_lrc(
                transcript_segments, format_output_path.with_suffix(".lrc")
            )
        else:
            self.logger.warning("Unsupported format: %s", format_name)
            return None

    def _convert_segments(
        self,
        segments: list[dict[str, Any]],
    ) -> list[TranscriptSegment]:
        """Convert raw segments to internal TranscriptSegment format."""
        transcript_segments = []

        for segment in segments:
            transcript_segment = TranscriptSegment(
                start=segment.get("start", 0.0),
                end=segment.get("end", 0.0),
                text=segment.get("text", "").strip(),
                speaker=segment.get("speaker"),
                speaker_confidence=segment.get("speaker_confidence", 0.0),
                words=segment.get("words"),
                confidence=segment.get("avg_logprob", 0.0),
            )

            # Only include segments with text
            if transcript_segment.text:
                transcript_segments.append(transcript_segment)

        return transcript_segments

    def _save_srt(self, segments: list[TranscriptSegment], output_path: Path) -> Path:
        """Save transcript in SRT format."""
        # Manual SRT generation to ensure content is written
        lines: list[str] = []
        for i, segment in enumerate(segments, 1):
            # Skip segments missing timing
            if segment.end is None or segment.start is None:
                self.logger.warning(
                    "Skipping segment with missing timing: start=%s, end=%s, text='%s'",
                    segment.start,
                    segment.end,
                    segment.text[:50] + "..."
                    if len(segment.text) > 50
                    else segment.text,
                )
                continue
            # Format timestamps as HH:MM:SS,mmm
            hours = int(segment.start // 3600)
            minutes = int((segment.start % 3600) // 60)
            seconds = int(segment.start % 60)
            millis = int((segment.start - int(segment.start)) * 1000)
            start_ts = f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"
            hours = int(segment.end // 3600)
            minutes = int((segment.end % 3600) // 60)
            seconds = int(segment.end % 60)
            millis = int((segment.end - int(segment.end)) * 1000)
            end_ts = f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"
            # Line content with optional speaker label
            text = self._format_text_with_speaker(segment)
            lines.append(f"{i}\n{start_ts} --> {end_ts}\n{text}\n\n")
        # Write SRT file
        with Path(output_path).open("w", encoding="utf-8") as f:
            f.write("".join(lines))
        return output_path

    def _save_webvtt(
        self,
        segments: list[TranscriptSegment],
        output_path: Path,
    ) -> Path:
        """Save transcript in WebVTT format."""
        vtt = webvtt.WebVTT()

        for segment in segments:
            # Handle None timing values gracefully
            if segment.end is None or segment.start is None:
                self.logger.warning(
                    "Skipping segment with missing timing: start=%s, end=%s, text='%s'",
                    segment.start,
                    segment.end,
                    segment.text[:50] + "..."
                    if len(segment.text) > 50
                    else segment.text,
                )
                continue

            start_time = self._seconds_to_webvtt_time(segment.start)
            end_time = self._seconds_to_webvtt_time(segment.end)

            # Format text with speaker label if available
            text = self._format_text_with_speaker(segment)

            caption = webvtt.Caption(start=start_time, end=end_time, text=text)
            vtt.captions.append(caption)

        # Save WebVTT file
        vtt.save(str(output_path))
        return output_path

    def _save_txt(self, segments: list[TranscriptSegment], output_path: Path) -> Path:
        """Save transcript in plain text format."""
        # Sort segments by start time to ensure full coverage in order
        segments_sorted = sorted(segments, key=lambda s: s.start or 0)
        lines = []
        current_speaker = None

        for segment in segments_sorted:
            # Handle None timing values gracefully
            if segment.start is None:
                self.logger.warning(
                    "Skipping segment with missing start time: start=%s, text='%s'",
                    segment.start,
                    segment.text[:50] + "..."
                    if len(segment.text) > 50
                    else segment.text,
                )
                continue

            # Add speaker change marker if needed
            if segment.speaker and segment.speaker != current_speaker:
                if current_speaker is not None:
                    lines.append("")  # Empty line between speakers
                lines.append(f"[{segment.speaker}]")
                current_speaker = segment.speaker

            # Add timestamp and text
            timestamp = self._format_timestamp(segment.start)
            if segment.speaker:
                lines.append(f"{timestamp} {segment.text}")
            else:
                lines.append(f"{timestamp} {segment.text}")

        # Write text file
        with Path(output_path).open("w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        return output_path

    def _save_json(
        self,
        segments: list[TranscriptSegment],
        output_path: Path,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save transcript in JSON format with detailed information."""
        # Prepare segments data
        segments_data = []
        for i, segment in enumerate(segments):
            # Handle None timing values gracefully
            if segment.end is None or segment.start is None:
                self.logger.warning(
                    "Skipping segment with missing timing: start=%s, end=%s, text='%s'",
                    segment.start,
                    segment.end,
                    segment.text[:50] + "..."
                    if len(segment.text) > 50
                    else segment.text,
                )
                continue

            segment_data = {
                "id": i,
                "start": segment.start,
                "end": segment.end,
                "duration": segment.end - segment.start,
                "text": segment.text,
            }

            if segment.speaker:
                segment_data["speaker"] = segment.speaker
                segment_data["speaker_confidence"] = segment.speaker_confidence

            if segment.words:
                segment_data["words"] = segment.words

            if segment.confidence:
                segment_data["confidence"] = segment.confidence

            segments_data.append(segment_data)

        # Prepare full transcript data
        transcript_data = {
            "segments": segments_data,
            "metadata": metadata or {},
            "statistics": {
                "total_segments": len(segments),
                "total_duration": max((s.end for s in segments), default=0.0),
                "speakers": list({s.speaker for s in segments if s.speaker}),
                "word_count": sum(len(segment.text.split()) for segment in segments),
            },
        }

        # Write JSON file
        with Path(output_path).open("w", encoding="utf-8") as f:
            json.dump(transcript_data, f, indent=2, ensure_ascii=False)

        return output_path

    def _save_tsv(self, segments: list[TranscriptSegment], output_path: Path) -> Path:
        """Save transcript in TSV (Tab-Separated Values) format."""
        # Sort segments by start time to ensure chronological order
        segments_sorted = sorted(segments, key=lambda s: s.start or 0)
        lines = ["start\tend\tduration\tspeaker\ttext\tconfidence"]

        for segment in segments_sorted:
            # Handle None end times gracefully
            if segment.end is None or segment.start is None:
                self.logger.warning(
                    "Skipping segment with missing timing: start=%s, end=%s, text='%s'",
                    segment.start,
                    segment.end,
                    segment.text[:50] + "..."
                    if len(segment.text) > 50
                    else segment.text,
                )
                continue

            duration = segment.end - segment.start
            speaker = segment.speaker or "UNKNOWN"
            confidence = f"{segment.confidence:.3f}" if segment.confidence else "0.000"

            # Escape tabs and newlines in text
            text = segment.text.replace("\t", " ").replace("\n", " ").replace("\r", " ")

            line = f"{segment.start:.3f}\t{segment.end:.3f}\t{duration:.3f}\t{speaker}\t{text}\t{confidence}"
            lines.append(line)

        # Write TSV file
        with Path(output_path).open("w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        return output_path

    def _save_lrc(self, segments: list[TranscriptSegment], output_path: Path) -> Path:
        """Save transcript in LRC (Lyric) format."""
        lines = []

        # Add LRC header
        lines.append("[ti:Transcription]")
        lines.append("[ar:Generated by py-atranscribe]")
        lines.append("[al:]")
        lines.append("[by:py-atranscribe]")
        lines.append("")

        for segment in segments:
            # Convert start time to LRC format [mm:ss.xx]
            minutes = int(segment.start // 60)
            seconds = segment.start % 60
            lrc_time = f"[{minutes:02d}:{seconds:05.2f}]"

            # Format text with speaker if available
            text = self._format_text_with_speaker(segment, prefix="")

            lines.append(f"{lrc_time}{text}")

        # Write LRC file
        with Path(output_path).open("w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return output_path

    def _format_text_with_speaker(
        self,
        segment: TranscriptSegment,
        prefix: str = "",
    ) -> str:
        """Format text with optional speaker label."""
        if segment.speaker:
            return f"{prefix}[{segment.speaker}] {segment.text}"
        return f"{prefix}{segment.text}"

    def _format_timestamp(self, seconds: float | None) -> str:
        """Format seconds as [HH:MM:SS] timestamp."""
        if seconds is None:
            return "[00:00:00.00]"

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"[{hours:02d}:{minutes:02d}:{secs:05.2f}]"

    def _seconds_to_webvtt_time(self, seconds: float | None) -> str:
        """Convert seconds to WebVTT time format."""
        if seconds is None:
            return "00:00:00.000"

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

    def get_transcript_summary(
        self,
        segments: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate a summary of the transcript.

        Args:
        ----
            segments: List of transcript segments

        Returns:
        -------
            Dict with transcript statistics and summary

        """
        if not segments:
            return {
                "total_segments": 0,
                "total_duration": 0.0,
                "word_count": 0,
                "speakers": [],
                "average_segment_length": 0.0,
                "speech_rate": 0.0,
            }

        total_duration = max((s.get("end", 0) for s in segments), default=0.0)
        total_words = sum(len(s.get("text", "").split()) for s in segments)
        speakers = list({s.get("speaker") for s in segments if s.get("speaker")})

        # Calculate speaker statistics
        speaker_stats = {}
        for segment in segments:
            speaker = segment.get("speaker", "UNKNOWN")
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    "duration": 0.0,
                    "segments": 0,
                    "words": 0,
                }

            speaker_stats[speaker]["duration"] += segment.get("end", 0) - segment.get(
                "start",
                0,
            )
            speaker_stats[speaker]["segments"] += 1
            speaker_stats[speaker]["words"] += len(segment.get("text", "").split())

        return {
            "total_segments": len(segments),
            "total_duration": total_duration,
            "word_count": total_words,
            "speakers": speakers,
            "num_speakers": len(speakers),
            "average_segment_length": total_duration / len(segments)
            if segments
            else 0.0,
            "speech_rate": total_words / (total_duration / 60)
            if total_duration > 0
            else 0.0,  # words per minute
            "speaker_statistics": speaker_stats,
        }

    def merge_short_segments(
        self,
        segments: list[dict[str, Any]],
        min_duration: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Merge very short segments with adjacent segments to improve readability.

        Args:
        ----
            segments: List of segments to merge
            min_duration: Minimum segment duration in seconds

        Returns:
        -------
            List of merged segments

        """
        if not segments:
            return segments

        merged_segments = []
        current_segment = None

        for segment in segments:
            duration = segment.get("end", 0) - segment.get("start", 0)

            if (
                duration < min_duration
                and current_segment is not None
                and (
                    segment.get("speaker") == current_segment.get("speaker")
                    or not segment.get("speaker")
                    or not current_segment.get("speaker")
                )
            ):
                # Merge text and extend end time
                current_segment["text"] += " " + segment.get("text", "")
                current_segment["end"] = segment.get("end", current_segment["end"])

                # Update confidence (weighted average)
                if "confidence" in segment and "confidence" in current_segment:
                    old_duration = (
                        current_segment["end"] - current_segment["start"] - duration
                    )
                    total_duration = current_segment["end"] - current_segment["start"]

                    if total_duration > 0:
                        current_segment["confidence"] = (
                            current_segment["confidence"] * old_duration
                            + segment.get("confidence", 0) * duration
                        ) / total_duration

                continue

            # Add current segment to results if it exists
            if current_segment is not None:
                merged_segments.append(current_segment)

            # Start new segment
            current_segment = segment.copy()

        # Add the last segment
        if current_segment is not None:
            merged_segments.append(current_segment)

        self.logger.debug(
            "Merged %d segments into %d segments",
            len(segments),
            len(merged_segments),
        )
        return merged_segments

    def filter_low_confidence_segments(
        self,
        segments: list[dict[str, Any]],
        min_confidence: float = -1.0,
    ) -> list[dict[str, Any]]:
        """Filter out segments with very low confidence scores.

        Args:
        ----
            segments: List of segments to filter
            min_confidence: Minimum confidence threshold

        Returns:
        -------
            List of filtered segments

        """
        very_permissive_threshold = -2.0
        if min_confidence <= very_permissive_threshold:  # Very permissive threshold
            return segments

        filtered_segments = []
        for segment in segments:
            confidence = segment.get("avg_logprob", segment.get("confidence", 0.0))

            if confidence >= min_confidence:
                filtered_segments.append(segment)
            else:
                self.logger.debug(
                    "Filtered low confidence segment: %.3f < %.3f",
                    confidence,
                    min_confidence,
                )

        self.logger.debug(
            "Filtered %d low confidence segments",
            len(segments) - len(filtered_segments),
        )
        return filtered_segments
