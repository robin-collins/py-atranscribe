"""Speaker diarization using pyannote.audio for identifying and labeling speakers in audio.

Provides integration with transcription pipeline for speaker-labeled transcripts.
"""

import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import torch

try:
    import torchaudio
    from pyannote.audio import Pipeline
    from pyannote.core import Annotation
except ImportError as e:
    logging.getLogger(__name__).exception("Failed to import pyannote.audio")
    msg = (
        "pyannote.audio is required for speaker diarization. "
        "Install with: pip install pyannote.audio"
    )
    raise ImportError(msg) from e

from src.config import DiarizationConfig
from src.utils.error_handling import ModelError, graceful_degradation, retry_on_error


@dataclass
class Speaker:
    """Information about a detected speaker."""

    id: str
    label: str
    segments: list[tuple[float, float]]  # List of (start, end) time segments
    total_duration: float
    confidence: float = 0.0


@dataclass
class DiarizationResult:
    """Result of speaker diarization process."""

    speakers: list[Speaker]
    segments: list[dict[str, Any]]  # Segments with speaker labels
    duration: float
    num_speakers: int
    confidence: float


class Diarizer:
    """Speaker diarization class using pyannote.audio.

    Identifies and labels different speakers in audio files.
    """

    _pipeline_cache: ClassVar[dict[str, Pipeline]] = {}

    def __init__(self, config: DiarizationConfig) -> None:
        """Initialize Diarizer with configuration.

        Args:
            config: Diarization configuration

        Raises:
            ModelError: If initialization fails

        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.pipeline: Pipeline | None = None

        if not config.enabled:
            self.logger.info("Speaker diarization is disabled")
            return

        # Validate HuggingFace token
        if not config.hf_token:
            self.logger.warning("HF_TOKEN not provided, diarization may fail")
        else:
            os.environ["HUGGINGFACE_HUB_TOKEN"] = config.hf_token

        # Initialize pipeline
        self._initialize_pipeline()

    def _initialize_pipeline(self) -> None:
        """Initialize the diarization pipeline."""
        if not self.config.enabled:
            return

        try:
            cache_key = f"{self.config.model}_{self.config.device}"

            if cache_key in self._pipeline_cache:
                self.pipeline = self._pipeline_cache[cache_key]
                self.logger.debug("Using cached diarization pipeline: %s", cache_key)
                return

            self.logger.info("Initializing diarization pipeline: %s", self.config.model)

            # Announce model download start
            self.logger.info(
                "🔽 Downloading diarization model '%s'...", self.config.model
            )
            print(
                f"🔽 Downloading diarization model '{self.config.model}'...", flush=True
            )

            # Load the diarization pipeline
            self.pipeline = Pipeline.from_pretrained(
                self.config.model,
                use_auth_token=self.config.hf_token if self.config.hf_token else True,
            )

            # Announce download completion
            self.logger.info(
                "✅ Diarization model '%s' download completed successfully",
                self.config.model,
            )
            print(
                f"✅ Diarization model '{self.config.model}' download completed successfully",
                flush=True,
            )

            # Set device
            device = self._get_optimal_device()
            if device == "cuda" and torch.cuda.is_available():
                self.pipeline = self.pipeline.to(torch.device("cuda"))
                self.logger.info("Diarization pipeline moved to GPU")
            else:
                self.logger.info("Using CPU for diarization")

            # Cache the pipeline
            self._pipeline_cache[cache_key] = self.pipeline

            self.logger.info("Diarization pipeline initialized successfully")

        except Exception as e:
            self.logger.exception("Failed to initialize diarization pipeline")
            if "unauthorized" in str(e).lower() or "authentication" in str(e).lower():
                msg = f"HuggingFace authentication failed. Please check HF_TOKEN: {e}"
                raise ModelError(
                    msg,
                ) from e
            msg = f"Failed to initialize diarization pipeline: {e}"
            raise ModelError(msg) from e

    def _get_optimal_device(self) -> str:
        """Determine optimal device for diarization."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                try:
                    # Check GPU memory
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
                    gpu_memory_gb = gpu_memory / (1024**3)

                    # Diarization needs less memory than transcription
                    if (
                        gpu_memory_gb >= 1.0
                        and not graceful_degradation.should_disable_feature(
                            "diarization",
                        )
                    ):
                        return "cuda"
                    self.logger.warning(
                        "Insufficient GPU memory for diarization, using CPU",
                    )
                    return "cpu"  # noqa: TRY300
                except (RuntimeError, OSError, ImportError) as e:
                    self.logger.warning("Error checking GPU for diarization: %s", e)
                    return "cpu"
            else:
                return "cpu"
        return self.config.device

    @retry_on_error()
    def diarize(self, audio_path: str) -> DiarizationResult | None:
        """Perform speaker diarization on audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            DiarizationResult | None: Diarization results with speaker information, or None if diarization fails

        """
        if not self.config.enabled:
            # Return empty result if diarization is disabled
            return DiarizationResult(
                speakers=[],
                segments=[],
                duration=0.0,
                num_speakers=0,
                confidence=0.0,
            )

        if self.pipeline is None:
            msg = "Diarization pipeline not initialized"
            raise ModelError(msg)

        try:
            self.logger.info("Starting speaker diarization: %s", audio_path)

            # Get audio duration
            audio_info = torchaudio.info(audio_path)
            duration = audio_info.num_frames / audio_info.sample_rate

            # Perform diarization
            diarization_params = {}
            if self.config.min_speakers is not None:
                diarization_params["min_speakers"] = self.config.min_speakers
            if self.config.max_speakers is not None:
                diarization_params["max_speakers"] = self.config.max_speakers

            # Apply the pipeline with error handling
            try:
                diarization = self.pipeline(audio_path, **diarization_params)
            except Exception as e:
                self.logger.warning("Diarization pipeline failed: %s", e)
                # Try without speaker constraints
                if diarization_params:
                    self.logger.info("Retrying diarization without speaker constraints")
                    diarization = self.pipeline(audio_path)
                else:
                    raise

            # Validate diarization results
            if diarization is None or not diarization.labels():
                self.logger.warning(
                    "Diarization produced no results for %s",
                    audio_path,
                )
                return DiarizationResult(
                    speakers=[],
                    segments=[],
                    duration=duration,
                    num_speakers=0,
                    confidence=0.0,
                )

            # Process results
            speakers = self._extract_speakers(diarization, duration)
            segments = self._create_segments(diarization)

            result = DiarizationResult(
                speakers=speakers,
                segments=segments,
                duration=duration,
                num_speakers=len(speakers),
                confidence=self._calculate_confidence(diarization),
            )

            self.logger.info(
                "Diarization completed: %d speakers detected, %d segments",
                len(speakers),
                len(segments),
            )
            return result  # BUG FIX: Added return statement

        except Exception:
            self.logger.exception("Diarization failed for %s", audio_path)
            # Return None on failure to allow graceful degradation
            return None

    def _extract_speakers(
        self,
        diarization: Annotation,
        total_duration: float,
    ) -> list[Speaker]:
        """Extract speaker information from diarization annotation.

        Args:
            diarization: Pyannote diarization annotation
            total_duration: Total audio duration

        Returns:
            List of detected speakers

        """
        speakers = {}

        try:
            for segment, _, speaker_label in diarization.itertracks(yield_label=True):
                speaker_id = str(speaker_label)

                if speaker_id not in speakers:
                    speakers[speaker_id] = {
                        "segments": [],
                        "total_duration": 0.0,
                    }

                speakers[speaker_id]["segments"].append((segment.start, segment.end))
                speakers[speaker_id]["total_duration"] += segment.duration

            # Create Speaker objects
            speaker_list = []
            for i, (speaker_id, info) in enumerate(sorted(speakers.items())):
                speaker = Speaker(
                    id=speaker_id,
                    label=f"SPEAKER_{i:02d}",
                    segments=info["segments"],
                    total_duration=info["total_duration"],
                    confidence=info["total_duration"] / total_duration
                    if total_duration > 0
                    else 0.0,
                )
                speaker_list.append(speaker)

        except (AttributeError, KeyError, ValueError, TypeError) as e:
            self.logger.warning("Error extracting speakers: %s", e)
            return []
        else:
            return speaker_list

    def _create_segments(self, diarization: Annotation) -> list[dict[str, Any]]:
        """Create time-based segments with speaker labels.

        Args:
            diarization: Pyannote diarization annotation

        Returns:
            List of segments with speaker information

        """
        segments = []

        try:
            for segment, _, speaker_label in diarization.itertracks(yield_label=True):
                segment_dict = {
                    "start": segment.start,
                    "end": segment.end,
                    "duration": segment.duration,
                    "speaker": str(speaker_label),
                    "speaker_label": f"SPEAKER_{self._get_speaker_index(str(speaker_label), diarization):02d}",
                }
                segments.append(segment_dict)

            # Sort segments by start time
            segments.sort(key=lambda x: x["start"])

        except (AttributeError, KeyError, ValueError, TypeError) as e:
            self.logger.warning("Error creating segments: %s", e)
            return []
        else:
            return segments

    def _get_speaker_index(self, speaker_id: str, diarization: Annotation) -> int:
        """Get consistent speaker index for labeling."""
        try:
            unique_speakers = sorted(
                {
                    str(label)
                    for _, _, label in diarization.itertracks(yield_label=True)
                },
            )
            return unique_speakers.index(speaker_id)
        except (ValueError, Exception):
            # Return 0 as fallback if speaker_id not found or other error
            return 0

    def _calculate_confidence(self, diarization: Annotation) -> float:
        """Calculate overall confidence score for diarization.

        Args:
            diarization: Pyannote diarization annotation

        Returns:
            Confidence score between 0 and 1

        """
        try:
            if not diarization.labels():
                return 0.0

            # Simple confidence based on coverage - with defensive programming
            total_duration = 0.0
            for segment, _, _ in diarization.itertracks(yield_label=True):
                total_duration += segment.duration

            if total_duration == 0:
                return 0.0

            # Calculate confidence based on segment consistency
            speaker_durations = {}
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                speaker_id = str(speaker)
                if speaker_id not in speaker_durations:
                    speaker_durations[speaker_id] = 0.0
                speaker_durations[speaker_id] += segment.duration

            # Confidence is higher when speakers have more balanced speaking time
            if len(speaker_durations) <= 1:
                return 0.8  # Single speaker is usually reliable

            durations = list(speaker_durations.values())
            balance_score = min(durations) / max(durations) if max(durations) > 0 else 0
            coverage_score = min(1.0, total_duration / 60.0)  # Normalize by 60 seconds

            return balance_score * 0.6 + coverage_score * 0.4

        except (
            AttributeError,
            KeyError,
            ValueError,
            TypeError,
            ZeroDivisionError,
        ) as e:
            self.logger.warning("Error calculating confidence: %s", e)
            return 0.0

    def _merge_speaker_turns(
        self, diarization_result: DiarizationResult
    ) -> list[dict[str, Any]]:
        """Merge consecutive diarization segments from the same speaker into single turns.

        This is a direct adaptation of the pre-processing logic in insanely-fast-whisper.

        Args:
            diarization_result: The raw diarization result containing speaker segments.

        Returns:
            A list of merged speaker turns, each with a start, end, and speaker label.

        """
        dia_segments = diarization_result.segments
        if not dia_segments:
            return []

        merged_speaker_turns = []
        prev_segment = cur_segment = dia_segments[0]

        for i in range(1, len(dia_segments)):
            cur_segment = dia_segments[i]
            # Check if speaker has changed
            if cur_segment["speaker_label"] != prev_segment["speaker_label"]:
                # Add the merged segment to the new list
                merged_speaker_turns.append(
                    {
                        "start": prev_segment["start"],
                        "end": cur_segment["start"],  # End of turn is start of next
                        "speaker": prev_segment["speaker_label"],
                    }
                )
                prev_segment = cur_segment

        # Add the final speaker turn
        merged_speaker_turns.append(
            {
                "start": prev_segment["start"],
                "end": cur_segment["end"],
                "speaker": prev_segment["speaker_label"],
            }
        )

        return merged_speaker_turns

    def _align_segments_by_timestamp(
        self,
        merged_turns: list[dict[str, Any]],
        transcript_segments: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Align speaker turns with transcription segments using the closest end timestamp.

        This is a direct adaptation of the post-processing logic in insanely-fast-whisper.

        Args:
            merged_turns: A list of merged speaker turns.
            transcript_segments: A list of segments from the transcription model.

        Returns:
            The list of transcription segments, now with speaker labels assigned.

        """
        transcript = list(transcript_segments)
        end_timestamps = np.array(
            [
                chunk["timestamp"][1]
                if "timestamp" in chunk
                and isinstance(chunk["timestamp"], (list, tuple))
                and len(chunk["timestamp"]) >= 2
                and chunk["timestamp"][1] is not None
                else sys.float_info.max
                for chunk in transcript
            ]
        )

        labeled_segments = []

        for turn in merged_turns:
            if not transcript:
                break  # No more transcription chunks to process

            end_time = turn["end"]

            # Find the ASR chunk with the end timestamp closest to the speaker turn's end time
            upto_idx = np.argmin(np.abs(end_timestamps - end_time))

            # Assign the current speaker to all transcription chunks up to that index
            for i in range(upto_idx + 1):
                chunk_to_label = transcript[i].copy()
                chunk_to_label["speaker"] = turn["speaker"]
                chunk_to_label["speaker_confidence"] = 1.0  # Method is deterministic
                labeled_segments.append(chunk_to_label)

            # Crop the transcript and timestamp lists for the next iteration
            transcript = transcript[upto_idx + 1 :]
            end_timestamps = end_timestamps[upto_idx + 1 :]
            if len(end_timestamps) == 0:
                break

        # If any transcription segments remain, assign them to UNKNOWN
        if transcript:
            for remaining_chunk in transcript:
                chunk_with_speaker = remaining_chunk.copy()
                chunk_with_speaker["speaker"] = "UNKNOWN"
                chunk_with_speaker["speaker_confidence"] = 0.0
                labeled_segments.append(chunk_with_speaker)

        return labeled_segments

    def assign_speakers_to_segments(
        self,
        transcription_segments: list[dict[str, Any]],
        diarization_result: DiarizationResult,
    ) -> list[dict[str, Any]]:
        """Assign speaker labels to transcription segments using the insanely-fast-whisper algorithm.

        This function orchestrates the merging of speaker turns and the alignment with
        transcription segments.

        Args:
            transcription_segments: Segments from transcription.
            diarization_result: Results from diarization.

        Returns:
            A list of transcription segments with speaker labels assigned.

        """
        if (
            not self.config.enabled
            or not diarization_result
            or not diarization_result.segments
        ):
            # Fallback: assign a default speaker if diarization is disabled or empty.
            for segment in transcription_segments:
                segment["speaker"] = "SPEAKER_00"
                segment["speaker_confidence"] = 0.0
            return transcription_segments

        # Step 1: Merge consecutive speaker segments into "turns".
        merged_speaker_turns = self._merge_speaker_turns(diarization_result)

        # Step 2: Align the speaker turns with transcription segments.
        labeled_segments = self._align_segments_by_timestamp(
            merged_speaker_turns, transcription_segments
        )

        self.logger.debug(
            "Assigned speakers to %d transcription segments using IFW method",
            len(labeled_segments),
        )

        return labeled_segments

    def get_speaker_statistics(
        self,
        diarization_result: DiarizationResult,
    ) -> dict[str, Any]:
        """Get statistics about detected speakers.

        Args:
            diarization_result: Diarization results

        Returns:
            Dictionary with speaker statistics

        """
        if not diarization_result.speakers:
            return {"num_speakers": 0, "total_speech_duration": 0.0, "speakers": []}

        total_speech = sum(
            speaker.total_duration for speaker in diarization_result.speakers
        )

        speaker_stats = []
        for speaker in diarization_result.speakers:
            stats = {
                "id": speaker.id,
                "label": speaker.label,
                "duration": speaker.total_duration,
                "percentage": (speaker.total_duration / total_speech * 100)
                if total_speech > 0
                else 0,
                "num_segments": len(speaker.segments),
                "confidence": speaker.confidence,
            }
            speaker_stats.append(stats)

        return {
            "num_speakers": len(diarization_result.speakers),
            "total_speech_duration": total_speech,
            "confidence": diarization_result.confidence,
            "speakers": speaker_stats,
        }

    @classmethod
    def clear_cache(cls) -> None:
        """Clear cached pipeline instances."""
        cls._pipeline_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.getLogger(__name__).info("Diarization pipeline cache cleared")

    def is_enabled(self) -> bool:
        """Check if diarization is enabled."""
        return self.config.enabled and self.pipeline is not None
