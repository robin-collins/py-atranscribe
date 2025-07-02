"""
Speaker diarization using pyannote.audio for identifying and labeling speakers in audio.
Provides integration with transcription pipeline for speaker-labeled transcripts.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    import torchaudio
    from pyannote.audio import Pipeline
    from pyannote.core import Annotation, Segment
except ImportError as e:
    logging.getLogger(__name__).error(f"Failed to import pyannote.audio: {e}")
    raise ImportError(
        "pyannote.audio is required for speaker diarization. "
        "Install with: pip install pyannote.audio"
    )

from ..config import DiarizationConfig
from ..utils.error_handling import ModelError, graceful_degradation, retry_on_error


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
    """
    Speaker diarization class using pyannote.audio.
    Identifies and labels different speakers in audio files.
    """

    _pipeline_cache: dict[str, Pipeline] = {}

    def __init__(self, config: DiarizationConfig):
        """
        Initialize Diarizer with configuration.

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
                self.logger.debug(f"Using cached diarization pipeline: {cache_key}")
                return

            self.logger.info(f"Initializing diarization pipeline: {self.config.model}")

            # Load the diarization pipeline
            self.pipeline = Pipeline.from_pretrained(
                self.config.model,
                use_auth_token=self.config.hf_token if self.config.hf_token else True,
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
            self.logger.error(f"Failed to initialize diarization pipeline: {e}")
            if "unauthorized" in str(e).lower() or "authentication" in str(e).lower():
                raise ModelError(
                    f"HuggingFace authentication failed. Please check HF_TOKEN: {e}"
                )
            raise ModelError(f"Failed to initialize diarization pipeline: {e}")

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
                            "diarization"
                        )
                    ):
                        return "cuda"
                    else:
                        self.logger.warning(
                            "Insufficient GPU memory for diarization, using CPU"
                        )
                        return "cpu"
                except Exception as e:
                    self.logger.warning(f"Error checking GPU for diarization: {e}")
                    return "cpu"
            return "cpu"
        return self.config.device

    @retry_on_error()
    def diarize(self, audio_path: str) -> DiarizationResult:
        """
        Perform speaker diarization on audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            DiarizationResult: Diarization results with speaker information

        Raises:
            ModelError: If diarization fails
        """
        if not self.config.enabled:
            # Return empty result if diarization is disabled
            return DiarizationResult(
                speakers=[], segments=[], duration=0.0, num_speakers=0, confidence=0.0
            )

        if self.pipeline is None:
            raise ModelError("Diarization pipeline not initialized")

        try:
            self.logger.info(f"Starting speaker diarization: {audio_path}")

            # Get audio duration
            audio_info = torchaudio.info(audio_path)
            duration = audio_info.num_frames / audio_info.sample_rate

            # Perform diarization
            diarization_params = {
                "min_speakers": self.config.min_speakers,
                "max_speakers": self.config.max_speakers,
            }

            # Apply the pipeline
            diarization = self.pipeline(audio_path, **diarization_params)

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
                f"Diarization completed: {len(speakers)} speakers detected, "
                f"{len(segments)} segments"
            )

            return result

        except Exception as e:
            self.logger.error(f"Diarization failed for {audio_path}: {e}")
            raise ModelError(f"Diarization failed: {e}")

    def _extract_speakers(
        self, diarization: Annotation, total_duration: float
    ) -> list[Speaker]:
        """
        Extract speaker information from diarization annotation.

        Args:
            diarization: Pyannote diarization annotation
            total_duration: Total audio duration

        Returns:
            List of detected speakers
        """
        speakers = {}

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

        return speaker_list

    def _create_segments(self, diarization: Annotation) -> list[dict[str, Any]]:
        """
        Create time-based segments with speaker labels.

        Args:
            diarization: Pyannote diarization annotation

        Returns:
            List of segments with speaker information
        """
        segments = []

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

        return segments

    def _get_speaker_index(self, speaker_id: str, diarization: Annotation) -> int:
        """Get consistent speaker index for labeling."""
        unique_speakers = sorted(
            set(str(label) for _, _, label in diarization.itertracks(yield_label=True))
        )
        return unique_speakers.index(speaker_id)

    def _calculate_confidence(self, diarization: Annotation) -> float:
        """
        Calculate overall confidence score for diarization.

        Args:
            diarization: Pyannote diarization annotation

        Returns:
            Confidence score between 0 and 1
        """
        if len(diarization) == 0:
            return 0.0

        # Simple confidence based on coverage
        total_duration = sum(
            segment.duration for segment, _, _ in diarization.itertracks()
        )
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

    def assign_speakers_to_segments(
        self,
        transcription_segments: list[dict[str, Any]],
        diarization_result: DiarizationResult,
    ) -> list[dict[str, Any]]:
        """
        Assign speaker labels to transcription segments based on overlap.

        Args:
            transcription_segments: Segments from transcription
            diarization_result: Results from diarization

        Returns:
            List of segments with speaker assignments
        """
        if not self.config.enabled or not diarization_result.segments:
            # Return segments without speaker labels if diarization is disabled
            for segment in transcription_segments:
                segment["speaker"] = "SPEAKER_00"
                segment["speaker_confidence"] = 0.0
            return transcription_segments

        labeled_segments = []

        for trans_segment in transcription_segments:
            trans_start = trans_segment["start"]
            trans_end = trans_segment["end"]
            trans_mid = (trans_start + trans_end) / 2

            # Find overlapping diarization segments
            best_speaker = "SPEAKER_00"
            best_confidence = 0.0
            max_overlap = 0.0

            for dia_segment in diarization_result.segments:
                dia_start = dia_segment["start"]
                dia_end = dia_segment["end"]

                # Calculate overlap
                overlap_start = max(trans_start, dia_start)
                overlap_end = min(trans_end, dia_end)
                overlap_duration = max(0, overlap_end - overlap_start)

                # Prefer segments that contain the midpoint or have maximum overlap
                if (dia_start <= trans_mid <= dia_end) or (
                    overlap_duration > max_overlap
                ):
                    max_overlap = overlap_duration
                    best_speaker = dia_segment["speaker_label"]

                    # Calculate confidence based on overlap ratio
                    trans_duration = trans_end - trans_start
                    if trans_duration > 0:
                        best_confidence = overlap_duration / trans_duration
                    else:
                        best_confidence = 1.0 if overlap_duration > 0 else 0.0

            # Add speaker information to segment
            labeled_segment = trans_segment.copy()
            labeled_segment["speaker"] = best_speaker
            labeled_segment["speaker_confidence"] = best_confidence

            labeled_segments.append(labeled_segment)

        self.logger.debug(
            f"Assigned speakers to {len(labeled_segments)} transcription segments"
        )
        return labeled_segments

    def get_speaker_statistics(
        self, diarization_result: DiarizationResult
    ) -> dict[str, Any]:
        """
        Get statistics about detected speakers.

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
