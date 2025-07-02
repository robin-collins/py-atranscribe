"""
Main pipeline orchestrator for batch transcription and diarization.
Coordinates the entire processing workflow from audio input to multi-format output.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import AppConfig
from ..diarization.diarizer import DiarizationResult, Diarizer
from ..output.subtitle_manager import SubtitleManager
from ..transcription.whisper_factory import FasterWhisperInference, WhisperFactory
from ..utils.error_handling import (
    AudioProcessingError,
    FileSystemError,
    error_tracker,
    graceful_degradation,
    retry_on_error,
)


@dataclass
class ProcessingResult:
    """Result of processing a single audio file."""

    input_file: Path
    output_files: dict[str, Path]
    success: bool
    error_message: str | None = None
    processing_time: float = 0.0
    transcription_info: dict[str, Any] | None = None
    diarization_info: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class ProcessingProgress:
    """Progress information for processing operations."""

    stage: str
    progress: float  # 0.0 to 1.0
    message: str
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BatchTranscriber:
    """
    Main orchestrator for the transcription and diarization pipeline.
    Handles the complete workflow from audio input to multi-format output.
    """

    def __init__(self, config: AppConfig):
        """
        Initialize BatchTranscriber with configuration.

        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.whisper_factory = WhisperFactory()
        self.diarizer = Diarizer(config.diarization)
        self.subtitle_manager = SubtitleManager()

        # Processing state
        self._whisper_inference: FasterWhisperInference | None = None
        self._processing_stats = {
            "files_processed": 0,
            "files_failed": 0,
            "total_duration": 0.0,
            "total_processing_time": 0.0,
        }

    async def initialize(self) -> None:
        """Initialize the transcription pipeline components."""
        try:
            self.logger.info("Initializing transcription pipeline...")

            # Initialize Whisper model
            self._whisper_inference = self.whisper_factory.create_whisper_inference(
                self.config.transcription.whisper
            )

            # Verify diarization if enabled
            if self.config.diarization.enabled:
                if not self.diarizer.is_enabled():
                    self.logger.warning(
                        "Diarization requested but not properly initialized"
                    )

            self.logger.info("Transcription pipeline initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize transcription pipeline: {e}")
            raise

    @retry_on_error()
    async def process_file(
        self,
        file_path: Path,
        progress_callback: Callable[[ProcessingProgress], None] | None = None,
    ) -> ProcessingResult:
        """
        Process a single audio file through the complete pipeline.

        Args:
            file_path: Path to audio file to process
            progress_callback: Optional callback for progress updates

        Returns:
            ProcessingResult: Results of processing
        """
        start_time = time.time()

        def report_progress(stage: str, progress: float, message: str = ""):
            if progress_callback:
                progress_info = ProcessingProgress(
                    stage=stage,
                    progress=progress,
                    message=message,
                    timestamp=time.time(),
                )
                try:
                    progress_callback(progress_info)
                except Exception as e:
                    self.logger.warning(f"Error in progress callback: {e}")

        try:
            self.logger.info(f"Starting processing of: {file_path}")
            report_progress(
                "initialization", 0.0, f"Starting processing of {file_path.name}"
            )

            # Validate input file
            if not file_path.exists():
                raise FileSystemError(f"Input file not found: {file_path}")

            if not self._whisper_inference:
                await self.initialize()

            # Stage 1: Transcription
            report_progress("transcription", 0.1, "Starting transcription...")
            transcription_result = await self._transcribe_audio(file_path)

            if not transcription_result or not transcription_result.get("segments"):
                self.logger.warning(f"No transcription results for {file_path}")
                return ProcessingResult(
                    input_file=file_path,
                    output_files={},
                    success=False,
                    error_message="No transcription results (silent audio or processing error)",
                    processing_time=time.time() - start_time,
                )

            report_progress(
                "transcription",
                0.4,
                f"Transcription completed: {len(transcription_result['segments'])} segments",
            )

            # Stage 2: Diarization (if enabled)
            diarization_result = None
            if (
                self.config.diarization.enabled
                and not graceful_degradation.should_disable_feature("diarization")
            ):
                report_progress("diarization", 0.5, "Starting speaker diarization...")
                diarization_result = await self._perform_diarization(file_path)
                report_progress(
                    "diarization",
                    0.7,
                    f"Diarization completed: {diarization_result.num_speakers} speakers",
                )
            else:
                report_progress("diarization", 0.7, "Diarization skipped")

            # Stage 3: Merge transcription and diarization
            report_progress(
                "merging", 0.75, "Merging transcription and diarization results..."
            )
            labeled_segments = self._merge_results(
                transcription_result, diarization_result
            )

            # Stage 4: Generate outputs
            report_progress("output", 0.8, "Generating output files...")
            output_files = await self._generate_outputs(
                file_path,
                labeled_segments,
                {
                    "transcription": transcription_result,
                    "diarization": diarization_result,
                    "processing_time": time.time() - start_time,
                    "timestamp": datetime.now().isoformat(),
                },
            )

            # Stage 5: Post-processing
            report_progress("postprocessing", 0.9, "Performing post-processing...")
            await self._post_process_file(file_path)

            processing_time = time.time() - start_time

            # Update statistics
            self._processing_stats["files_processed"] += 1
            self._processing_stats["total_duration"] += transcription_result.get(
                "duration", 0
            )
            self._processing_stats["total_processing_time"] += processing_time

            report_progress(
                "completed", 1.0, f"Processing completed in {processing_time:.2f}s"
            )

            self.logger.info(
                f"Successfully processed {file_path} in {processing_time:.2f}s"
            )

            return ProcessingResult(
                input_file=file_path,
                output_files=output_files,
                success=True,
                processing_time=processing_time,
                transcription_info=transcription_result.get("transcription_info"),
                diarization_info=self._get_diarization_info(diarization_result),
                metadata={
                    "duration": transcription_result.get("duration", 0),
                    "language": transcription_result.get("language"),
                    "num_segments": len(labeled_segments),
                    "num_speakers": diarization_result.num_speakers
                    if diarization_result
                    else 1,
                },
            )

        except Exception as e:
            processing_time = time.time() - start_time
            error_message = str(e)

            self.logger.error(f"Failed to process {file_path}: {error_message}")

            # Update error statistics
            self._processing_stats["files_failed"] += 1

            report_progress("error", 1.0, f"Processing failed: {error_message}")

            return ProcessingResult(
                input_file=file_path,
                output_files={},
                success=False,
                error_message=error_message,
                processing_time=processing_time,
            )

    async def _transcribe_audio(self, file_path: Path) -> dict[str, Any]:
        """Perform audio transcription."""
        try:
            language = self.config.transcription.language
            if language == "auto":
                language = None

            # Run transcription in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._whisper_inference.transcribe, str(file_path), language
            )

            return result

        except Exception as e:
            self.logger.error(f"Transcription failed for {file_path}: {e}")
            raise AudioProcessingError(f"Transcription failed: {e}", str(file_path))

    async def _perform_diarization(
        self, file_path: Path
    ) -> DiarizationResult | None:
        """Perform speaker diarization."""
        try:
            if not self.diarizer.is_enabled():
                return None

            # Run diarization in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self.diarizer.diarize, str(file_path)
            )

            return result

        except Exception as e:
            self.logger.warning(
                f"Diarization failed for {file_path}, continuing without speaker labels: {e}"
            )
            # Don't raise exception for diarization failures, continue without speaker labels
            return None

    def _merge_results(
        self,
        transcription_result: dict[str, Any],
        diarization_result: DiarizationResult | None,
    ) -> list[dict[str, Any]]:
        """Merge transcription and diarization results."""
        segments = transcription_result.get("segments", [])

        if diarization_result and diarization_result.speakers:
            # Assign speakers to transcription segments
            segments = self.diarizer.assign_speakers_to_segments(
                segments, diarization_result
            )
        else:
            # Add default speaker labels
            for segment in segments:
                segment["speaker"] = "SPEAKER_00"
                segment["speaker_confidence"] = 0.0

        # Apply post-processing filters
        segments = self.subtitle_manager.filter_low_confidence_segments(
            segments,
            min_confidence=-1.5,  # Permissive threshold
        )

        segments = self.subtitle_manager.merge_short_segments(
            segments,
            min_duration=0.5,  # Merge very short segments
        )

        return segments

    async def _generate_outputs(
        self, input_path: Path, segments: list[dict[str, Any]], metadata: dict[str, Any]
    ) -> dict[str, Path]:
        """Generate output files in multiple formats."""
        try:
            # Determine output path
            output_base = self.config.directories.output / input_path.stem

            # Generate transcript summary for metadata
            transcript_summary = self.subtitle_manager.get_transcript_summary(segments)
            metadata.update(transcript_summary)

            # Save transcripts in requested formats
            output_files = self.subtitle_manager.save_transcripts(
                segments=segments,
                output_path=output_base,
                formats=self.config.transcription.output_formats,
                metadata=metadata,
            )

            return output_files

        except Exception as e:
            self.logger.error(f"Failed to generate outputs for {input_path}: {e}")
            raise FileSystemError(f"Failed to generate outputs: {e}", str(input_path))

    async def _post_process_file(self, file_path: Path) -> None:
        """Perform post-processing on the input file."""
        try:
            action = self.config.post_processing.action

            if action == "delete":
                file_path.unlink()
                self.logger.info(f"Deleted input file: {file_path}")

            elif action == "move":
                backup_dir = self.config.directories.backup

                # Create backup directory structure
                if self.config.post_processing.backup_structure == "date":
                    today = datetime.now().strftime("%Y-%m-%d")
                    backup_dir = backup_dir / today
                elif self.config.post_processing.backup_structure == "original":
                    # Preserve original directory structure
                    relative_path = file_path.relative_to(self.config.directories.input)
                    backup_dir = backup_dir / relative_path.parent

                backup_dir.mkdir(parents=True, exist_ok=True)
                backup_path = backup_dir / file_path.name

                # Handle filename conflicts
                counter = 1
                original_backup_path = backup_path
                while backup_path.exists():
                    stem = original_backup_path.stem
                    suffix = original_backup_path.suffix
                    backup_path = backup_dir / f"{stem}_{counter}{suffix}"
                    counter += 1

                file_path.rename(backup_path)
                self.logger.info(f"Moved input file to backup: {backup_path}")

            elif action == "keep":
                self.logger.debug(f"Keeping input file: {file_path}")
            else:
                self.logger.warning(f"Unknown post-processing action: {action}")

        except Exception as e:
            self.logger.error(f"Post-processing failed for {file_path}: {e}")
            # Don't raise exception for post-processing failures

    def _get_diarization_info(
        self, diarization_result: DiarizationResult | None
    ) -> dict[str, Any] | None:
        """Extract diarization information for metadata."""
        if not diarization_result:
            return None

        return {
            "num_speakers": diarization_result.num_speakers,
            "confidence": diarization_result.confidence,
            "total_speech_duration": sum(
                s.total_duration for s in diarization_result.speakers
            ),
            "speakers": [
                {
                    "label": s.label,
                    "duration": s.total_duration,
                    "segments": len(s.segments),
                    "confidence": s.confidence,
                }
                for s in diarization_result.speakers
            ],
        }

    def get_processing_stats(self) -> dict[str, Any]:
        """Get processing statistics."""
        stats = self._processing_stats.copy()

        if stats["files_processed"] > 0:
            stats["average_processing_time"] = (
                stats["total_processing_time"] / stats["files_processed"]
            )
            stats["average_file_duration"] = (
                stats["total_duration"] / stats["files_processed"]
            )
            stats["processing_speed_ratio"] = (
                stats["total_duration"] / stats["total_processing_time"]
            )
        else:
            stats["average_processing_time"] = 0.0
            stats["average_file_duration"] = 0.0
            stats["processing_speed_ratio"] = 0.0

        # Add error statistics
        error_stats = error_tracker.get_error_stats()
        stats["error_stats"] = error_stats

        return stats

    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self._processing_stats = {
            "files_processed": 0,
            "files_failed": 0,
            "total_duration": 0.0,
            "total_processing_time": 0.0,
        }
        self.logger.info("Processing statistics reset")

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Clear model caches to free memory
            WhisperFactory.clear_cache()
            Diarizer.clear_cache()

            self.logger.info("Batch transcriber cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            # Run cleanup synchronously
            if hasattr(self, "whisper_factory"):
                WhisperFactory.clear_cache()
            if hasattr(self, "diarizer"):
                Diarizer.clear_cache()
        except Exception:
            pass  # Ignore errors during destruction
