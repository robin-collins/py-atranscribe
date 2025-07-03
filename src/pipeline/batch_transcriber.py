"""Main pipeline orchestrator for batch transcription and diarization.

Coordinates the entire processing workflow from audio input to multi-format output.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn

from src.config import AppConfig
from src.diarization.diarizer import DiarizationResult, Diarizer
from src.output.subtitle_manager import SubtitleManager
from src.transcription.whisper_factory import FasterWhisperInference, WhisperFactory
from src.utils.error_handling import (
    AudioProcessingError,
    FileSystemError,
    error_tracker,
    graceful_degradation,
    retry_on_error,
)
from src.utils.file_handler import FileHandler
from src.utils.output_formatter import OutputFormatter


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
        """Convert ProcessingProgress to a dictionary."""
        return asdict(self)


class BatchTranscriber:
    """Main orchestrator for the transcription and diarization pipeline.

    Handles the complete workflow from audio input to multi-format output.
    """

    def __init__(self, config: AppConfig) -> None:
        """Initialize BatchTranscriber with configuration.

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
                self.config.transcription.whisper,
            )

            # Verify diarization if enabled
            if self.config.diarization.enabled and not self.diarizer.is_enabled():
                self.logger.warning(
                    "Diarization requested but not properly initialized",
                )

            self.logger.info("Transcription pipeline initialized successfully")

        except Exception:
            self.logger.exception("Failed to initialize transcription pipeline")
            raise

    @retry_on_error()
    async def process_file(
        self,
        file_path: Path,
        progress_callback: Callable[[ProcessingProgress], None] | None = None,
    ) -> ProcessingResult:
        """Process a single audio file through the complete pipeline.

        Args:
            file_path: Path to audio file to process
            progress_callback: Optional callback for progress updates

        Returns:
            ProcessingResult: Results of processing

        """
        start_time = time.time()

        def report_progress(stage: str, progress: float, message: str = "") -> None:
            if progress_callback:
                progress_info = ProcessingProgress(
                    stage=stage,
                    progress=progress,
                    message=message,
                    timestamp=time.time(),
                )
                try:
                    progress_callback(progress_info)
                except ExceptionGroup as e:
                    self.logger.warning("Error in progress callback: %s", e)
                except Exception as e:  # noqa: BLE001
                    self.logger.warning("Error in progress callback: %s", e)

        try:
            self.logger.info("Starting processing of: %s", file_path)
            report_progress(
                "initialization", 0.0, f"Starting processing of {file_path.name}",
            )

            # Validate input file
            if not file_path.exists():
                msg = f"Input file not found: {file_path}"
                raise FileSystemError(msg)  # noqa: TRY301

            if not self._whisper_inference:
                await self.initialize()

            # Stage 1: Transcription
            report_progress("transcription", 0.1, "Starting transcription...")
            transcription_result = await self._transcribe_audio(file_path)

            if not transcription_result or not transcription_result.get("segments"):
                self.logger.warning("No transcription results for %s", file_path)
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
                "merging", 0.75, "Merging transcription and diarization results...",
            )
            labeled_segments = self._merge_results(
                transcription_result, diarization_result,
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
                    "timestamp": datetime.now().astimezone().isoformat(),
                },
            )

            # Stage 5: Post-processing
            report_progress("postprocessing", 0.9, "Performing post-processing...")
            await self._post_process_file(file_path)

            processing_time = time.time() - start_time

            # Update statistics
            self._processing_stats["files_processed"] += 1
            self._processing_stats["total_duration"] += transcription_result.get(
                "duration", 0,
            )
            self._processing_stats["total_processing_time"] += processing_time

            report_progress(
                "completed", 1.0, f"Processing completed in {processing_time:.2f}s",
            )

            self.logger.info(
                "Successfully processed %s in %.2fs", file_path, processing_time,
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

            self.logger.exception("Failed to process %s: %s", file_path, error_message)

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

            # Prepare transcription parameters
            transcribe_params = {
                "language": language,
                "initial_prompt": self.config.transcription.whisper.initial_prompt,
                "beam_size": 5,  # Good balance of quality and speed
                "best_of": 5,  # Number of candidates for beam search
                "temperature": 0.0,  # Deterministic output
                "condition_on_previous_text": True,  # Use context from previous segments
                "compression_ratio_threshold": 2.4,  # Filter out segments with low compression
                "log_prob_threshold": -1.0,  # Filter out segments with low probability
                "no_speech_threshold": 0.6,  # Filter out segments without speech
                "word_timestamps": True,  # Enable word-level timestamps
                "prepend_punctuations": '"\'([{-',
                "append_punctuations": '"\'.,:)]}',
            }

            # Run transcription in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            segments_list, transcription_info = await loop.run_in_executor(
                None, self._whisper_inference.transcribe, str(file_path), **transcribe_params
            )

            # Convert segments to the expected format
            processed_segments = []
            total_duration = 0.0

            for segment in segments_list:
                segment_dict = {
                    "id": getattr(segment, 'id', len(processed_segments)),
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "words": [],
                    "avg_logprob": getattr(segment, 'avg_logprob', 0.0),
                    "no_speech_prob": getattr(segment, 'no_speech_prob', 0.0),
                    "compression_ratio": getattr(segment, 'compression_ratio', 0.0),
                }

                # Add word-level information if available
                if hasattr(segment, "words") and segment.words:
                    for word in segment.words:
                        word_dict = {
                            "start": word.start,
                            "end": word.end,
                            "word": word.word,
                            "probability": getattr(word, 'probability', 1.0),
                        }
                        segment_dict["words"].append(word_dict)

                processed_segments.append(segment_dict)
                total_duration = max(total_duration, segment.end)

            # Prepare result dictionary in the expected format
            result = {
                "segments": processed_segments,
                "language": transcription_info.get("language"),
                "language_probability": transcription_info.get("language_probability", 1.0),
                "duration": transcription_info.get("duration", total_duration),
                "duration_after_vad": transcription_info.get("duration", total_duration),
                "transcription_info": {
                    "model_size": self.config.transcription.whisper.model_size,
                    "compute_type": self.config.transcription.whisper.compute_type,
                    "device": self.config.transcription.whisper.device,
                    "beam_size": transcribe_params["beam_size"],
                    "temperature": transcribe_params["temperature"],
                },
                "text": " ".join(segment["text"] for segment in processed_segments),
            }

            self.logger.info(
                "Transcription completed: %d segments, %.2fs duration, language: %s",
                len(processed_segments),
                total_duration,
                transcription_info.get("language", "unknown"),
            )

            return result

        except Exception as e:
            self.logger.exception("Transcription failed for %s", file_path)
            # Re-raise as AudioProcessingError to be handled by the error handling system
            raise AudioProcessingError(f"Transcription failed for {file_path}: {e}") from e

    async def _perform_diarization(self, file_path: Path) -> DiarizationResult | None:
        """Perform speaker diarization."""
        try:
            if not self.diarizer.is_enabled():
                return None

            # Run diarization in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self.diarizer.diarize, str(file_path),
            )


        except ExceptionGroup as e:
            self.logger.warning(
                "Diarization failed for %s, continuing without speaker labels: %s", file_path, e,
            )
            return None
        except Exception as e:  # noqa: BLE001
            self.logger.warning(
                "Diarization failed for %s, continuing without speaker labels: %s", file_path, e,
            )
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
                segments, diarization_result,
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

        return self.subtitle_manager.merge_short_segments(
            segments,
            min_duration=0.5,  # Merge very short segments
        )


    async def _generate_outputs(
        self, input_path: Path, segments: list[dict[str, Any]], metadata: dict[str, Any],
    ) -> dict[str, Path]:
        """Generate output files in multiple formats."""
        try:
            # Determine output path
            output_base = self.config.directories.output / input_path.stem

            # Generate transcript summary for metadata
            transcript_summary = self.subtitle_manager.get_transcript_summary(segments)
            metadata.update(transcript_summary)

            # Save transcripts in requested formats
            return self.subtitle_manager.save_transcripts(
                segments=segments,
                output_path=output_base,
                formats=self.config.transcription.output_formats,
                metadata=metadata,
            )


        except Exception as e:
            self.logger.exception("Failed to generate outputs for %s", input_path)
            msg = f"Failed to generate outputs: {e}"
            raise FileSystemError(msg, str(input_path)) from e

    async def _post_process_file(self, file_path: Path) -> None:
        """Perform post-processing on the input file."""
        try:
            action = self.config.post_processing.action

            if action == "delete":
                file_path.unlink()
                self.logger.info("Deleted input file: %s", file_path)

            elif action == "move":
                backup_dir = self.config.directories.backup

                # Create backup directory structure
                if self.config.post_processing.backup_structure == "date":
                    today = datetime.now().astimezone().strftime("%Y-%m-%d")
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
                self.logger.info("Moved input file to backup: %s", backup_path)

            elif action == "keep":
                self.logger.debug("Keeping input file: %s", file_path)
            else:
                self.logger.warning("Unknown post-processing action: %s", action)

        except Exception:
            self.logger.exception("Post-processing failed for %s", file_path)
            # Don't raise exception for post-processing failures

    def _get_diarization_info(
        self, diarization_result: DiarizationResult | None,
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
        """Clean up resources used by the transcriber."""
        try:
            self.logger.info("Cleaning up BatchTranscriber resources...")

            # Clean up whisper inference
            if self._whisper_inference:
                if hasattr(self._whisper_inference, 'cleanup'):
                    self._whisper_inference.cleanup()
                self._whisper_inference = None

            # Clean up diarizer
            if self.diarizer:
                await self.diarizer.cleanup()

            # Clean up factory cache
            if self.whisper_factory:
                self.whisper_factory.clear_cache()

            self.logger.info("BatchTranscriber cleanup completed")

        except Exception:
            self.logger.exception("Error during BatchTranscriber cleanup")

    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        try:
            # Run cleanup synchronously
            if hasattr(self, "whisper_factory"):
                WhisperFactory.clear_cache()
            if hasattr(self, "diarizer"):
                Diarizer.clear_cache()
        except Exception as e:  # noqa: BLE001
            # Log the exception instead of passing silently for S110 compliance
            logging.getLogger(__name__).warning("Destructor cleanup failed: %s", e)
