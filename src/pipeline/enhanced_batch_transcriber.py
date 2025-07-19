"""Enhanced batch transcriber with flash attention and optimized models.

Integrates insanely-fast-whisper approach with the existing pipeline architecture.
"""

import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import AppConfig
from src.diarization.diarizer import DiarizationResult, Diarizer
from src.output.subtitle_manager import SubtitleManager
from src.transcription.enhanced_whisper import (
    EnhancedWhisperTranscriber,
    OutputConverter,
)
from src.utils.error_handling import (
    AudioProcessingError,
    FileSystemError,
    TranscriptionError,
    error_tracker,
    graceful_degradation,
    retry_on_error,
)
from src.utils.file_handler import FileHandler


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
    percentage: float  # 0.0 to 100.0
    message: str
    timestamp: float

    @property
    def progress(self) -> float:
        """Get progress as 0.0 to 1.0 for compatibility."""
        return self.percentage / 100.0


class EnhancedBatchTranscriber:
    """Enhanced batch transcriber using flash attention and optimized models."""

    def __init__(self, config: AppConfig) -> None:
        """Initialize enhanced batch transcriber.

        Args:
        ----
            config: Application configuration

        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.enhanced_whisper = EnhancedWhisperTranscriber(config.transcription.whisper)
        self.diarizer = Diarizer(config.diarization)
        self.subtitle_manager = SubtitleManager()
        self.file_handler = FileHandler()
        self.output_converter = OutputConverter()

        # Processing statistics
        self._processing_stats = {
            "files_processed": 0,
            "files_failed": 0,
            "total_duration": 0.0,
            "total_processing_time": 0.0,
        }

    async def initialize(self) -> None:
        """Initialize the enhanced transcription pipeline."""
        try:
            self.logger.info("Initializing enhanced transcription pipeline...")

            # Initialize enhanced Whisper
            await self.enhanced_whisper.initialize()

            # Verify diarization if enabled
            if self.config.diarization.enabled and not self.diarizer.is_enabled():
                self.logger.warning(
                    "Diarization requested but not properly initialized"
                )

            self.logger.info("Enhanced transcription pipeline initialized successfully")

        except Exception:
            self.logger.exception(
                "Failed to initialize enhanced transcription pipeline"
            )
            raise

    @retry_on_error()
    async def process_file(
        self,
        file_path: Path,
        progress_callback: Callable[[ProcessingProgress], None] | None = None,
    ) -> ProcessingResult:
        """Process a single audio file through the enhanced pipeline.

        Args:
        ----
            file_path: Path to audio file to process
            progress_callback: Optional callback for progress updates

        Returns:
        -------
            ProcessingResult: Results of processing

        """
        start_time = time.time()

        def report_progress(stage: str, percentage: float, message: str = "") -> None:
            if progress_callback:
                progress_info = ProcessingProgress(
                    stage=stage,
                    percentage=percentage,
                    message=message,
                    timestamp=time.time(),
                )
                try:
                    progress_callback(progress_info)
                except Exception as e:
                    self.logger.warning("Error in progress callback: %s", e)

        try:
            self.logger.info("Starting enhanced processing of: %s", file_path)
            report_progress(
                "initialization",
                0.0,
                f"Starting enhanced processing of {file_path.name}",
            )

            # Validate input file
            await self._validate_input_file(file_path)

            # Execute enhanced processing pipeline
            result_data = await self._execute_enhanced_pipeline(
                file_path, report_progress
            )

            processing_time = time.time() - start_time
            return self._create_success_result(
                file_path,
                result_data,
                processing_time,
                report_progress,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            error_message = str(e)

            self.logger.exception("Failed to process %s: %s", file_path, error_message)
            self._processing_stats["files_failed"] += 1

            report_progress("error", 100.0, f"Processing failed: {error_message}")

            return ProcessingResult(
                input_file=file_path,
                output_files={},
                success=False,
                error_message=error_message,
                processing_time=processing_time,
            )

    async def _validate_input_file(self, file_path: Path) -> None:
        """Validate input file exists."""
        if not file_path.exists():
            msg = f"Input file not found: {file_path}"
            raise FileSystemError(msg)

    async def _execute_enhanced_pipeline(
        self, file_path: Path, report_progress: Callable[[str, float, str], None]
    ) -> dict[str, Any]:
        """Execute the enhanced processing pipeline."""
        # Stage 1: Enhanced transcription with flash attention
        report_progress("transcription", 10.0, "Starting enhanced transcription...")

        def transcription_progress(percentage: float, message: str) -> None:
            # Map internal transcription progress to overall progress (10-60%)
            overall_percentage = 10.0 + (percentage * 50.0)
            report_progress("transcription", overall_percentage, message)

        transcription_result = await self.enhanced_whisper.transcribe(
            file_path, progress_callback=transcription_progress
        )

        if not transcription_result.chunks:
            raise TranscriptionError(
                "No transcription results (silent audio or processing error)"
            )

        report_progress(
            "transcription",
            60.0,
            f"Enhanced transcription completed: {len(transcription_result.chunks)} segments",
        )

        # Stage 2: Diarization (if enabled)
        diarization_result = await self._handle_diarization(file_path, report_progress)

        # Stage 3: Merge results and generate outputs
        report_progress("output", 80.0, "Generating enhanced output files...")

        # Convert to JSON format compatible with output converter
        json_data = self.output_converter.to_json(
            transcription_result,
            speakers=diarization_result.speakers if diarization_result else [],
        )

        # Generate output files
        output_files = await self._generate_enhanced_outputs(
            file_path,
            transcription_result,
            json_data,
            diarization_result,
        )

        # Stage 4: Post-processing
        report_progress("postprocessing", 95.0, "Performing post-processing...")
        await self._post_process_file(file_path)

        return {
            "transcription_result": transcription_result,
            "diarization_result": diarization_result,
            "output_files": output_files,
            "json_data": json_data,
        }

    async def _handle_diarization(
        self, file_path: Path, report_progress: Callable[[str, float, str], None]
    ) -> DiarizationResult | None:
        """Handle diarization processing if enabled."""
        if (
            self.config.diarization.enabled
            and not graceful_degradation.should_disable_feature("diarization")
        ):
            report_progress("diarization", 65.0, "Starting speaker diarization...")

            try:
                loop = asyncio.get_event_loop()
                diarization_result = await loop.run_in_executor(
                    None,
                    self.diarizer.diarize,
                    str(file_path),
                )
                if diarization_result:
                    report_progress(
                        "diarization",
                        75.0,
                        f"Diarization completed: {diarization_result.num_speakers} speakers",
                    )
                    return diarization_result
                else:
                    report_progress(
                        "diarization",
                        75.0,
                        "Diarization failed, continuing without speaker labels",
                    )
                    return None
            except Exception as e:
                self.logger.warning("Diarization failed: %s", e)
                report_progress(
                    "diarization", 75.0, "Diarization failed, continuing..."
                )
                return None
        else:
            report_progress("diarization", 75.0, "Diarization skipped")
            return None

    async def _generate_enhanced_outputs(
        self,
        file_path: Path,
        transcription_result: Any,
        json_data: dict[str, Any],
        diarization_result: DiarizationResult | None,
    ) -> dict[str, Path]:
        """Generate enhanced output files in multiple formats."""
        base_name = file_path.stem

        try:
            # Convert chunks to segments for subtitle manager compatibility
            segments = self._convert_chunks_to_segments(transcription_result.chunks)

            # Prepare metadata
            metadata = {
                "duration": sum(
                    chunk_timestamp[1] - chunk_timestamp[0]
                    for chunk in transcription_result.chunks
                    if (chunk_timestamp := chunk.get("timestamp")) is not None
                    and isinstance(chunk_timestamp, (list, tuple))
                    and len(chunk_timestamp) >= 2
                    and all(isinstance(t, (int, float)) for t in chunk_timestamp[:2])
                ),
                "language": transcription_result.language,
                "model": transcription_result.model_info.get("model_name")
                if transcription_result.model_info
                else "enhanced",
                "num_speakers": diarization_result.num_speakers
                if diarization_result
                else 0,
                "enhanced": True,
                "json_data": json_data,  # Include the enhanced JSON data for JSON format
            }

            # Use subtitle manager to save all formats with proper subdirectory organization
            output_files = self.subtitle_manager.save_transcripts(
                segments=segments,
                output_path=self.config.directories.output / base_name,
                formats=self.config.transcription.output_formats,
                metadata=metadata,
            )

            self.logger.info(
                "Generated %d output files for %s", len(output_files), file_path.name
            )
            return output_files

        except Exception as e:
            self.logger.exception("Failed to generate enhanced outputs: %s", e)
            msg = f"Failed to generate outputs: {e}"
            raise AudioProcessingError(msg) from e

    def _convert_chunks_to_segments(
        self, chunks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert enhanced whisper chunks to segment format for compatibility."""
        segments = []
        for i, chunk in enumerate(chunks):
            timestamp = chunk.get("timestamp")

            # Handle None or invalid timestamp data gracefully
            if (
                timestamp is None
                or not isinstance(timestamp, (list, tuple))
                or len(timestamp) < 2
                or not all(isinstance(t, (int, float)) for t in timestamp[:2])
            ):
                self.logger.warning(
                    "Skipping chunk %d with invalid timestamp: %s, text='%s'",
                    i,
                    timestamp,
                    str(chunk.get("text", ""))[:50] + "..."
                    if len(str(chunk.get("text", ""))) > 50
                    else str(chunk.get("text", "")),
                )
                continue

            segments.append(
                {
                    "id": i,
                    "start": timestamp[0],
                    "end": timestamp[1],
                    "text": chunk.get("text", ""),
                    "speaker": None,  # Will be filled by diarization if available
                }
            )
        return segments

    async def _post_process_file(self, file_path: Path) -> None:
        """Perform post-processing actions on the input file."""
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

                # Use FileHandler's safe_move_file method which handles cross-device moves
                final_backup_path = self.file_handler.safe_move_file(
                    file_path,
                    backup_path,
                )
                self.logger.info("Moved input file to backup: %s", final_backup_path)

            elif action == "keep":
                self.logger.debug("Keeping input file: %s", file_path)
            else:
                self.logger.warning("Unknown post-processing action: %s", action)

        except Exception as e:
            self.logger.warning("Post-processing failed for %s: %s", file_path, e)
            # Don't raise exception for post-processing failures

    def _create_success_result(
        self,
        file_path: Path,
        result_data: dict[str, Any],
        processing_time: float,
        report_progress: Callable[[str, float, str], None],
    ) -> ProcessingResult:
        """Create a successful processing result."""
        transcription_result = result_data["transcription_result"]
        diarization_result = result_data["diarization_result"]
        output_files = result_data["output_files"]

        # Update statistics
        self._processing_stats["files_processed"] += 1
        audio_duration = sum(
            chunk_timestamp[1] - chunk_timestamp[0]
            for chunk in transcription_result.chunks
            if (chunk_timestamp := chunk.get("timestamp")) is not None
            and isinstance(chunk_timestamp, (list, tuple))
            and len(chunk_timestamp) >= 2
            and all(isinstance(t, (int, float)) for t in chunk_timestamp[:2])
        )
        self._processing_stats["total_duration"] += audio_duration
        self._processing_stats["total_processing_time"] += processing_time

        # Create metadata
        metadata = {
            "duration": audio_duration,
            "num_segments": len(transcription_result.chunks),
            "num_speakers": diarization_result.num_speakers
            if diarization_result
            else 0,
            "processing_speed_ratio": audio_duration / processing_time
            if processing_time > 0
            else 0,
            "model_info": transcription_result.model_info,
        }

        # Create transcription info
        transcription_info = {
            "language": transcription_result.language,
            "model": transcription_result.model_info.get("model_name")
            if transcription_result.model_info
            else "enhanced",
            "flash_attention": transcription_result.model_info.get("flash_attention")
            if transcription_result.model_info
            else False,
            "processing_time": transcription_result.processing_time,
        }

        # Create diarization info
        diarization_info = None
        if diarization_result:
            diarization_info = {
                "num_speakers": diarization_result.num_speakers,
                "speakers": diarization_result.speakers,
            }

        report_progress("complete", 100.0, "Enhanced processing completed successfully")

        return ProcessingResult(
            input_file=file_path,
            output_files=output_files,
            success=True,
            processing_time=processing_time,
            transcription_info=transcription_info,
            diarization_info=diarization_info,
            metadata=metadata,
        )

    def get_processing_stats(self) -> dict[str, Any]:
        """Get current processing statistics."""
        stats = self._processing_stats.copy()

        # Calculate derived statistics
        if stats["files_processed"] > 0:
            stats["average_processing_time"] = (
                stats["total_processing_time"] / stats["files_processed"]
            )
            stats["processing_speed_ratio"] = (
                stats["total_duration"] / stats["total_processing_time"]
                if stats["total_processing_time"] > 0
                else 0
            )
        else:
            stats["average_processing_time"] = 0
            stats["processing_speed_ratio"] = 0

        # Add error statistics
        stats["error_stats"] = error_tracker.get_error_stats()

        return stats

    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            await self.enhanced_whisper.cleanup()
            self.logger.info("Enhanced batch transcriber cleaned up")
        except Exception:
            self.logger.exception("Error during enhanced batch transcriber cleanup")
