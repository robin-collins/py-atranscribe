#!/usr/bin/env python3
"""
Automated Audio Transcription with Speaker Diarization

Main application entry point for the containerized transcription service.
Continuously monitors input directory for audio files and processes them
with faster-whisper and pyannote.audio for transcription and diarization.

Usage:
    python auto_diarize_transcribe.py [--config CONFIG_PATH] [--log-level LEVEL]

Example:
    python auto_diarize_transcribe.py --config config.yaml --log-level INFO
"""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Optional
import structlog

from src.config import load_config, create_directories, validate_config
from src.monitoring.file_monitor import FileMonitor, ProcessingQueue
from src.pipeline.batch_transcriber import BatchTranscriber, ProcessingProgress
from src.utils.error_handling import error_tracker, graceful_degradation


class TranscriptionService:
    """
    Main service class that orchestrates file monitoring and transcription processing.
    Implements graceful shutdown and error recovery mechanisms.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize TranscriptionService.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path)

        # Setup logging
        self._setup_logging()

        self.logger = logging.getLogger(__name__)

        # Validate configuration and show warnings
        warnings = validate_config(self.config)
        for warning in warnings:
            self.logger.warning(warning)

        # Create required directories
        try:
            create_directories(self.config)
        except Exception as e:
            self.logger.error(f"Failed to create directories: {e}")
            raise

        # Initialize components
        self.file_monitor: Optional[FileMonitor] = None
        self.batch_transcriber: Optional[BatchTranscriber] = None
        self.processing_queue: Optional[ProcessingQueue] = None
        self.worker_tasks = []

        # Service state
        self._running = False
        self._shutdown_event = asyncio.Event()

        self.logger.info(
            f"TranscriptionService initialized with config from {config_path}"
        )

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.logging.level.upper())

        if self.config.logging.format == "structured":
            # Configure structured logging with JSON output
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.UnicodeDecoder(),
                    structlog.processors.JSONRenderer(),
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.stdlib.BoundLogger,
                cache_logger_on_first_use=True,
            )

        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            if self.config.logging.format == "plain"
            else None,
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        # Configure file logging if enabled
        if self.config.logging.file_enabled:
            file_handler = logging.FileHandler(self.config.logging.file_path)
            file_handler.setLevel(log_level)
            logging.getLogger().addHandler(file_handler)

    async def start(self) -> None:
        """Start the transcription service."""
        if self._running:
            self.logger.warning("Service is already running")
            return

        try:
            self.logger.info("Starting TranscriptionService...")

            # Initialize components
            await self._initialize_components()

            # Setup signal handlers for graceful shutdown
            self._setup_signal_handlers()

            # Start file monitoring
            await self.file_monitor.start()

            # Start processing workers
            await self._start_workers()

            self._running = True
            self.logger.info("TranscriptionService started successfully")

            # Log initial statistics
            self._log_service_status()

        except Exception as e:
            self.logger.error(f"Failed to start TranscriptionService: {e}")
            await self.stop()
            raise

    async def _initialize_components(self) -> None:
        """Initialize service components."""
        # Create processing queue
        self.processing_queue = ProcessingQueue(
            max_size=self.config.health_check.queue_size_max
        )

        # Create file monitor
        self.file_monitor = FileMonitor(
            config=self.config, callback=self._on_file_detected
        )

        # Create batch transcriber
        self.batch_transcriber = BatchTranscriber(self.config)
        await self.batch_transcriber.initialize()

        self.logger.info("All components initialized successfully")

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            self.logger.info(
                f"Received signal {signum}, initiating graceful shutdown..."
            )
            asyncio.create_task(self.stop())

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    async def _start_workers(self) -> None:
        """Start processing worker tasks."""
        num_workers = min(
            self.config.performance.max_concurrent_files,
            4,  # Maximum of 4 workers to prevent resource exhaustion
        )

        for i in range(num_workers):
            worker_task = asyncio.create_task(
                self._processing_worker(worker_id=i), name=f"worker-{i}"
            )
            self.worker_tasks.append(worker_task)

        self.logger.info(f"Started {num_workers} processing workers")

    async def _processing_worker(self, worker_id: int) -> None:
        """
        Processing worker that handles files from the queue.

        Args:
            worker_id: Unique identifier for this worker
        """
        worker_logger = logging.getLogger(f"{__name__}.worker-{worker_id}")
        worker_logger.info(f"Processing worker {worker_id} started")

        while self._running:
            try:
                # Get next file from queue (with timeout to allow shutdown)
                try:
                    file_path = await asyncio.wait_for(
                        self.processing_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                worker_logger.info(f"Worker {worker_id} processing: {file_path}")

                # Process the file
                try:

                    def progress_callback(progress: ProcessingProgress):
                        worker_logger.debug(
                            f"Progress: {progress.stage} - {progress.progress:.1%} - {progress.message}"
                        )

                    result = await self.batch_transcriber.process_file(
                        file_path=file_path, progress_callback=progress_callback
                    )

                    if result.success:
                        worker_logger.info(
                            f"Successfully processed {file_path} in {result.processing_time:.2f}s"
                        )

                        # Mark file as processed in monitor
                        self.file_monitor.mark_file_processed(file_path)
                    else:
                        worker_logger.error(
                            f"Failed to process {file_path}: {result.error_message}"
                        )

                except Exception as e:
                    worker_logger.error(f"Unexpected error processing {file_path}: {e}")

                finally:
                    # Mark task as done
                    await self.processing_queue.mark_done(file_path)

            except asyncio.CancelledError:
                worker_logger.info(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                worker_logger.error(f"Error in worker {worker_id}: {e}")
                await asyncio.sleep(1.0)  # Brief pause before continuing

        worker_logger.info(f"Processing worker {worker_id} stopped")

    def _on_file_detected(self, file_path: Path) -> None:
        """
        Callback when a new file is detected and ready for processing.

        Args:
            file_path: Path to the detected file
        """
        try:
            # Add file to processing queue
            asyncio.create_task(self.processing_queue.put(file_path))
            self.logger.info(f"Added file to processing queue: {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to queue file {file_path}: {e}")

    async def wait_for_completion(self) -> None:
        """Wait for service shutdown."""
        try:
            # Wait for shutdown signal
            await self._shutdown_event.wait()
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the transcription service gracefully."""
        if not self._running:
            return

        self.logger.info("Stopping TranscriptionService...")
        self._running = False

        try:
            # Stop file monitoring
            if self.file_monitor:
                await self.file_monitor.stop()

            # Cancel all worker tasks
            for task in self.worker_tasks:
                task.cancel()

            # Wait for workers to finish
            if self.worker_tasks:
                await asyncio.gather(*self.worker_tasks, return_exceptions=True)

            # Cleanup batch transcriber
            if self.batch_transcriber:
                await self.batch_transcriber.cleanup()

            # Log final statistics
            self._log_final_statistics()

            self._shutdown_event.set()
            self.logger.info("TranscriptionService stopped")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

    def _log_service_status(self) -> None:
        """Log current service status and configuration."""
        self.logger.info("=== TranscriptionService Status ===")
        self.logger.info(f"Input directory: {self.config.directories.input}")
        self.logger.info(f"Output directory: {self.config.directories.output}")
        self.logger.info(f"Backup directory: {self.config.directories.backup}")
        self.logger.info(
            f"Supported formats: {self.config.monitoring.supported_formats}"
        )
        self.logger.info(
            f"Whisper model: {self.config.transcription.whisper.model_size}"
        )
        self.logger.info(f"Diarization enabled: {self.config.diarization.enabled}")
        self.logger.info(f"Output formats: {self.config.transcription.output_formats}")
        self.logger.info(f"Post-processing: {self.config.post_processing.action}")
        self.logger.info("===================================")

    def _log_final_statistics(self) -> None:
        """Log final processing statistics."""
        if self.batch_transcriber:
            stats = self.batch_transcriber.get_processing_stats()

            self.logger.info("=== Final Processing Statistics ===")
            self.logger.info(f"Files processed: {stats['files_processed']}")
            self.logger.info(f"Files failed: {stats['files_failed']}")
            self.logger.info(f"Total duration: {stats['total_duration']:.2f}s")
            self.logger.info(
                f"Total processing time: {stats['total_processing_time']:.2f}s"
            )

            if stats["files_processed"] > 0:
                self.logger.info(
                    f"Average processing time: {stats['average_processing_time']:.2f}s"
                )
                self.logger.info(
                    f"Processing speed ratio: {stats['processing_speed_ratio']:.2f}x"
                )

            error_stats = stats.get("error_stats", {})
            if error_stats.get("total_errors", 0) > 0:
                self.logger.info(f"Total errors: {error_stats['total_errors']}")
                self.logger.info(f"Recent errors: {error_stats['recent_errors']}")

            self.logger.info("==================================")


async def main() -> None:
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="Automated Audio Transcription with Speaker Diarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (default: config.yaml or CONFIG_PATH env var)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
        help="Override log level from configuration",
    )

    parser.add_argument("--version", action="version", version="py-atranscribe 1.0.0")

    args = parser.parse_args()

    # Override log level if provided
    if args.log_level:
        import os

        os.environ["LOG_LEVEL"] = args.log_level

    try:
        # Create and start the service
        service = TranscriptionService(config_path=args.config)
        await service.start()

        # Wait for completion
        await service.wait_for_completion()

    except KeyboardInterrupt:
        print("\nReceived interrupt signal, shutting down...")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # Run the main application
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete.")
        sys.exit(0)
