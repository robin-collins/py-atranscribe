"""
File monitoring system for detecting new audio files for transcription.
Uses watchdog for efficient file system monitoring with stability detection.
"""

import asyncio
import hashlib
import logging
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

from watchdog.events import FileCreatedEvent, FileModifiedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from ..config import AppConfig


@dataclass
class FileInfo:
    """Information about a detected file."""

    path: Path
    size: int
    modified_time: float
    checksum: str | None = None
    stable_since: float | None = None
    processed: bool = False


class FileStabilityTracker:
    """
    Tracks file stability to ensure files are completely written before processing.
    Uses size and modification time to detect when files have stopped changing.
    """

    def __init__(self, stability_delay: float = 5.0):
        """
        Initialize file stability tracker.

        Args:
            stability_delay: Seconds to wait after last change before considering file stable
        """
        self.stability_delay = stability_delay
        self.file_info: dict[Path, FileInfo] = {}
        self.logger = logging.getLogger(__name__)

    def update_file(self, file_path: Path) -> bool:
        """
        Update file information and check if it's stable.

        Args:
            file_path: Path to the file to check

        Returns:
            bool: True if file is stable and ready for processing
        """
        try:
            if not file_path.exists():
                # File was deleted, remove from tracking
                if file_path in self.file_info:
                    del self.file_info[file_path]
                return False

            stat = file_path.stat()
            current_time = time.time()
            current_size = stat.st_size
            current_mtime = stat.st_mtime

            if file_path not in self.file_info:
                # New file detected
                self.file_info[file_path] = FileInfo(
                    path=file_path,
                    size=current_size,
                    modified_time=current_mtime,
                    stable_since=None,
                    processed=False,
                )
                self.logger.debug(f"New file detected: {file_path}")
                return False

            file_info = self.file_info[file_path]

            # Check if file has changed
            if (
                current_size != file_info.size
                or abs(current_mtime - file_info.modified_time) > 0.1
            ):
                # File has changed, reset stability tracking
                file_info.size = current_size
                file_info.modified_time = current_mtime
                file_info.stable_since = None
                self.logger.debug(f"File changed: {file_path} (size: {current_size})")
                return False

            # File hasn't changed, check stability
            if file_info.stable_since is None:
                file_info.stable_since = current_time
                self.logger.debug(f"File stabilizing: {file_path}")
                return False

            # Check if file has been stable long enough
            stable_duration = current_time - file_info.stable_since
            if stable_duration >= self.stability_delay:
                if not file_info.processed:
                    self.logger.info(
                        f"File stable and ready: {file_path} (size: {current_size})"
                    )
                    return True

            return False

        except OSError as e:
            self.logger.warning(f"Error checking file {file_path}: {e}")
            return False

    def mark_processed(self, file_path: Path) -> None:
        """Mark a file as processed to avoid reprocessing."""
        if file_path in self.file_info:
            self.file_info[file_path].processed = True
            self.logger.debug(f"File marked as processed: {file_path}")

    def is_processed(self, file_path: Path) -> bool:
        """Check if a file has already been processed."""
        return self.file_info.get(file_path, FileInfo(Path(), 0, 0)).processed

    def get_pending_files(self) -> list[Path]:
        """Get list of files that are stable but not yet processed, excluding the most recent file unless it's older than 24 hours."""
        current_time = time.time()
        pending = []

        for file_path, info in self.file_info.items():
            if (
                not info.processed
                and info.stable_since is not None
                and (current_time - info.stable_since) >= self.stability_delay
            ):
                pending.append(file_path)

        # If we have files to process, exclude the most recent one unless it's older than 24 hours
        if len(pending) > 1:
            # Sort by modification time (oldest first)
            pending_with_mtime = [(f, self.file_info[f].modified_time) for f in pending]
            pending_with_mtime.sort(key=lambda x: x[1])

            # Get the most recent file (last in sorted list)
            most_recent_file = pending_with_mtime[-1][0]
            most_recent_mtime = pending_with_mtime[-1][1]

            # Check if most recent file is older than 24 hours
            twenty_four_hours_ago = current_time - (24 * 60 * 60)
            if most_recent_mtime > twenty_four_hours_ago:
                # Most recent file is newer than 24 hours, exclude it
                pending = [f for f, _ in pending_with_mtime[:-1]]
                self.logger.debug(f"Excluding most recent file from processing: {most_recent_file} (created: {time.ctime(most_recent_mtime)})")
            else:
                # Most recent file is older than 24 hours, include all files
                pending = [f for f, _ in pending_with_mtime]
        elif len(pending) == 1:
            # Only one file, check if it's older than 24 hours
            single_file = pending[0]
            single_file_mtime = self.file_info[single_file].modified_time
            twenty_four_hours_ago = current_time - (24 * 60 * 60)

            if single_file_mtime > twenty_four_hours_ago:
                # Single file is newer than 24 hours, exclude it
                pending = []
                self.logger.debug(f"Excluding single recent file from processing: {single_file} (created: {time.ctime(single_file_mtime)})")

        return pending

    def cleanup_old_entries(self, max_age_hours: float = 24.0) -> None:
        """Remove old entries from tracking to prevent memory leaks."""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        to_remove = []
        for file_path, info in self.file_info.items():
            if (
                info.processed
                and info.stable_since is not None
                and (current_time - info.stable_since) > max_age_seconds
            ):
                to_remove.append(file_path)

        for file_path in to_remove:
            del self.file_info[file_path]
            self.logger.debug(f"Cleaned up old entry: {file_path}")


class AudioFileHandler(FileSystemEventHandler):
    """File system event handler for audio files."""

    def __init__(
        self,
        supported_formats: list[str],
        stability_tracker: FileStabilityTracker,
        callback: Callable[[Path], None] | None = None,
    ):
        """
        Initialize audio file handler.

        Args:
            supported_formats: List of supported file extensions (with dots)
            stability_tracker: File stability tracker instance
            callback: Optional callback function when stable files are detected
        """
        self.supported_formats = {fmt.lower() for fmt in supported_formats}
        self.stability_tracker = stability_tracker
        self.callback = callback
        self.logger = logging.getLogger(__name__)

    def is_supported_format(self, file_path: Path) -> bool:
        """Check if file format is supported."""
        return file_path.suffix.lower() in self.supported_formats

    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory:
            self._handle_file_event(Path(event.src_path))

    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory:
            self._handle_file_event(Path(event.src_path))

    def _handle_file_event(self, file_path: Path) -> None:
        """Handle file system events for audio files."""
        if not self.is_supported_format(file_path):
            return

        self.logger.debug(f"File event: {file_path}")

        # Update stability tracking
        if self.stability_tracker.update_file(file_path):
            # File is stable and ready for processing
            if self.callback:
                try:
                    self.callback(file_path)
                except Exception as e:
                    self.logger.error(f"Error in file callback for {file_path}: {e}")


class FileMonitor:
    """
    Main file monitoring class that watches for new audio files.
    Implements efficient file system monitoring with stability detection.
    """

    def __init__(
        self, config: AppConfig, callback: Callable[[Path], None] | None = None
    ):
        """
        Initialize file monitor.

        Args:
            config: Application configuration
            callback: Callback function called when files are ready for processing
        """
        self.config = config
        self.callback = callback
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.stability_tracker = FileStabilityTracker(
            stability_delay=config.monitoring.stability_delay
        )

        self.handler = AudioFileHandler(
            supported_formats=config.monitoring.supported_formats,
            stability_tracker=self.stability_tracker,
            callback=self._on_file_ready,
        )

        self.observer = Observer()
        self._running = False
        self._scan_task: asyncio.Task | None = None

    def _on_file_ready(self, file_path: Path) -> None:
        """Internal callback when a file is ready for processing."""
        if not self.stability_tracker.is_processed(file_path):
            self.logger.info(f"File ready for processing: {file_path}")
            if self.callback:
                try:
                    self.callback(file_path)
                except Exception as e:
                    self.logger.error(
                        f"Error in processing callback for {file_path}: {e}"
                    )

    async def start(self) -> None:
        """Start file monitoring."""
        if self._running:
            self.logger.warning("File monitor is already running")
            return

        # Ensure input directory exists
        input_dir = self.config.directories.input
        if not input_dir.exists():
            self.logger.error(f"Input directory does not exist: {input_dir}")
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        # Start watchdog observer
        self.observer.schedule(self.handler, str(input_dir), recursive=False)
        self.observer.start()

        # Start periodic scanning for existing files
        self._running = True
        self._scan_task = asyncio.create_task(self._periodic_scan())

        self.logger.info(f"File monitor started, watching: {input_dir}")

    async def stop(self) -> None:
        """Stop file monitoring."""
        if not self._running:
            return

        self._running = False

        # Stop periodic scanning
        if self._scan_task:
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass

        # Stop watchdog observer
        self.observer.stop()
        self.observer.join(timeout=5.0)

        self.logger.info("File monitor stopped")

    async def _periodic_scan(self) -> None:
        """
        Periodically scan for existing files and check stability.
        This catches files that were created before monitoring started
        or events that were missed.
        """
        while self._running:
            try:
                await self._scan_existing_files()
                await self._check_pending_files()
                self.stability_tracker.cleanup_old_entries()

                await asyncio.sleep(self.config.monitoring.poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in periodic scan: {e}")
                await asyncio.sleep(5.0)  # Wait before retrying

    async def _scan_existing_files(self) -> None:
        """Scan input directory for existing files."""
        input_dir = self.config.directories.input

        try:
            for file_path in input_dir.iterdir():
                if (
                    file_path.is_file()
                    and self.handler.is_supported_format(file_path)
                    and not self.stability_tracker.is_processed(file_path)
                ):
                    # Update stability tracking for existing files
                    self.stability_tracker.update_file(file_path)

        except OSError as e:
            self.logger.error(f"Error scanning input directory {input_dir}: {e}")

    async def _check_pending_files(self) -> None:
        """Check for files that have become stable and ready for processing."""
        pending_files = self.stability_tracker.get_pending_files()

        for file_path in pending_files:
            if not self.stability_tracker.is_processed(file_path):
                self._on_file_ready(file_path)

    def mark_file_processed(self, file_path: Path) -> None:
        """Mark a file as processed to avoid reprocessing."""
        self.stability_tracker.mark_processed(file_path)

    def get_status(self) -> dict[str, any]:
        """Get monitoring status information."""
        return {
            "running": self._running,
            "input_directory": str(self.config.directories.input),
            "supported_formats": self.config.monitoring.supported_formats,
            "stability_delay": self.config.monitoring.stability_delay,
            "tracked_files": len(self.stability_tracker.file_info),
            "pending_files": len(self.stability_tracker.get_pending_files()),
        }


class ProcessingQueue:
    """
    Thread-safe queue for managing files to be processed.
    Provides deduplication and priority handling.
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize processing queue.

        Args:
            max_size: Maximum number of files in queue
        """
        self.max_size = max_size
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self.in_queue: set[Path] = set()
        self.processing: set[Path] = set()
        self.lock = asyncio.Lock()
        self.logger = logging.getLogger(__name__)

    async def put(self, file_path: Path) -> bool:
        """
        Add a file to the processing queue.

        Args:
            file_path: Path to file to be processed

        Returns:
            bool: True if file was added, False if already in queue or processing
        """
        async with self.lock:
            if file_path in self.in_queue or file_path in self.processing:
                self.logger.debug(f"File already queued or processing: {file_path}")
                return False

            try:
                await self.queue.put(file_path)
                self.in_queue.add(file_path)
                self.logger.debug(f"File added to queue: {file_path}")
                return True
            except asyncio.QueueFull:
                self.logger.warning(
                    f"Processing queue is full, dropping file: {file_path}"
                )
                return False

    async def get(self) -> Path:
        """
        Get next file from processing queue.

        Returns:
            Path: Next file to process
        """
        file_path = await self.queue.get()

        async with self.lock:
            self.in_queue.discard(file_path)
            self.processing.add(file_path)

        return file_path

    async def mark_done(self, file_path: Path) -> None:
        """
        Mark a file as completed processing.

        Args:
            file_path: Path to completed file
        """
        async with self.lock:
            self.processing.discard(file_path)

        self.queue.task_done()
        self.logger.debug(f"File processing completed: {file_path}")

    async def size(self) -> int:
        """Get current queue size."""
        return self.queue.qsize()

    async def get_status(self) -> dict[str, int]:
        """Get queue status information."""
        async with self.lock:
            return {
                "queued": len(self.in_queue),
                "processing": len(self.processing),
                "max_size": self.max_size,
            }
