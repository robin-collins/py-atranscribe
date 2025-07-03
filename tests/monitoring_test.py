"""Unit tests for file monitoring components."""

import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.config import AppConfig, DirectoriesConfig, FileMonitoringConfig
from src.monitoring.file_monitor import (
    AudioFileHandler,
    FileMonitor,
    FileStabilityTracker,
    ProcessingQueue,
)


class TestFileStabilityTracker:
    """Test the FileStabilityTracker class."""

    def setup_method(self) -> None:
        """Set up the tracker."""
        self.tracker = FileStabilityTracker(stability_delay=0.1)
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.test_dir.name) / "test.txt"

    def teardown_method(self) -> None:
        """Clean up the temporary directory."""
        self.test_dir.cleanup()

    def test_new_file_not_stable(self) -> None:
        """Test that a newly detected file is not immediately stable."""
        self.test_path.write_text("initial")
        assert self.tracker.update_file(self.test_path) is False
        assert self.test_path in self.tracker.file_info

    def test_file_becomes_stable(self) -> None:
        """Test that a file becomes stable after the delay."""
        self.test_path.write_text("initial")
        self.tracker.update_file(self.test_path)
        time.sleep(0.2)
        assert self.tracker.update_file(self.test_path) is True

    def test_modified_file_resets_stability(self) -> None:
        """Test that modifying a file resets its stability timer."""
        self.test_path.write_text("initial")
        self.tracker.update_file(self.test_path)
        time.sleep(0.2)
        assert self.tracker.update_file(self.test_path) is True

        # Modify the file
        self.test_path.write_text("modified")
        assert self.tracker.update_file(self.test_path) is False

        # It should become stable again after the delay
        time.sleep(0.2)
        assert self.tracker.update_file(self.test_path) is True

    def test_mark_and_is_processed(self) -> None:
        """Test marking a file as processed."""
        self.test_path.write_text("data")
        self.tracker.update_file(self.test_path)
        time.sleep(0.2)
        assert self.tracker.update_file(self.test_path) is True

        self.tracker.mark_processed(self.test_path)
        assert self.tracker.is_processed(self.test_path) is True

        # A processed file should not be reported as stable again
        assert self.tracker.update_file(self.test_path) is False


@pytest.mark.asyncio
class TestProcessingQueue:
    """Test the ProcessingQueue class."""

    async def test_put_and_get(self) -> None:
        """Test adding and retrieving items from the queue."""
        queue = ProcessingQueue()
        test_path = Path("/test.wav")

        assert await queue.put(test_path) is True
        assert await queue.size() == 1

        retrieved_path = await queue.get()
        assert retrieved_path == test_path
        assert await queue.size() == 0

    async def test_duplicate_put(self) -> None:
        """Test that duplicate items are not added."""
        queue = ProcessingQueue()
        test_path = Path("/test.wav")

        await queue.put(test_path)
        assert await queue.put(test_path) is False  # Already in queue
        assert await queue.size() == 1

    async def test_mark_done(self) -> None:
        """Test marking an item as done."""
        queue = ProcessingQueue()
        test_path = Path("/test.wav")

        await queue.put(test_path)
        item = await queue.get()
        assert item in queue.processing

        await queue.mark_done(item)
        assert item not in queue.processing


@pytest.mark.asyncio
class TestFileMonitor:
    """Test the FileMonitor class."""

    def setup_method(self) -> None:
        """Set up the file monitor and a temporary directory."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.config = AppConfig(
            directories=DirectoriesConfig(input=Path(self.test_dir.name)),
            monitoring=FileMonitoringConfig(stability_delay=0.1, poll_interval=0.1),
        )
        self.callback = Mock()
        self.monitor = FileMonitor(self.config, self.callback)

    def teardown_method(self) -> None:
        """Clean up the temporary directory."""
        self.test_dir.cleanup()

    async def test_file_detection_and_callback(self) -> None:
        """Test that the monitor detects a new, stable file and calls the callback."""
        await self.monitor.start()

        try:
            test_file = Path(self.test_dir.name) / "test.mp3"
            test_file.write_text("audio data")

            # Give the monitor time to detect the file and for it to become stable
            await asyncio.sleep(0.3)

            self.callback.assert_called_once_with(test_file)
        finally:
            await self.monitor.stop()

    async def test_unsupported_file_is_ignored(self) -> None:
        """Test that unsupported file types are ignored."""
        await self.monitor.start()

        try:
            test_file = Path(self.test_dir.name) / "test.txt"
            test_file.write_text("text data")

            await asyncio.sleep(0.3)

            self.callback.assert_not_called()
        finally:
            await self.monitor.stop()

    async def test_mark_file_processed(self) -> None:
        """Test that a processed file is not reported again."""
        await self.monitor.start()

        try:
            test_file = Path(self.test_dir.name) / "test.mp3"
            test_file.write_text("audio data")
            await asyncio.sleep(0.3)
            self.callback.assert_called_once_with(test_file)

            # Mark the file as processed and reset the mock
            self.monitor.mark_file_processed(test_file)
            self.callback.reset_mock()

            # Modify the file again, it should not trigger the callback
            test_file.write_text("more audio data")
            await asyncio.sleep(0.3)
            self.callback.assert_not_called()

        finally:
            await self.monitor.stop()
