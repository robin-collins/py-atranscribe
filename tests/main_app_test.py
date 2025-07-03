"""Tests for the main application entrypoint and startup checker."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

import pytest

# The main script is not a standard module, so we import it carefully
import auto_diarize_transcribe as main_app
from src.config import AppConfig


class TestStartupChecker:
    """Test the StartupChecker class."""

    def setup_method(self) -> None:
        """Set up the checker."""
        self.checker = main_app.StartupChecker()
        self.config = AppConfig()

    @patch("shutil.which", return_value=True)
    @patch("sys.version_info", (3, 10, 0))
    def test_all_checks_pass(self, mock_which: MagicMock, mock_version: Any) -> None:
        """Test a successful startup check."""
        # Mock all check functions to return True
        with (
            patch.object(self.checker, "check_python_dependencies", return_value=True),
            patch.object(self.checker, "check_torch_installation", return_value=True),
            patch.object(self.checker, "check_cuda_availability", return_value=True),
            patch.object(self.checker, "check_file_permissions", return_value=True),
            patch.object(self.checker, "check_network_connectivity", return_value=True),
        ):
            result = self.checker.run_all_checks(self.config)
            assert result is True
            assert self.checker.checks_failed == 0

    @patch("shutil.which", return_value=False)
    def test_critical_check_fails(self, mock_which: MagicMock) -> None:
        """Test when a critical check fails."""
        # Mock other checks to pass
        with (
            patch.object(self.checker, "check_python_dependencies", return_value=True),
            patch.object(self.checker, "check_torch_installation", return_value=True),
        ):
            result = self.checker.run_all_checks(self.config)
            assert result is False
            assert self.checker.checks_failed > 0

    @patch("os.getenv", return_value=None)  # No HF token
    def test_non_critical_check_warns(self, mock_getenv: MagicMock) -> None:
        """Test that a non-critical check produces a warning but passes overall."""
        with (
            patch.object(self.checker, "check_python_version", return_value=True),
            patch.object(self.checker, "check_system_commands", return_value=True),
            patch.object(self.checker, "check_python_dependencies", return_value=True),
            patch.object(self.checker, "check_torch_installation", return_value=True),
            patch.object(self.checker, "check_file_permissions", return_value=True),
        ):
            # The check_huggingface_token will fail, which is non-critical
            result = self.checker.run_all_checks(self.config)
            assert result is True  # Still returns True
            assert self.checker.checks_failed == 0
            assert self.checker.warnings > 0


@pytest.mark.asyncio
@patch("auto_diarize_transcribe.StartupChecker")
class TestTranscriptionService:
    """Test the main TranscriptionService class."""

    async def test_service_startup_and_shutdown(
        self, MockStartupChecker: MagicMock
    ) -> None:
        """Test the basic startup and shutdown flow of the service."""
        # Ensure startup checker passes
        mock_startup_instance = MockStartupChecker.return_value
        mock_startup_instance.run_all_checks.return_value = True

        # Mock the core components to avoid real processing
        with (
            patch("src.monitoring.file_monitor.FileMonitor") as MockFileMonitor,
            patch(
                "src.pipeline.batch_transcriber.BatchTranscriber"
            ) as MockBatchTranscriber,
            patch("src.monitoring.health_check.HealthChecker") as MockHealthChecker,
        ):
            service = main_app.TranscriptionService(config_path=None)

            # Mock the async methods
            MockFileMonitor.return_value.start = AsyncMock()
            MockFileMonitor.return_value.stop = AsyncMock()
            MockBatchTranscriber.return_value.initialize = AsyncMock()
            MockBatchTranscriber.return_value.cleanup = AsyncMock()
            MockHealthChecker.return_value.start_server = AsyncMock()
            MockHealthChecker.return_value.stop_server = AsyncMock()

            # Start the service in a background task
            start_task = asyncio.create_task(service.start())
            await asyncio.sleep(0.1)  # Give it a moment to start

            assert service._running is True
            MockFileMonitor.return_value.start.assert_called_once()
            MockBatchTranscriber.return_value.initialize.assert_called_once()

            # Stop the service
            stop_task = asyncio.create_task(service.stop())
            await asyncio.sleep(0.1)

            assert service._running is False
            MockFileMonitor.return_value.stop.assert_called_once()
            MockBatchTranscriber.return_value.cleanup.assert_called_once()

            # Cancel tasks to clean up
            start_task.cancel()
            stop_task.cancel()

    def test_startup_fails_if_checks_fail(self, MockStartupChecker: MagicMock) -> None:
        """Test that the service exits if startup checks fail."""
        mock_startup_instance = MockStartupChecker.return_value
        mock_startup_instance.run_all_checks.return_value = False

        with pytest.raises(SystemExit):
            main_app.TranscriptionService(config_path=None)
