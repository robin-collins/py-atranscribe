#!/usr/bin/env python3
"""Automated Audio Transcription with Speaker Diarization.

Main application entry point for the containerized transcription service.
Continuously monitors input directory for audio files and processes them
with faster-whisper and pyannote.audio for transcription and diarization.

Usage:
    python auto_diarize_transcribe.py [--config CONFIG_PATH] [--log-level LEVEL]

Example:
-------
    python auto_diarize_transcribe.py --config config.yaml --log-level INFO

"""

import argparse
import asyncio
import contextlib
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil
import structlog
import yaml

# Optional imports for ML functionality
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    import torchaudio

    TORCHAUDIO_AVAILABLE = True
except ImportError:
    torchaudio = None
    TORCHAUDIO_AVAILABLE = False

try:
    from faster_whisper import WhisperModel

    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    WhisperModel = None
    FASTER_WHISPER_AVAILABLE = False

try:
    from pyannote.audio import Pipeline

    PYANNOTE_AVAILABLE = True
except ImportError:
    Pipeline = None
    PYANNOTE_AVAILABLE = False

from src.config import AppConfig, create_directories, load_config, validate_config
from src.monitoring.file_monitor import FileMonitor, ProcessingQueue
from src.monitoring.health_check import HealthChecker
from src.pipeline.batch_transcriber import BatchTranscriber, ProcessingProgress
from src.utils.error_handling import (
    AudioProcessingError,
    FileSystemError,
    GPUError,
    ModelError,
    TranscriptionMemoryError,
    error_tracker,
)

# System monitoring thresholds
CPU_USAGE_ALERT_THRESHOLD = 90  # Percentage
MEMORY_USAGE_ALERT_THRESHOLD = 85  # Percentage
GPU_MEMORY_ALERT_THRESHOLD = 90  # Percentage
PROCESS_MEMORY_ALERT_THRESHOLD = 50  # Percentage
ERROR_RATE_ALERT_THRESHOLD = 5  # Errors per minute

# System requirements
REQUIRED_PYTHON_MAJOR_VERSION = 3
REQUIRED_PYTHON_MINOR_VERSION = 10

# Display limits
CUDNN_LIBS_DISPLAY_LIMIT = 3  # Number of cuDNN libraries to show before truncating

# Resource requirements
MINIMUM_MEMORY_GB = 2.0  # Minimum available memory in GB


class SystemMonitor:
    """Continuous system resource monitoring and alerting."""

    def __init__(self, config: AppConfig) -> None:
        """Initialize system monitor."""
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.SystemMonitor")
        self._monitoring = False
        self._monitor_task: asyncio.Task | None = None
        self._last_alert_time = {}  # Throttle alerts
        self.alert_interval = 300  # 5 minutes between same alert types

    async def start_monitoring(self) -> None:
        """Start continuous system monitoring."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        self.logger.info("System monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop system monitoring."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitor_task
        self.logger.info("System monitoring stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                await self._check_system_resources()
                await self._check_gpu_status()
                await self._check_disk_usage()
                await self._check_memory_usage()
                await self._check_processing_health()

                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception:
                self.logger.exception("Error in system monitoring loop")
                await asyncio.sleep(60)  # Wait longer on error

    async def _check_system_resources(self) -> None:
        """Check overall system resource utilization."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()

            # Log periodic resource status
            self.logger.debug(
                "System resources - CPU: %.1f%%, Memory: %.1f%% (%.1fGB/%.1fGB)",
                cpu_percent,
                memory.percent,
                memory.used / (1024**3),
                memory.total / (1024**3),
            )

            # Alert on high resource usage
            if cpu_percent > CPU_USAGE_ALERT_THRESHOLD:
                await self._alert("high_cpu", f"High CPU usage: {cpu_percent:.1f}%")
            if memory.percent > MEMORY_USAGE_ALERT_THRESHOLD:
                await self._alert(
                    "high_memory",
                    f"High memory usage: {memory.percent:.1f}%",
                )

        except Exception:
            self.logger.exception("Error checking system resources")

    async def _check_gpu_status(self) -> None:
        """Check GPU status and memory usage."""
        try:
            if not TORCH_AVAILABLE:
                return  # PyTorch not available

            if torch.cuda.is_available():
                for device_id in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(device_id) / (1024**3)
                    reserved = torch.cuda.memory_reserved(device_id) / (1024**3)
                    total = torch.cuda.get_device_properties(device_id).total_memory / (
                        1024**3
                    )

                    usage_percent = (reserved / total) * 100 if total > 0 else 0

                    self.logger.debug(
                        "GPU %d - Allocated: %.1fGB, Reserved: %.1fGB, Total: %.1fGB (%.1f%%)",
                        device_id,
                        allocated,
                        reserved,
                        total,
                        usage_percent,
                    )

                    if usage_percent > GPU_MEMORY_ALERT_THRESHOLD:
                        await self._alert(
                            "gpu_memory",
                            f"GPU {device_id} memory high: {usage_percent:.1f}%",
                        )

        except Exception:
            self.logger.exception("Error checking GPU status")

    async def _check_disk_usage(self) -> None:
        """Check disk usage for all configured directories."""
        try:
            directories = [
                (self.config.directories.input, "input"),
                (self.config.directories.output, "output"),
                (self.config.directories.backup, "backup"),
                (self.config.directories.temp, "temp"),
            ]

            for path, name in directories:
                if path.exists():
                    usage = psutil.disk_usage(str(path))
                    free_gb = usage.free / (1024**3)
                    used_percent = (usage.used / usage.total) * 100

                    self.logger.debug(
                        "Disk usage %s - Free: %.1fGB, Used: %.1f%%",
                        name,
                        free_gb,
                        used_percent,
                    )

                    if free_gb < 1.0:  # Less than 1GB free
                        await self._alert(
                            "low_disk",
                            f"Low disk space in {name}: {free_gb:.1f}GB free",
                        )

        except Exception:
            self.logger.exception("Error checking disk usage")

    async def _check_memory_usage(self) -> None:
        """Check memory usage patterns and potential leaks."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()

            self.logger.debug(
                "Process memory - RSS: %.1fMB, VMS: %.1fMB, Percent: %.1f%%",
                memory_info.rss / (1024**2),
                memory_info.vms / (1024**2),
                memory_percent,
            )

            # Alert if process is using more than threshold% of system memory
            if memory_percent > PROCESS_MEMORY_ALERT_THRESHOLD:
                await self._alert(
                    "process_memory",
                    f"Process memory usage high: {memory_percent:.1f}%",
                )

        except Exception:
            self.logger.exception("Error checking memory usage")

    async def _check_processing_health(self) -> None:
        """Check processing pipeline health indicators."""
        try:
            error_stats = error_tracker.get_error_stats()

            # Log error statistics periodically
            if error_stats.get("total_errors", 0) > 0:
                self.logger.debug(
                    "Error statistics - Total: %d, Recent: %d, Rate: %.2f/min",
                    error_stats.get("total_errors", 0),
                    error_stats.get("recent_errors", 0),
                    error_stats.get("error_rate", 0),
                )

            # Alert on high error rates
            if error_stats.get("error_rate", 0) > ERROR_RATE_ALERT_THRESHOLD:
                await self._alert(
                    "high_error_rate",
                    f"High error rate: {error_stats['error_rate']:.1f}/min",
                )

        except Exception:
            self.logger.exception("Error checking processing health")

    async def _alert(self, alert_type: str, message: str) -> None:
        """Send alert with throttling to prevent spam."""
        current_time = time.time()
        last_alert = self._last_alert_time.get(alert_type, 0)

        if current_time - last_alert > self.alert_interval:
            self.logger.warning("ALERT [%s]: %s", alert_type.upper(), message)
            self._last_alert_time[alert_type] = current_time


class StartupChecker:
    """Comprehensive startup validation for all dependencies and requirements."""

    def __init__(self) -> None:
        """Initialize startup checker."""
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = 0
        self.startup_time = time.time()
        self.logger = logging.getLogger(f"{__name__}.StartupChecker")

    def print_header(self) -> None:
        """Print startup check header."""

    def print_footer(self) -> None:
        """Print startup check footer with summary."""
        elapsed_time = time.time() - self.startup_time

        if self.checks_failed > 0:
            # Log detailed failure information
            self.logger.critical(
                "Startup failed - %d checks failed, %d warnings, completed in %.2fs",
                self.checks_failed,
                self.warnings,
                elapsed_time,
            )
            return False
        if self.warnings > 0:
            self.logger.warning(
                "Startup completed with warnings - %d warnings, completed in %.2fs",
                self.warnings,
                elapsed_time,
            )
        else:
            self.logger.info(
                "Startup validation successful - all %d checks passed in %.2fs",
                self.checks_passed,
                elapsed_time,
            )
        return True

    def check_item(
        self, name: str, check_func: Callable[[], bool], critical: bool = True
    ) -> bool:
        """Run a single check and report results."""
        start_time = time.time()
        try:
            result = check_func()
            elapsed = time.time() - start_time

            if result:
                self.checks_passed += 1
                self.logger.debug("Check passed: %s (%.3fs)", name, elapsed)
                return True
            if critical:
                self.checks_failed += 1
                self.logger.error("Critical check failed: %s (%.3fs)", name, elapsed)
            else:
                self.warnings += 1
                self.logger.warning(
                    "Check failed with warning: %s (%.3fs)",
                    name,
                    elapsed,
                )
        except Exception as e:
            elapsed = time.time() - start_time
            if critical:
                self.checks_failed += 1
                self.logger.exception(
                    "Critical check exception: %s - %s (%.3fs)",
                    name,
                    e,
                    elapsed,
                )
                return False
            self.warnings += 1
            self.logger.warning(
                "Check exception with warning: %s - %s (%.3fs)",
                name,
                e,
                elapsed,
            )
            return False
        else:
            return False

    def check_python_version(self) -> bool:
        """Check Python version compatibility."""
        version = sys.version_info
        return bool(
            version.major == REQUIRED_PYTHON_MAJOR_VERSION
            and version.minor >= REQUIRED_PYTHON_MINOR_VERSION
        )

    def check_system_commands(self) -> bool:
        """Check required system commands."""
        commands = ["ffmpeg", "ffprobe"]
        missing = []
        for cmd in commands:
            if not shutil.which(cmd):
                missing.append(cmd)

        return not missing

    def check_python_dependencies(self) -> bool:
        """Check critical Python dependencies."""
        critical_deps = [
            ("torch", "PyTorch"),
            ("torchaudio", "TorchAudio"),
            ("faster_whisper", "Faster-Whisper"),
            ("pyannote.audio", "Pyannote Audio"),
            ("transformers", "Transformers"),
            ("watchdog", "Watchdog"),
            ("psutil", "PSUtil"),
            ("yaml", "PyYAML"),
            ("fastapi", "FastAPI"),
            ("uvicorn", "Uvicorn"),
            ("srt", "SRT"),
            ("webvtt", "WebVTT"),
            ("structlog", "StructLog"),
        ]

        missing = []
        for module, name in critical_deps:
            try:
                __import__(module)
            except ImportError:
                missing.append(name)

        return not missing

    def check_optional_dependencies(self) -> bool:
        """Check optional Python dependencies."""
        optional_deps = [
            ("librosa", "Librosa"),
            ("soundfile", "SoundFile"),
            ("pydub", "PyDub"),
            ("httpx", "HTTPX"),
            ("requests", "Requests"),
        ]

        missing = []
        for module, name in optional_deps:
            try:
                __import__(module)
            except ImportError:
                missing.append(name)

        return not missing

    def check_torch_installation(self) -> bool:
        """Check PyTorch installation and functionality."""
        try:
            if not TORCH_AVAILABLE:
                return False

            # Print PyTorch version

            # Test basic torch functionality
            x = torch.tensor([1.0, 2.0, 3.0])
            x + 1

            # Test CUDA availability in PyTorch
            cuda_available = torch.cuda.is_available()

            if cuda_available:
                torch.cuda.device_count()
                torch.cuda.current_device()
                torch.cuda.get_device_name()

                # Test basic CUDA tensor operations
                try:
                    x_cuda = torch.tensor([1.0]).cuda()
                    x_cuda + 1
                except (RuntimeError, torch.cuda.OutOfMemoryError):
                    return False

                # Check cuDNN
                cudnn_enabled = torch.backends.cudnn.enabled
                (torch.backends.cudnn.version() if cudnn_enabled else "N/A")

        except (RuntimeError, OSError):
            return False
        else:
            return True

    def check_cuda_availability(self) -> bool:
        """Check CUDA availability and configuration with comprehensive diagnostics."""
        try:
            if not TORCH_AVAILABLE:
                return False

            # Check nvidia-smi availability
            self._check_nvidia_smi()

            # Check CUDA compiler
            self._check_cuda_compiler()

            # Check cuDNN library files
            self._check_cudnn_libraries()

            if torch.cuda.is_available():
                torch.cuda.device_count()
                torch.cuda.get_device_name(0)
                torch.cuda.get_device_properties(0).total_memory / (1024**3)

                # Test cuDNN specifically
                cudnn_available = (
                    torch.backends.cudnn.enabled and torch.backends.cudnn.is_available()
                )
                (torch.backends.cudnn.version() if cudnn_available else "N/A")

                # Test basic CUDA operations
                try:
                    x = torch.randn(100, 100).cuda()
                    y = torch.randn(100, 100).cuda()
                    torch.mm(x, y)
                except (RuntimeError, torch.cuda.OutOfMemoryError):
                    return False

                return cudnn_available  # Require cuDNN for full functionality
        except (RuntimeError, OSError):
            return False
        else:
            return False

    def _check_nvidia_smi(self) -> bool:
        """Check if nvidia-smi is available and working."""
        try:
            nvidia_smi_path = shutil.which("nvidia-smi")
            if nvidia_smi_path is None:
                return False
            result = subprocess.run(  # noqa: S603
                [nvidia_smi_path],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (
            subprocess.TimeoutExpired,
            subprocess.SubprocessError,
            FileNotFoundError,
        ):
            return False
        else:
            return result.returncode == 0

    def _check_cuda_compiler(self) -> bool:
        """Check if CUDA compiler (nvcc) is available."""
        try:
            nvcc_path = shutil.which("nvcc")
            if nvcc_path is None:
                return False
            result = subprocess.run(  # noqa: S603
                [nvcc_path, "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                # Extract release line
                lines = result.stdout.strip().split("\n")
                next(
                    (line for line in lines if "release" in line.lower()),
                    "",
                )
            else:
                pass
        except (
            subprocess.TimeoutExpired,
            subprocess.SubprocessError,
            FileNotFoundError,
            OSError,
        ):
            return False
        else:
            return result.returncode == 0

    def _check_cudnn_libraries(self) -> bool:
        """Check if cuDNN library files are present in common locations."""
        cudnn_paths = [
            "/usr/lib/x86_64-linux-gnu/",
            "/usr/local/cuda/lib64/",
            "/opt/conda/lib/",
            "/usr/lib64/",
        ]

        find_path = shutil.which("find")
        if find_path is None:
            return False

        for path in cudnn_paths:
            # Validate that path is a safe directory path
            if not Path(path).is_absolute() or ".." in path:
                continue
            try:
                result = subprocess.run(  # noqa: S603
                    [find_path, path, "-name", "*cudnn*"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return True
            except (
                subprocess.TimeoutExpired,
                subprocess.SubprocessError,
                FileNotFoundError,
            ):
                continue

        return False

    def check_cuda_diagnostics(self) -> bool:
        """Run comprehensive CUDA diagnostics similar to cuda-diagnostic.py."""
        try:
            # Check nvidia-smi with detailed output
            nvidia_smi_success = self._check_nvidia_smi_detailed()

            # Check CUDA compiler with version info
            cuda_compiler_success = self._check_cuda_compiler_detailed()

            # Check cuDNN files with detailed locations
            cudnn_files_success = self._check_cudnn_files_detailed()

            # Run PyTorch CUDA diagnostics
            pytorch_cuda_success = self._check_pytorch_cuda_detailed()

            # Summary
            sum(
                [
                    nvidia_smi_success,
                    cuda_compiler_success,
                    cudnn_files_success,
                    pytorch_cuda_success,
                ],
            )

            # Return True if at least PyTorch CUDA works (most important for our use case)
        except (ImportError, RuntimeError, OSError):
            return False
        else:
            return pytorch_cuda_success

    def _check_nvidia_smi_detailed(self) -> bool:
        """Check nvidia-smi with detailed output."""
        try:
            nvidia_smi_path = shutil.which("nvidia-smi")
            if nvidia_smi_path is None:
                return False
            result = subprocess.run(  # noqa: S603
                [nvidia_smi_path],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                # Extract first line with driver info
                lines = result.stdout.strip().split("\n")
                if lines:
                    pass
            else:
                pass
        except (
            subprocess.TimeoutExpired,
            subprocess.SubprocessError,
            FileNotFoundError,
            OSError,
        ):
            return False
        else:
            return result.returncode == 0

    def _check_cuda_compiler_detailed(self) -> bool:
        """Check CUDA compiler with version details."""
        try:
            nvcc_path = shutil.which("nvcc")
            if nvcc_path is None:
                return False
            result = subprocess.run(  # noqa: S603
                [nvcc_path, "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                # Extract release line
                lines = result.stdout.strip().split("\n")
                next(
                    (line for line in lines if "release" in line.lower()),
                    "",
                )
            else:
                pass
        except (
            subprocess.TimeoutExpired,
            subprocess.SubprocessError,
            FileNotFoundError,
            OSError,
        ):
            return False
        else:
            return result.returncode == 0

    def _check_cudnn_files_detailed(self) -> bool:
        """Check cuDNN files with detailed locations."""
        cudnn_paths = [
            "/usr/lib/x86_64-linux-gnu/",
            "/usr/local/cuda/lib64/",
            "/opt/conda/lib/",
            "/usr/lib64/",
        ]

        find_path = shutil.which("find")
        if find_path is None:
            return False

        found_cudnn = False
        for path in cudnn_paths:
            # Validate that path is a safe directory path
            if not Path(path).is_absolute() or ".." in path:
                continue
            try:
                result = subprocess.run(  # noqa: S603
                    [find_path, path, "-name", "*cudnn*"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    libs = result.stdout.strip().split("\n")
                    if libs:
                        # Show first few libraries
                        for _lib in libs[:CUDNN_LIBS_DISPLAY_LIMIT]:
                            pass
                        if len(libs) > CUDNN_LIBS_DISPLAY_LIMIT:
                            pass
                        found_cudnn = True
                        break
            except (
                subprocess.TimeoutExpired,
                subprocess.SubprocessError,
                FileNotFoundError,
                OSError,
            ):
                continue

        if not found_cudnn:
            pass

        return found_cudnn

    def _check_pytorch_cuda_detailed(self) -> bool:
        """Check PyTorch CUDA functionality with detailed diagnostics."""
        try:
            if not TORCH_AVAILABLE:
                return False

            # Always show PyTorch version (valuable for debugging)

            # Show CUDA availability
            cuda_available = torch.cuda.is_available()

            if not cuda_available:
                return False

            # Show CUDA version (from final-cuda-tests.py)

            # Device info
            device_count = torch.cuda.device_count()

            if device_count > 0:
                torch.cuda.get_device_name(0)
                torch.cuda.get_device_properties(0).total_memory / (1024**3)

            # cuDNN check
            cudnn_enabled = torch.backends.cudnn.enabled
            if cudnn_enabled:
                torch.backends.cudnn.version()
            else:
                pass

            # Test tensor operations (simplified like final-cuda-tests.py)
            try:
                # Simple tensor test
                x = torch.tensor([1.0]).cuda()
                x + 1  # Basic operation

                # More comprehensive test
                x_large = torch.randn(100, 100).cuda()
                y_large = torch.randn(100, 100).cuda()
                torch.mm(x_large, y_large)

                # Test cuDNN-specific operations that might fail at runtime
                try:
                    # Test convolution operation that uses cuDNN
                    conv_input = torch.randn(1, 3, 32, 32).cuda()
                    conv_layer = torch.nn.Conv2d(3, 16, 3, padding=1).cuda()
                    conv_layer(conv_input)

                    # Test batch normalization that uses cuDNN
                    bn_input = torch.randn(1, 16, 32, 32).cuda()
                    bn_layer = torch.nn.BatchNorm2d(16).cuda()
                    bn_layer(bn_input)

                except (RuntimeError, torch.cuda.OutOfMemoryError) as cudnn_error:
                    error_msg = str(cudnn_error).lower()
                    if "libcudnn" in error_msg or "cudnn" in error_msg:
                        return False
                    return False
                else:
                    return True

            except (RuntimeError, torch.cuda.OutOfMemoryError):
                return False

        except (RuntimeError, OSError):
            return False

    def check_audio_processing(self) -> bool:
        """Check audio processing capabilities."""
        try:
            if not TORCH_AVAILABLE:
                return False
            if not TORCHAUDIO_AVAILABLE:
                return False

            # Test basic audio functionality
            sample_rate = 16000
            duration = 1.0
            torch.randn(1, int(sample_rate * duration))
        except RuntimeError:
            return False
        else:
            return True

    def check_whisper_models(self) -> bool:
        """Check Whisper model availability."""
        if not FASTER_WHISPER_AVAILABLE:
            return False

        # This doesn't actually load the model, just checks if we can import
        return True

    def check_diarization_models(self) -> bool:
        """Check diarization model availability."""
        if not PYANNOTE_AVAILABLE:
            return False

        # Check if we can import the pipeline
        return True

    def check_huggingface_token(self) -> bool:
        """Check HuggingFace token for diarization."""
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        return bool(token)

    def check_file_permissions(self, config: AppConfig) -> bool:
        """Check file system permissions for directories and critical files."""
        issues = []

        # Check directory permissions
        directories = [
            (
                config.directories.input,
                "Input",
                True,
                True,
            ),  # (path, name, need_read, need_write)
            (config.directories.output, "Output", True, True),
            (config.directories.backup, "Backup", True, True),
            (config.directories.temp, "Temp", True, True),
        ]

        for path, name, need_read, need_write in directories:
            try:
                # Create directory if it doesn't exist
                path.mkdir(parents=True, exist_ok=True)

                # Test read permissions
                if need_read and not os.access(path, os.R_OK):
                    issues.append(f"{name} directory not readable")
                    continue

                # Test write permissions
                if need_write:
                    test_file = path / ".write_test"
                    try:
                        test_file.write_text("test")
                        test_file.unlink()
                    except (OSError, PermissionError) as e:
                        issues.append(f"{name} directory write failed: {e}")

            except (OSError, PermissionError) as e:
                issues.append(f"{name} directory: {e}")

        return not issues

    def check_config_file_permissions(self, config_path: str | None) -> bool:
        """Check configuration file read permissions."""
        try:
            config_file = self._determine_config_file_path(config_path)
            if not config_file:
                return True  # Not critical if using defaults

            if config_file.exists():
                return self._validate_config_file_access(config_file)
            else:
                return False

        except (OSError, PermissionError):
            return False

    def _determine_config_file_path(self, config_path: str | None) -> Path | None:
        """Determine the configuration file path to check."""
        if config_path:
            return Path(config_path)
        elif os.getenv("CONFIG_PATH"):
            return Path(os.getenv("CONFIG_PATH"))
        else:
            # Check default locations
            default_configs = [Path("config.yaml"), Path("config.yml")]
            for default_config in default_configs:
                if default_config.exists():
                    return default_config
            return None

    def _validate_config_file_access(self, config_file: Path) -> bool:
        """Validate configuration file access and content."""
        # Check if file is readable
        if not os.access(config_file, os.R_OK):
            return False

        # Try to actually read the file
        try:
            with config_file.open(encoding="utf-8") as f:
                content = f.read()
                if len(content) == 0:
                    return False

            # Basic YAML syntax check
            with contextlib.suppress(yaml.YAMLError):
                yaml.safe_load(content)

        except (OSError, PermissionError, UnicodeDecodeError):
            return False

        # Check parent directory write permissions (for potential config updates)
        parent_dir = config_file.parent
        if not os.access(parent_dir, os.W_OK):
            pass
            # This is just a warning, not a failure

        return True

    def check_logging_file_permissions(self, config: AppConfig) -> bool:
        """Check logging file permissions if file logging is enabled."""
        try:
            if not config.logging.file_enabled:
                return True

            log_file = Path(config.logging.file_path)

            # Check if log file exists
            if log_file.exists():
                return self._validate_existing_log_file(log_file)
            else:
                return self._validate_new_log_file(log_file)

        except (OSError, PermissionError):
            return False

    def _validate_existing_log_file(self, log_file: Path) -> bool:
        """Validate existing log file permissions."""
        # Check if existing log file is writable
        if not os.access(log_file, os.W_OK):
            return False

        # Check log file rotation permissions (if log gets large)
        parent_dir = log_file.parent
        if not os.access(parent_dir, os.W_OK):
            pass
            # This is a warning, not a failure

        return True

    def _validate_new_log_file(self, log_file: Path) -> bool:
        """Validate new log file creation permissions."""
        # Check if parent directory exists and is writable
        parent_dir = log_file.parent

        # Try to create parent directory if it doesn't exist
        try:
            parent_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError):
            return False

        # Check if parent directory is writable
        if not os.access(parent_dir, os.W_OK):
            return False

        # Try to create and write to the log file
        return self._test_log_file_creation(log_file)

    def _test_log_file_creation(self, log_file: Path) -> bool:
        """Test log file creation and writing."""
        try:
            test_content = f"# Test log entry - {datetime.now(UTC).isoformat()}\n"
            log_file.write_text(test_content, encoding="utf-8")

            # Verify we can read it back
            read_content = log_file.read_text(encoding="utf-8")
            if read_content != test_content:
                return False

            # Clean up test file if it was just created for testing
            if log_file.stat().st_size == len(test_content):
                log_file.unlink()

        except (OSError, PermissionError, UnicodeDecodeError):
            return False
        else:
            return True

    def check_disk_space(self, config: AppConfig) -> bool:
        """Check available disk space."""
        try:
            directories = [
                (config.directories.input, "Input"),
                (config.directories.output, "Output"),
                (config.directories.backup, "Backup"),
            ]

            low_space = []
            min_space_gb = 1.0

            for path, name in directories:
                if path.exists():
                    usage = psutil.disk_usage(str(path))
                    free_gb = usage.free / (1024**3)
                    if free_gb < min_space_gb:
                        low_space.append(f"{name}: {free_gb:.1f}GB")

            if low_space:
                return False
        except OSError:
            return False
        else:
            return True

    def check_memory_availability(self) -> bool:
        """Check system memory availability."""
        try:
            memory = psutil.virtual_memory()
            memory.total / (1024**3)
            available_gb = memory.available / (1024**3)

            if available_gb < MINIMUM_MEMORY_GB:
                return False
        except OSError:
            return False
        else:
            return True

    def check_container_mounts(self) -> bool:
        """Check if running in container and verify volume mounts."""
        try:
            # Check if running in a container
            is_container = (
                Path("/.dockerenv").exists()
                or os.getenv("CONTAINER") == "true"
                or os.getenv("KUBERNETES_SERVICE_HOST") is not None
            )

            if not is_container:
                return True

            # Check common container volume mount points
            container_paths = [
                ("/data/in", "INPUT_DIR"),
                ("/data/out", "OUTPUT_DIR"),
                ("/data/backup", "BACKUP_DIR"),
                ("/app/config.yaml", "CONFIG_PATH"),
                ("/var/log", "Log directory"),
            ]

            mount_issues = []
            for path_str, description in container_paths:
                path = Path(path_str)
                if path.exists():
                    # Check if it's actually mounted (not just an empty directory)
                    try:
                        # Try to get mount information
                        path.stat()
                    except (OSError, PermissionError) as e:
                        mount_issues.append(f"{description} at {path}: {e}")
                else:
                    pass

            # Check environment variables that should be set in containers
            container_env_vars = [
                "INPUT_DIR",
                "OUTPUT_DIR",
                "BACKUP_DIR",
                "CONFIG_PATH",
                "LOG_LEVEL",
            ]

            missing_env = []
            for env_var in container_env_vars:
                value = os.getenv(env_var)
                if value:
                    pass
                else:
                    missing_env.append(env_var)

            if missing_env:
                pass

            # This is informational, not a failure
        except (OSError, PermissionError):
            return False
        else:
            return True

    def check_network_connectivity(self) -> bool:
        """Check network connectivity for model downloads."""
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
        except (TimeoutError, OSError, ConnectionError):
            return False
        else:
            return True

    def run_all_checks(self, config: AppConfig, config_path: str | None = None) -> bool:
        """Run all startup checks."""
        self.print_header()

        self.check_item("Python Version (3.10+)", self.check_python_version)
        self.check_item("System Commands (ffmpeg, ffprobe)", self.check_system_commands)

        self.check_item("Critical Dependencies", self.check_python_dependencies)
        self.check_item(
            "Optional Dependencies",
            self.check_optional_dependencies,
            critical=False,
        )

        self.check_item("PyTorch Installation", self.check_torch_installation)
        self.check_item("CUDA Support", self.check_cuda_availability, critical=False)
        self.check_item("CUDA Diagnostics", self.check_cuda_diagnostics, critical=False)
        self.check_item("Audio Processing", self.check_audio_processing)

        self.check_item("Whisper Models", self.check_whisper_models)
        self.check_item("Diarization Models", self.check_diarization_models)
        self.check_item(
            "HuggingFace Token",
            self.check_huggingface_token,
            critical=False,
        )

        self.check_item(
            "Configuration File Access",
            lambda: self.check_config_file_permissions(config_path),
        )
        self.check_item(
            "Logging File Permissions",
            lambda: self.check_logging_file_permissions(config),
        )
        self.check_item(
            "Working Directory Permissions",
            lambda: self.check_file_permissions(config),
        )
        self.check_item(
            "Container Volume Mounts",
            self.check_container_mounts,
            critical=False,
        )

        self.check_item(
            "Disk Space",
            lambda: self.check_disk_space(config),
            critical=False,
        )
        self.check_item(
            "Memory Availability",
            self.check_memory_availability,
            critical=False,
        )

        self.check_item(
            "Internet Connectivity",
            self.check_network_connectivity,
            critical=False,
        )

        return self.print_footer()

    # ... rest of the code remains the same ...

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""

        def signal_handler(signum: int, frame: object) -> None:  # noqa: ARG001
            self.logger.info(
                "Received signal %s, initiating graceful shutdown...",
                signum,
            )
            task = asyncio.create_task(self.stop())
            # Store reference to prevent garbage collection
            self._shutdown_task = task

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
                self._processing_worker(worker_id=i),
                name=f"worker-{i}",
            )
            self.worker_tasks.append(worker_task)

        self.logger.info("Started %s processing workers", num_workers)

    async def _processing_worker(self, worker_id: int) -> None:
        """Process worker that handles files from the queue.

        Args:
        ----
            worker_id: Unique identifier for this worker

        """
        worker_logger = logging.getLogger(f"{__name__}.worker-{worker_id}")
        worker_logger.info("Processing worker %s started", worker_id)

        worker_stats = {
            "files_processed": 0,
            "files_failed": 0,
            "total_processing_time": 0.0,
            "last_activity": None,
        }

        while self._running:
            try:
                # Get next file from queue (with timeout to allow shutdown)
                try:
                    file_path = await asyncio.wait_for(
                        self.processing_queue.get(),
                        timeout=1.0,
                    )
                except TimeoutError:
                    continue

                worker_logger.info("Worker %s processing: %s", worker_id, file_path)

                # Process the file and handle result
                await self._process_file_in_worker(
                    worker_id, file_path, worker_stats, worker_logger
                )

            except asyncio.CancelledError:
                worker_logger.info("Worker %s cancelled", worker_id)
                break
            except Exception:
                worker_logger.exception(
                    "Worker %s encountered unexpected error",
                    worker_id,
                )
                # Brief pause before retrying to prevent tight error loop
                await asyncio.sleep(1.0)

        # Log final worker statistics
        self._log_worker_final_stats(worker_id, worker_stats, worker_logger)

    async def _process_file_in_worker(
        self,
        worker_id: int,
        file_path: Path,
        worker_stats: dict,
        worker_logger: logging.Logger,
    ) -> None:
        """Process a single file in a worker."""
        # Update stats
        self._processing_stats["files_queued"] += 1
        self._processing_stats["last_activity"] = datetime.now(UTC).isoformat()
        worker_stats["last_activity"] = time.time()

        # Process the file
        processing_start = time.time()

        try:
            # Create progress callback for this file
            def progress_callback(
                progress: ProcessingProgress, current_file: Path = file_path
            ) -> None:
                """Handle progress updates for file processing."""
                worker_logger.debug(
                    "Worker %s progress for %s: %s (%.1f%%)",
                    worker_id,
                    current_file.name,
                    progress.stage,
                    progress.percentage,
                )

                # Update last activity timestamp
                self._processing_stats["last_activity"] = datetime.now(UTC).isoformat()
                worker_stats["last_activity"] = time.time()

            # Process the file with progress tracking
            result = await self.batch_transcriber.process_file(
                file_path,
                progress_callback=progress_callback,
            )

            processing_time = time.time() - processing_start

            # Handle processing result
            if result.success:
                await self._handle_successful_processing(
                    worker_id,
                    file_path,
                    processing_time,
                    result,
                    worker_stats,
                    worker_logger,
                )
            else:
                await self._handle_failed_processing(
                    worker_id,
                    file_path,
                    processing_time,
                    result,
                    worker_stats,
                    worker_logger,
                )

        except (
            FileSystemError,
            ModelError,
            TranscriptionMemoryError,
            AudioProcessingError,
            GPUError,
            OSError,
            IOError,
            asyncio.CancelledError,
            KeyboardInterrupt,
        ) as e:
            processing_time = time.time() - processing_start
            await self._handle_processing_exception(
                worker_id, file_path, processing_time, e, worker_stats, worker_logger
            )

        finally:
            # Mark task as done in queue
            await self.processing_queue.mark_done(file_path)

    async def _handle_successful_processing(
        self,
        worker_id: int,
        file_path: Path,
        processing_time: float,
        result: Any,
        worker_stats: dict,
        worker_logger: logging.Logger,
    ) -> None:
        """Handle successful file processing."""
        worker_logger.info(
            "✅ Worker %s successfully processed %s in %.2fs (transcription: %.2fs)",
            worker_id,
            file_path.name,
            processing_time,
            result.processing_time,
        )

        # Console output regardless of log level
        if result.transcription_info:
            result.transcription_info.get("language", "unknown")
        if result.metadata:
            result.metadata.get("duration", 0)
            result.metadata.get("num_speakers", 0)

        # Update statistics
        worker_stats["files_processed"] += 1
        worker_stats["total_processing_time"] += processing_time
        self._processing_stats["files_completed"] += 1
        self._processing_stats["total_processing_time"] += processing_time

        # Mark file as processed in monitor
        self.file_monitor.mark_file_processed(file_path)

        # Log processing metadata if available
        if result.metadata:
            worker_logger.info(
                "File metadata - Duration: %.1fs, Segments: %d, Speakers: %d, Language: %s",
                result.metadata.get("duration", 0),
                result.metadata.get("num_segments", 0),
                result.metadata.get("num_speakers", 0),
                result.transcription_info.get("language", "unknown")
                if result.transcription_info
                else "unknown",
            )

        # Print updated queue status to console
        await self._print_queue_count()

    async def _handle_failed_processing(
        self,
        worker_id: int,
        file_path: Path,
        processing_time: float,
        result: Any,
        worker_stats: dict,
        worker_logger: logging.Logger,
    ) -> None:
        """Handle failed file processing."""
        worker_logger.error(
            "❌ Worker %s failed to process %s after %.2fs: %s",
            worker_id,
            file_path.name,
            processing_time,
            result.error_message,
        )

        # Update failure statistics
        worker_stats["files_failed"] += 1
        self._processing_stats["files_failed"] += 1

        # Print updated queue status to console
        await self._print_queue_count()

    async def _handle_processing_exception(
        self,
        worker_id: int,
        file_path: Path,
        processing_time: float,
        exception: Exception,
        worker_stats: dict,
        worker_logger: logging.Logger,
    ) -> None:
        """Handle processing exception."""
        worker_logger.exception(
            "💥 Worker %s unexpected error processing %s after %.2fs: %s",
            worker_id,
            file_path.name,
            processing_time,
            exception,
        )

        worker_stats["files_failed"] += 1
        self._processing_stats["files_failed"] += 1

        # Print updated queue status to console
        await self._print_queue_count()

    def _log_worker_final_stats(
        self, worker_id: int, worker_stats: dict, worker_logger: logging.Logger
    ) -> None:
        """Log final worker statistics."""
        total_time = worker_stats["total_processing_time"]
        files_processed = worker_stats["files_processed"]
        avg_time = total_time / files_processed if files_processed > 0 else 0

        worker_logger.info(
            "Worker %s finished - Processed: %d files, Failed: %d files, "
            "Total time: %.2fs, Average: %.2fs per file",
            worker_id,
            files_processed,
            worker_stats["files_failed"],
            total_time,
            avg_time,
        )

    def _on_file_detected(self, file_path: Path) -> None:
        """Handle callback when a new file is detected and ready for processing.

        Args:
        ----
            file_path: Path to the detected file

        """
        try:
            # Get file information for logging
            file_size = file_path.stat().st_size if file_path.exists() else 0
            file_size_mb = file_size / (1024 * 1024)

            # Add file to processing queue
            task = asyncio.create_task(self.processing_queue.put(file_path))
            # Store reference to prevent garbage collection
            self._queue_task = task

            self.logger.info(
                "📁 New file detected and queued: %s (%.1f MB)",
                file_path.name,
                file_size_mb,
            )

            # Console output regardless of log level

            # Log queue status and print files waiting count
            _ = asyncio.create_task(self._log_queue_status_and_count())  # noqa: RUF006

        except Exception:
            self.logger.exception("❌ Failed to queue file %s", file_path)

    async def _log_queue_status_and_count(self) -> None:
        """Log queue status and print files waiting count to console."""
        try:
            await self._log_queue_status()

            if self.processing_queue:
                await self.processing_queue.get_status()
                # Console output regardless of log level
        except Exception:
            self.logger.exception("Error logging queue status and count")

    async def _print_queue_count(self) -> None:
        """Print current queue count to console regardless of log level."""
        try:
            if self.processing_queue:
                await self.processing_queue.get_status()
        except Exception:  # noqa: BLE001, S110
            # Don't log exceptions for console output helper
            pass

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

        shutdown_start = time.time()
        self.logger.info("🛑 Stopping TranscriptionService...")
        self._running = False

        try:
            # Update health status
            if self.health_checker:
                self.health_checker.update_service_status("stopping")

            # Stop system monitoring
            if self.system_monitor:
                await self.system_monitor.stop_monitoring()
                self.logger.info("System monitoring stopped")

            # Stop health check server
            if self.health_checker:
                health_stop_start = time.time()
                await self.health_checker.stop_server()
                health_stop_time = time.time() - health_stop_start
                self.logger.info(
                    "Health check server stopped in %.2fs",
                    health_stop_time,
                )

            # Stop file monitoring
            if self.file_monitor:
                monitor_stop_start = time.time()
                await self.file_monitor.stop()
                monitor_stop_time = time.time() - monitor_stop_start
                self.logger.info("File monitoring stopped in %.2fs", monitor_stop_time)

            # Cancel all worker tasks
            if self.worker_tasks:
                worker_stop_start = time.time()
                self.logger.info(
                    "Cancelling %d worker tasks...",
                    len(self.worker_tasks),
                )

                for task in self.worker_tasks:
                    task.cancel()

                # Wait for workers to finish
                await asyncio.gather(*self.worker_tasks, return_exceptions=True)
                worker_stop_time = time.time() - worker_stop_start
                self.logger.info("All workers stopped in %.2fs", worker_stop_time)

            # Cleanup batch transcriber
            if self.batch_transcriber:
                cleanup_start = time.time()
                await self.batch_transcriber.cleanup()
                cleanup_time = time.time() - cleanup_start
                self.logger.info(
                    "Batch transcriber cleanup completed in %.2fs",
                    cleanup_time,
                )

            # Log final statistics
            self._log_final_statistics()

            self._shutdown_event.set()

            total_shutdown_time = time.time() - shutdown_start
            service_uptime = time.time() - self.service_start_time

            self.logger.info(
                "✅ TranscriptionService stopped gracefully in %.2fs (uptime: %.1fs)",
                total_shutdown_time,
                service_uptime,
            )

        except Exception:
            shutdown_time = time.time() - shutdown_start
            self.logger.exception("❌ Error during shutdown after %.2fs", shutdown_time)


class TranscriptionService:
    """Main service class that orchestrates file monitoring and transcription processing.

    Implements graceful shutdown and error recovery mechanisms.
    """

    def __init__(self, config_path: str | None = None) -> None:
        """Initialize TranscriptionService.

        Args:
        ----
            config_path: Path to configuration file

        """
        self.service_start_time = time.time()
        self.initialization_start = time.time()

        # Load configuration first
        self.config = load_config(config_path)

        # Run comprehensive startup checks BEFORE any other initialization
        startup_checker = StartupChecker()
        if not startup_checker.run_all_checks(self.config, config_path):
            sys.exit(1)

        # Setup logging
        self._setup_logging()

        self.logger = logging.getLogger(__name__)

        # Log initialization start
        self.logger.info("🚀 TranscriptionService initialization started")
        self.logger.info("Configuration loaded from: %s", config_path or "default")

        # Validate configuration and show warnings
        warnings = validate_config(self.config)
        for warning in warnings:
            self.logger.warning("Configuration warning: %s", warning)

        # Create required directories
        try:
            create_directories(self.config)
            self.logger.info("Required directories created/verified")
        except Exception:
            self.logger.exception("Failed to create directories")
            raise

        # Initialize components
        self.file_monitor: FileMonitor | None = None
        self.batch_transcriber: BatchTranscriber | None = None
        self.processing_queue: ProcessingQueue | None = None
        self.system_monitor: SystemMonitor | None = None
        self.health_checker: HealthChecker | None = None
        self.worker_tasks = []

        # Service state tracking
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._processing_stats = {
            "files_queued": 0,
            "files_completed": 0,
            "files_failed": 0,
            "total_processing_time": 0.0,
            "last_activity": None,
            "startup_time": self.service_start_time,
        }

        # Initialize system monitoring
        self.system_monitor = SystemMonitor(self.config)

        # Initialize health checker
        self.health_checker = HealthChecker(self.config)

        initialization_time = time.time() - self.initialization_start
        self.logger.info(
            "✅ TranscriptionService initialized successfully in %.2fs (config: %s)",
            initialization_time,
            config_path or "default",
        )

    def _setup_logging(self) -> None:
        """Set up logging configuration."""
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

        service_startup_start = time.time()
        try:
            self.logger.info("🚀 Starting TranscriptionService components...")

            # Initialize components with timing
            component_start = time.time()
            await self._initialize_components()
            component_time = time.time() - component_start
            self.logger.info("Components initialized in %.2fs", component_time)

            # Setup signal handlers for graceful shutdown
            self._setup_signal_handlers()
            self.logger.debug("Signal handlers configured")

            # Start health check server
            if self.config.health_check.enabled:
                health_start = time.time()
                self.health_checker.update_service_status(
                    "starting",
                    file_monitor_running=False,
                    transcriber_initialized=False,
                )
                await self.health_checker.start_server()
                health_time = time.time() - health_start
                self.logger.info("Health check server started in %.2fs", health_time)

            # Start system monitoring
            await self.system_monitor.start_monitoring()
            self.logger.info("System monitoring started")

            # Start file monitoring
            monitor_start = time.time()
            await self.file_monitor.start()
            monitor_time = time.time() - monitor_start
            self.logger.info("File monitoring started in %.2fs", monitor_time)

            # Update health status
            self.health_checker.update_service_status(
                "running",
                file_monitor_running=True,
                transcriber_initialized=True,
            )

            # Start processing workers
            worker_start = time.time()
            await self._start_workers()
            worker_time = time.time() - worker_start
            self.logger.info("Processing workers started in %.2fs", worker_time)

            self._running = True

            # Mark service as ready
            self.health_checker.update_service_status("ready")

            total_startup_time = time.time() - service_startup_start
            self.logger.info(
                "✅ TranscriptionService started successfully in %.2fs",
                total_startup_time,
            )

            # Log initial statistics and status
            self._log_service_status()
            self._log_runtime_environment()

        except Exception:
            startup_time = time.time() - service_startup_start
            self.logger.exception(
                "❌ Failed to start TranscriptionService after %.2fs",
                startup_time,
            )
            await self.stop()
            raise

    async def _initialize_components(self) -> None:
        """Initialize service components."""
        # Pre-check CUDA/cuDNN setup to catch issues early
        self._pre_check_cuda_setup()

        # Create processing queue
        queue_start = time.time()
        self.processing_queue = ProcessingQueue(
            max_size=self.config.health_check.queue_size_max,
        )
        queue_time = time.time() - queue_start
        self.logger.debug("Processing queue initialized in %.3fs", queue_time)

        # Connect health checker to processing queue
        self.health_checker.set_processing_queue_ref(self.processing_queue)

        # Create file monitor
        monitor_start = time.time()
        self.file_monitor = FileMonitor(
            config=self.config,
            callback=self._on_file_detected,
        )
        monitor_time = time.time() - monitor_start
        self.logger.debug("File monitor initialized in %.3fs", monitor_time)

        # Create batch transcriber
        transcriber_start = time.time()
        self.batch_transcriber = BatchTranscriber(self.config)
        await self.batch_transcriber.initialize()
        transcriber_time = time.time() - transcriber_start
        self.logger.info("Batch transcriber initialized in %.2fs", transcriber_time)

        self.logger.info("All components initialized successfully")

    def _pre_check_cuda_setup(self) -> None:
        """Pre-check CUDA/cuDNN setup to catch issues early."""
        try:
            if not TORCH_AVAILABLE:
                self.logger.info("PyTorch not available - skipping CUDA checks")
                return

            # Skip check if using CPU explicitly
            if self.config.transcription.whisper.device == "cpu":
                self.logger.info("Using CPU device explicitly - skipping CUDA checks")
                return

            # Check basic CUDA availability
            if not torch.cuda.is_available():
                self.logger.warning("CUDA not available - will use CPU fallback")
                return

            # Check cuDNN availability
            if (
                not torch.backends.cudnn.enabled
                or not torch.backends.cudnn.is_available()
            ):
                self.logger.warning(
                    "cuDNN not available - will use CPU fallback or reduced GPU performance",
                )
                return

            self.logger.info("CUDA and cuDNN pre-check passed")

            # Test basic cuDNN operations that might fail
            try:
                # Test convolution operation that uses cuDNN
                test_input = torch.randn(1, 3, 32, 32).cuda()
                conv_layer = torch.nn.Conv2d(3, 16, 3, padding=1).cuda()
                _ = conv_layer(test_input)

                # Clean up test tensors
                del test_input, conv_layer
                torch.cuda.empty_cache()

                self.logger.info("cuDNN operations pre-check passed")

            except (
                RuntimeError,
                torch.cuda.OutOfMemoryError,
                ImportError,
            ) as cudnn_test_error:
                error_msg = str(cudnn_test_error).lower()
                if "libcudnn" in error_msg or "cudnn" in error_msg:
                    self.logger.warning(
                        "cuDNN pre-check failed - will attempt fallback during model loading: %s",
                        cudnn_test_error,
                    )
                else:
                    self.logger.warning(
                        "GPU operations pre-check failed - will use CPU fallback: %s",
                        cudnn_test_error,
                    )

        except (RuntimeError, OSError) as e:
            self.logger.warning("CUDA pre-check failed: %s", e)

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""

        def signal_handler(signum: int, frame: object) -> None:  # noqa: ARG001
            self.logger.info(
                "Received signal %s, initiating graceful shutdown...",
                signum,
            )
            task = asyncio.create_task(self.stop())
            # Store reference to prevent garbage collection
            self._shutdown_task = task

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
                self._processing_worker(worker_id=i),
                name=f"worker-{i}",
            )
            self.worker_tasks.append(worker_task)

        self.logger.info("Started %s processing workers", num_workers)

    async def _processing_worker(self, worker_id: int) -> None:
        """Process worker that handles files from the queue.

        Args:
        ----
            worker_id: Unique identifier for this worker

        """
        worker_logger = logging.getLogger(f"{__name__}.worker-{worker_id}")
        worker_logger.info("Processing worker %s started", worker_id)

        worker_stats = {
            "files_processed": 0,
            "files_failed": 0,
            "total_processing_time": 0.0,
            "last_activity": None,
        }

        while self._running:
            try:
                # Get next file from queue (with timeout to allow shutdown)
                try:
                    file_path = await asyncio.wait_for(
                        self.processing_queue.get(),
                        timeout=1.0,
                    )
                except TimeoutError:
                    continue

                worker_logger.info("Worker %s processing: %s", worker_id, file_path)

                # Process the file and handle result
                await self._process_file_in_worker(
                    worker_id, file_path, worker_stats, worker_logger
                )

            except asyncio.CancelledError:
                worker_logger.info("Worker %s cancelled", worker_id)
                break
            except Exception:
                worker_logger.exception(
                    "Worker %s encountered unexpected error",
                    worker_id,
                )
                # Brief pause before retrying to prevent tight error loop
                await asyncio.sleep(1.0)

        # Log final worker statistics
        self._log_worker_final_stats(worker_id, worker_stats, worker_logger)

    async def _process_file_in_worker(
        self,
        worker_id: int,
        file_path: Path,
        worker_stats: dict,
        worker_logger: logging.Logger,
    ) -> None:
        """Process a single file in a worker."""
        # Update stats
        self._processing_stats["files_queued"] += 1
        self._processing_stats["last_activity"] = datetime.now(UTC).isoformat()
        worker_stats["last_activity"] = time.time()

        # Process the file
        processing_start = time.time()

        try:
            # Create progress callback for this file
            def progress_callback(
                progress: ProcessingProgress, current_file: Path = file_path
            ) -> None:
                """Handle progress updates for file processing."""
                worker_logger.debug(
                    "Worker %s progress for %s: %s (%.1f%%)",
                    worker_id,
                    current_file.name,
                    progress.stage,
                    progress.percentage,
                )

                # Update last activity timestamp
                self._processing_stats["last_activity"] = datetime.now(UTC).isoformat()
                worker_stats["last_activity"] = time.time()

            # Process the file with progress tracking
            result = await self.batch_transcriber.process_file(
                file_path,
                progress_callback=progress_callback,
            )

            processing_time = time.time() - processing_start

            # Handle processing result
            if result.success:
                await self._handle_successful_processing(
                    worker_id,
                    file_path,
                    processing_time,
                    result,
                    worker_stats,
                    worker_logger,
                )
            else:
                await self._handle_failed_processing(
                    worker_id,
                    file_path,
                    processing_time,
                    result,
                    worker_stats,
                    worker_logger,
                )

        except (
            OSError,
            IOError,
            RuntimeError,
            asyncio.CancelledError,
            KeyboardInterrupt,
        ) as e:
            # Handle any exception that occurs during processing
            processing_time = time.time() - processing_start
            await self._handle_processing_exception(
                worker_id, file_path, processing_time, e, worker_stats, worker_logger
            )
        except Exception as e:  # noqa: BLE001
            # Handle any other unexpected exception that occurs during processing
            processing_time = time.time() - processing_start
            await self._handle_processing_exception(
                worker_id, file_path, processing_time, e, worker_stats, worker_logger
            )

        finally:
            # Mark task as done in queue
            await self.processing_queue.mark_done(file_path)

    async def _handle_successful_processing(
        self,
        worker_id: int,
        file_path: Path,
        processing_time: float,
        result: Any,
        worker_stats: dict,
        worker_logger: logging.Logger,
    ) -> None:
        """Handle successful file processing."""
        worker_logger.info(
            "✅ Worker %s successfully processed %s in %.2fs (transcription: %.2fs)",
            worker_id,
            file_path.name,
            processing_time,
            result.processing_time,
        )

        # Console output regardless of log level
        language = "unknown"
        duration = 0
        speakers = 0
        if result.transcription_info:
            language = result.transcription_info.get("language", "unknown")
        if result.metadata:
            duration = result.metadata.get("duration", 0)
            speakers = result.metadata.get("num_speakers", 0)

        print(
            f"✅ Completed: {file_path.name} ({processing_time:.1f}s) - "
            f"{language}, {duration:.1f}s, {speakers} speakers",
            flush=True,
        )

        # Update statistics
        worker_stats["files_processed"] += 1
        worker_stats["total_processing_time"] += processing_time
        self._processing_stats["files_completed"] += 1
        self._processing_stats["total_processing_time"] += processing_time

        # Mark file as processed in monitor
        self.file_monitor.mark_file_processed(file_path)

        # Log processing metadata if available
        if result.metadata:
            worker_logger.info(
                "File metadata - Duration: %.1fs, Segments: %d, Speakers: %d, Language: %s",
                result.metadata.get("duration", 0),
                result.metadata.get("num_segments", 0),
                result.metadata.get("num_speakers", 0),
                result.transcription_info.get("language", "unknown")
                if result.transcription_info
                else "unknown",
            )

        # Print updated queue status to console
        await self._print_queue_count()

    async def _handle_failed_processing(
        self,
        worker_id: int,
        file_path: Path,
        processing_time: float,
        result: Any,
        worker_stats: dict,
        worker_logger: logging.Logger,
    ) -> None:
        """Handle failed file processing."""
        worker_logger.error(
            "❌ Worker %s failed to process %s after %.2fs: %s",
            worker_id,
            file_path.name,
            processing_time,
            result.error_message,
        )

        # Console output regardless of log level
        print(
            f"❌ Failed: {file_path.name} ({processing_time:.1f}s) - {result.error_message}",
            flush=True,
        )

        # Update failure statistics
        worker_stats["files_failed"] += 1
        self._processing_stats["files_failed"] += 1

        # Print updated queue status to console
        await self._print_queue_count()

    async def _handle_processing_exception(
        self,
        worker_id: int,
        file_path: Path,
        processing_time: float,
        exception: Exception,
        worker_stats: dict,
        worker_logger: logging.Logger,
    ) -> None:
        """Handle processing exception."""
        worker_logger.exception(
            "💥 Worker %s unexpected error processing %s after %.2fs: %s",
            worker_id,
            file_path.name,
            processing_time,
            exception,
        )

        # Console output regardless of log level
        print(
            f"💥 Error: {file_path.name} ({processing_time:.1f}s) - {exception!s}",
            flush=True,
        )

        worker_stats["files_failed"] += 1
        self._processing_stats["files_failed"] += 1

        # Print updated queue status to console
        await self._print_queue_count()

    def _log_worker_final_stats(
        self, worker_id: int, worker_stats: dict, worker_logger: logging.Logger
    ) -> None:
        """Log final worker statistics."""
        total_time = worker_stats["total_processing_time"]
        files_processed = worker_stats["files_processed"]
        avg_time = total_time / files_processed if files_processed > 0 else 0

        worker_logger.info(
            "Worker %s finished - Processed: %d files, Failed: %d files, "
            "Total time: %.2fs, Average: %.2fs per file",
            worker_id,
            files_processed,
            worker_stats["files_failed"],
            total_time,
            avg_time,
        )

    async def _log_queue_status(self) -> None:
        """Log current processing queue status."""
        try:
            if self.processing_queue:
                status = await self.processing_queue.get_status()
                self.logger.info(
                    "📦 Queue status: %d queued, %d processing, %d max",
                    status["queued"],
                    status["processing"],
                    status["max_size"],
                )
        except Exception:
            self.logger.exception("Error getting queue status")

    async def _log_queue_status_and_count(self) -> None:
        """Log queue status and print files waiting count to console."""
        try:
            await self._log_queue_status()

            if self.processing_queue:
                status = await self.processing_queue.get_status()
                # Console output regardless of log level
                print(
                    f"📋 Files waiting: {status['queued']}, Processing: {status['processing']}",
                    flush=True,
                )
        except Exception:
            self.logger.exception("Error logging queue status and count")

    async def _print_queue_count(self) -> None:
        """Print current queue count to console regardless of log level."""
        try:
            if self.processing_queue:
                status = await self.processing_queue.get_status()
                print(
                    f"📋 Queue: {status['queued']} waiting, {status['processing']} processing",
                    flush=True,
                )
        except Exception:  # noqa: BLE001,S110
            # Silently ignore exceptions for console output helper to prevent log spam
            pass

    def _on_file_detected(self, file_path: Path) -> None:
        """Handle callback when a new file is detected and ready for processing.

        Args:
        ----
            file_path: Path to the detected file

        """
        try:
            # Get file information for logging
            file_size = file_path.stat().st_size if file_path.exists() else 0
            file_size_mb = file_size / (1024 * 1024)

            # Add file to processing queue
            task = asyncio.create_task(self.processing_queue.put(file_path))
            # Store reference to prevent garbage collection
            self._queue_task = task

            self.logger.info(
                "📁 New file detected and queued: %s (%.1f MB)",
                file_path.name,
                file_size_mb,
            )

            # Console output regardless of log level
            print(f"📁 Queued: {file_path.name} ({file_size_mb:.1f} MB)", flush=True)

            # Log queue status and print files waiting count
            _ = asyncio.create_task(self._log_queue_status_and_count())  # noqa: RUF006

        except Exception:
            self.logger.exception("❌ Failed to queue file %s", file_path)

    async def wait_for_completion(self) -> None:
        """Wait for service shutdown completion."""
        await self._shutdown_event.wait()

    async def stop(self) -> None:
        """Stop the transcription service gracefully."""
        if not self._running:
            return

        shutdown_start = time.time()
        self.logger.info("🛑 Stopping TranscriptionService...")
        self._running = False

        try:
            # Update health status
            if self.health_checker:
                self.health_checker.update_service_status("stopping")

            # Stop system monitoring
            if self.system_monitor:
                await self.system_monitor.stop_monitoring()
                self.logger.info("System monitoring stopped")

            # Stop health check server
            if self.health_checker:
                health_stop_start = time.time()
                await self.health_checker.stop_server()
                health_stop_time = time.time() - health_stop_start
                self.logger.info(
                    "Health check server stopped in %.2fs",
                    health_stop_time,
                )

            # Stop file monitoring
            if self.file_monitor:
                monitor_stop_start = time.time()
                await self.file_monitor.stop()
                monitor_stop_time = time.time() - monitor_stop_start
                self.logger.info("File monitoring stopped in %.2fs", monitor_stop_time)

            # Cancel all worker tasks
            if self.worker_tasks:
                worker_stop_start = time.time()
                self.logger.info(
                    "Cancelling %d worker tasks...",
                    len(self.worker_tasks),
                )

                for task in self.worker_tasks:
                    task.cancel()

                # Wait for workers to finish
                await asyncio.gather(*self.worker_tasks, return_exceptions=True)
                worker_stop_time = time.time() - worker_stop_start
                self.logger.info("All workers stopped in %.2fs", worker_stop_time)

            # Cleanup batch transcriber
            if self.batch_transcriber:
                cleanup_start = time.time()
                await self.batch_transcriber.cleanup()
                cleanup_time = time.time() - cleanup_start
                self.logger.info(
                    "Batch transcriber cleanup completed in %.2fs",
                    cleanup_time,
                )

            # Log final statistics
            self._log_final_statistics()

            self._shutdown_event.set()

            total_shutdown_time = time.time() - shutdown_start
            service_uptime = time.time() - self.service_start_time

            self.logger.info(
                "✅ TranscriptionService stopped gracefully in %.2fs (uptime: %.1fs)",
                total_shutdown_time,
                service_uptime,
            )

        except Exception:
            shutdown_time = time.time() - shutdown_start
            self.logger.exception("❌ Error during shutdown after %.2fs", shutdown_time)

    def _log_service_status(self) -> None:
        """Log current service status and configuration."""
        self.logger.info("=== TranscriptionService Status ===")
        self.logger.info("📂 Input directory: %s", self.config.directories.input)
        self.logger.info("📂 Output directory: %s", self.config.directories.output)
        self.logger.info("📂 Backup directory: %s", self.config.directories.backup)
        self.logger.info(
            "🎵 Supported formats: %s",
            self.config.monitoring.supported_formats,
        )
        self.logger.info(
            "🧠 Whisper model: %s",
            self.config.transcription.whisper.model_size,
        )
        self.logger.info("🎙️  Diarization enabled: %s", self.config.diarization.enabled)
        self.logger.info(
            "📄 Output formats: %s",
            self.config.transcription.output_formats,
        )
        self.logger.info("⚙️  Post-processing: %s", self.config.post_processing.action)
        self.logger.info("👥 Worker count: %d", len(self.worker_tasks))
        self.logger.info(
            "📊 Queue max size: %d",
            self.config.health_check.queue_size_max,
        )
        self.logger.info("===================================")

    def _log_runtime_environment(self) -> None:
        """Log runtime environment information."""
        try:
            self.logger.info("=== Runtime Environment ===")
            self.logger.info("🐍 Python version: %s", sys.version.split()[0])
            self.logger.info("💻 Platform: %s", sys.platform)
            if TORCH_AVAILABLE:
                self.logger.info("🔧 PyTorch version: %s", torch.__version__)
            else:
                self.logger.info("🔧 PyTorch: Not available")

            # Memory information
            memory = psutil.virtual_memory()
            self.logger.info(
                "💾 System memory: %.1f GB total, %.1f GB available",
                memory.total / (1024**3),
                memory.available / (1024**3),
            )

            # GPU information
            if TORCH_AVAILABLE and torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                self.logger.info("🚀 GPU available: %d device(s)", gpu_count)
                for i in range(gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    memory_gb = props.total_memory / (1024**3)
                    self.logger.info(
                        "   GPU %d: %s (%.1f GB)",
                        i,
                        props.name,
                        memory_gb,
                    )
            else:
                self.logger.info("🚀 GPU: Not available, using CPU")

            # Process information
            process = psutil.Process()
            self.logger.info("⚡ Process PID: %d", process.pid)
            self.logger.info("🧵 Thread count: %d", process.num_threads())

            self.logger.info("===========================")

        except Exception:
            self.logger.exception("Error logging runtime environment")

    def _log_final_statistics(self) -> None:
        """Log final processing statistics."""
        service_uptime = time.time() - self.service_start_time

        self.logger.info("=== Final Processing Statistics ===")
        self.logger.info(
            "⏱️  Service uptime: %.1f seconds (%.1f hours)",
            service_uptime,
            service_uptime / 3600,
        )

        # Service-level statistics
        self.logger.info("📊 Service Statistics:")
        self.logger.info("   Files queued: %d", self._processing_stats["files_queued"])
        self.logger.info(
            "   Files completed: %d",
            self._processing_stats["files_completed"],
        )
        self.logger.info("   Files failed: %d", self._processing_stats["files_failed"])
        self.logger.info(
            "   Total processing time: %.2fs",
            self._processing_stats["total_processing_time"],
        )

        if self._processing_stats["files_completed"] > 0:
            avg_time = (
                self._processing_stats["total_processing_time"]
                / self._processing_stats["files_completed"]
            )
            self.logger.info("   Average processing time: %.2fs per file", avg_time)

        # Batch transcriber statistics
        if self.batch_transcriber:
            stats = self.batch_transcriber.get_processing_stats()

            self.logger.info("🎵 Transcriber Statistics:")
            self.logger.info("   Files processed: %d", stats["files_processed"])
            self.logger.info("   Files failed: %d", stats["files_failed"])
            self.logger.info("   Total audio duration: %.2fs", stats["total_duration"])
            self.logger.info(
                "   Total processing time: %.2fs",
                stats["total_processing_time"],
            )

            if stats["files_processed"] > 0:
                self.logger.info(
                    "   Average processing time: %.2fs",
                    stats["average_processing_time"],
                )
                self.logger.info(
                    "   Processing speed ratio: %.2fx",
                    stats["processing_speed_ratio"],
                )

            error_stats = stats.get("error_stats", {})
            if error_stats.get("total_errors", 0) > 0:
                self.logger.info("❌ Error Statistics:")
                self.logger.info("   Total errors: %d", error_stats["total_errors"])
                self.logger.info("   Recent errors: %d", error_stats["recent_errors"])
                self.logger.info(
                    "   Error rate: %.2f errors/min",
                    error_stats.get("error_rate", 0),
                )

        # System resource summary
        try:
            memory = psutil.virtual_memory()
            process = psutil.Process()
            self.logger.info("💻 Final System Status:")
            self.logger.info("   System memory usage: %.1f%%", memory.percent)
            self.logger.info(
                "   Process memory usage: %.1f MB",
                process.memory_info().rss / (1024**2),
            )

            # GPU memory if available
            if TORCH_AVAILABLE and torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    self.logger.info(
                        "   GPU %d memory: %.1f GB allocated, %.1f GB reserved",
                        i,
                        allocated,
                        reserved,
                    )

        except Exception:
            self.logger.exception("Error gathering final system status")

        self.logger.info("====================================")


async def main() -> None:
    """Run main application entry point."""
    app_start_time = time.time()

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

    parser.add_argument(
        "--status-interval",
        type=int,
        default=300,
        help="Status logging interval in seconds (default: 300, 0 to disable)",
    )

    parser.add_argument("--version", action="version", version="py-atranscribe 1.0.0")

    args = parser.parse_args()

    # Override log level if provided
    if args.log_level:
        os.environ["LOG_LEVEL"] = args.log_level

    # Setup basic logging for main function
    logger = logging.getLogger(__name__)

    try:
        logger.info("🚀 py-atranscribe starting up...")
        logger.info("📋 Command line args: %s", vars(args))

        # Create and start the service
        service_create_start = time.time()
        service = TranscriptionService(config_path=args.config)
        service_create_time = time.time() - service_create_start
        logger.info("Service created in %.2fs", service_create_time)

        # Start periodic status logging if enabled
        status_task = None
        if args.status_interval > 0:
            status_task = asyncio.create_task(
                _periodic_status_logger(service, args.status_interval),
            )
            logger.info(
                "Periodic status logging enabled (interval: %ds)",
                args.status_interval,
            )

        # Start the service
        await service.start()

        total_startup_time = time.time() - app_start_time
        logger.info("🎉 Application fully started in %.2fs", total_startup_time)

        # Wait for completion
        await service.wait_for_completion()

    except KeyboardInterrupt:
        logger.info("🛑 Received interrupt signal, shutting down...")

        # Cancel status logging
        if "status_task" in locals() and status_task:
            status_task.cancel()

        sys.exit(0)
    except Exception:
        # Log the exception details for debugging
        logger.exception(
            "💥 Fatal error occurred after %.2fs",
            time.time() - app_start_time,
        )
        sys.exit(1)


async def _periodic_status_logger(service: TranscriptionService, interval: int) -> None:
    """Periodically log service status and statistics.

    Args:
    ----
        service: TranscriptionService instance
        interval: Logging interval in seconds

    """
    logger = logging.getLogger(f"{__name__}.StatusLogger")
    logger.info("📊 Periodic status logging started (interval: %ds)", interval)

    while service._running:
        try:
            await asyncio.sleep(interval)

            if not service._running:
                break

            # Log current status
            uptime = time.time() - service.service_start_time
            logger.info(
                "=== Periodic Status Report (uptime: %.1f hours) ===",
                uptime / 3600,
            )

            # Service statistics
            stats = service._processing_stats
            logger.info(
                "📊 Processing: Queued=%d, Completed=%d, Failed=%d",
                stats["files_queued"],
                stats["files_completed"],
                stats["files_failed"],
            )

            if stats["files_completed"] > 0:
                avg_time = stats["total_processing_time"] / stats["files_completed"]
                logger.info("📊 Average processing time: %.2fs per file", avg_time)

            # Queue status
            if service.processing_queue:
                queue_status = await service.processing_queue.get_status()
                logger.info(
                    "📦 Queue: %d queued, %d processing",
                    queue_status["queued"],
                    queue_status["processing"],
                )

            # System resources
            memory = psutil.virtual_memory()
            process = psutil.Process()
            logger.info(
                "💻 System: Memory=%.1f%%, Process=%.1fMB",
                memory.percent,
                process.memory_info().rss / (1024**2),
            )

            # GPU status if available
            if TORCH_AVAILABLE and torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    torch.cuda.memory_allocated(i) / (1024**3)
                    reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    usage_percent = (reserved / total) * 100 if total > 0 else 0
                    logger.info(
                        "🚀 GPU %d: %.1fGB/%.1fGB (%.1f%%)",
                        i,
                        reserved,
                        total,
                        usage_percent,
                    )

            # Error statistics
            error_stats = error_tracker.get_error_stats()
            if error_stats.get("total_errors", 0) > 0:
                logger.info(
                    "❌ Errors: %d total, %d recent, %.2f/min rate",
                    error_stats["total_errors"],
                    error_stats["recent_errors"],
                    error_stats.get("error_rate", 0),
                )

            logger.info("=== End Status Report ===")

        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("Error in periodic status logging")

    logger.info("📊 Periodic status logging stopped")


if __name__ == "__main__":
    # Run the main application
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
