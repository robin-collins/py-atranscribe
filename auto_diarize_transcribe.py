#!/usr/bin/env python3
"""Automated Audio Transcription with Speaker Diarization.

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
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Optional

import psutil
import structlog

from src.config import create_directories, load_config, validate_config
from src.monitoring.file_monitor import FileMonitor, ProcessingQueue
from src.monitoring.health_check import HealthChecker
from src.pipeline.batch_transcriber import BatchTranscriber, ProcessingProgress
from src.utils.error_handling import error_tracker, graceful_degradation


class SystemMonitor:
    """Continuous system resource monitoring and alerting."""

    def __init__(self, config) -> None:
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
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
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
            if cpu_percent > 90:
                await self._alert("high_cpu", f"High CPU usage: {cpu_percent:.1f}%")
            if memory.percent > 85:
                await self._alert(
                    "high_memory", f"High memory usage: {memory.percent:.1f}%"
                )

        except Exception:
            self.logger.exception("Error checking system resources")

    async def _check_gpu_status(self) -> None:
        """Check GPU status and memory usage."""
        try:
            import torch

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

                    if usage_percent > 90:
                        await self._alert(
                            "gpu_memory",
                            f"GPU {device_id} memory high: {usage_percent:.1f}%",
                        )

        except ImportError:
            pass  # PyTorch not available
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

            # Alert if process is using more than 50% of system memory
            if memory_percent > 50:
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
            if error_stats.get("error_rate", 0) > 5:  # More than 5 errors per minute
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
        print("\n" + "=" * 80)
        print("🚀 py-atranscribe Startup Validation")
        print("=" * 80)

    def print_footer(self) -> None:
        """Print startup check footer with summary."""
        elapsed_time = time.time() - self.startup_time
        print("\n" + "=" * 80)
        print(f"📊 Startup Check Summary (completed in {elapsed_time:.2f}s):")
        print(f"   ✅ Passed: {self.checks_passed}")
        print(f"   ❌ Failed: {self.checks_failed}")
        print(f"   ⚠️  Warnings: {self.warnings}")

        if self.checks_failed > 0:
            print(
                f"\n💥 CRITICAL: {self.checks_failed} checks failed! Service cannot start."
            )
            print("=" * 80)
            # Log detailed failure information
            self.logger.critical(
                "Startup failed - %d checks failed, %d warnings, completed in %.2fs",
                self.checks_failed,
                self.warnings,
                elapsed_time,
            )
            return False
        if self.warnings > 0:
            print(
                f"\n⚠️  WARNING: {self.warnings} warnings detected. Service will start with limited functionality."
            )
            self.logger.warning(
                "Startup completed with warnings - %d warnings, completed in %.2fs",
                self.warnings,
                elapsed_time,
            )
        else:
            print("\n🎉 SUCCESS: All checks passed! Service ready to start.")
            self.logger.info(
                "Startup validation successful - all %d checks passed in %.2fs",
                self.checks_passed,
                elapsed_time,
            )
        print("=" * 80)
        return True

    def check_item(self, name: str, check_func, critical: bool = True) -> bool:
        """Run a single check and report results."""
        start_time = time.time()
        try:
            result = check_func()
            elapsed = time.time() - start_time

            if result:
                print(f"✅ {name}")
                self.checks_passed += 1
                self.logger.debug("Check passed: %s (%.3fs)", name, elapsed)
                return True
            if critical:
                print(f"❌ {name}")
                self.checks_failed += 1
                self.logger.error("Critical check failed: %s (%.3fs)", name, elapsed)
            else:
                print(f"⚠️  {name}")
                self.warnings += 1
                self.logger.warning(
                    "Check failed with warning: %s (%.3fs)", name, elapsed
                )
            return False
        except Exception as e:
            elapsed = time.time() - start_time
            if critical:
                print(f"❌ {name}: {e}")
                self.checks_failed += 1
                self.logger.error(
                    "Critical check exception: %s - %s (%.3fs)", name, e, elapsed
                )
            else:
                print(f"⚠️  {name}: {e}")
                self.warnings += 1
                self.logger.warning(
                    "Check exception with warning: %s - %s (%.3fs)", name, e, elapsed
                )
            return False

    def check_python_version(self) -> bool:
        """Check Python version compatibility."""
        version = sys.version_info
        if version.major == 3 and version.minor >= 10:
            return True
        print(f"   Required: Python 3.10+, Found: {version.major}.{version.minor}")
        return False

    def check_system_commands(self) -> bool:
        """Check required system commands."""
        commands = ["ffmpeg", "ffprobe"]
        missing = []
        for cmd in commands:
            if not shutil.which(cmd):
                missing.append(cmd)

        if missing:
            print(f"   Missing commands: {', '.join(missing)}")
            return False
        return True

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

        if missing:
            print(f"   Missing dependencies: {', '.join(missing)}")
            return False
        return True

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

        if missing:
            print(f"   Missing optional dependencies: {', '.join(missing)}")
            return False
        return True

    def check_torch_installation(self) -> bool:
        """Check PyTorch installation and functionality."""
        try:
            import torch

            # Print PyTorch version
            print(f"   PyTorch version: {torch.__version__}")

            # Test basic torch functionality
            x = torch.tensor([1.0, 2.0, 3.0])
            y = x + 1

            # Test CUDA availability in PyTorch
            cuda_available = torch.cuda.is_available()
            print(f"   CUDA available in PyTorch: {cuda_available}")

            if cuda_available:
                device_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name()

                print(f"   CUDA device count: {device_count}")
                print(f"   Current device: {current_device}")
                print(f"   Device name: {device_name}")

                # Test basic CUDA tensor operations
                try:
                    x_cuda = torch.tensor([1.0]).cuda()
                    y_cuda = x_cuda + 1
                    print("   Basic CUDA tensor operations: Working")
                except Exception as cuda_error:
                    print(f"   Basic CUDA tensor operations: Failed - {cuda_error}")
                    return False

                # Check cuDNN
                cudnn_enabled = torch.backends.cudnn.enabled
                cudnn_version = (
                    torch.backends.cudnn.version() if cudnn_enabled else "N/A"
                )
                print(f"   cuDNN enabled: {cudnn_enabled}")
                print(f"   cuDNN version: {cudnn_version}")

            return True
        except Exception as e:
            print(f"   PyTorch test failed: {e}")
            return False

    def check_cuda_availability(self) -> bool:
        """Check CUDA availability and configuration with comprehensive diagnostics."""
        try:
            import torch

            # Check nvidia-smi availability
            nvidia_smi_available = self._check_nvidia_smi()

            # Check CUDA compiler
            cuda_compiler_available = self._check_cuda_compiler()

            # Check cuDNN library files
            cudnn_libraries_found = self._check_cudnn_libraries()

            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0)
                memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

                # Test cuDNN specifically
                cudnn_available = (
                    torch.backends.cudnn.enabled and torch.backends.cudnn.is_available()
                )
                cudnn_version = (
                    torch.backends.cudnn.version() if cudnn_available else "N/A"
                )

                print(
                    f"   CUDA: {device_count} device(s), {device_name}, {memory_gb:.1f}GB"
                )
                print(
                    f"   cuDNN: {'Available' if cudnn_available else 'Not Available'} (v{cudnn_version})"
                )
                print(
                    f"   nvidia-smi: {'Available' if nvidia_smi_available else 'Not Available'}"
                )
                print(
                    f"   CUDA compiler: {'Available' if cuda_compiler_available else 'Not Available'}"
                )
                print(
                    f"   cuDNN libraries: {'Found' if cudnn_libraries_found else 'Not Found'}"
                )

                # Test basic CUDA operations
                try:
                    x = torch.randn(100, 100).cuda()
                    y = torch.randn(100, 100).cuda()
                    z = torch.mm(x, y)
                    print("   CUDA tensor operations: Working")
                except Exception as cuda_op_error:
                    print(f"   CUDA tensor operations: Failed - {cuda_op_error}")
                    return False

                return cudnn_available  # Require cuDNN for full functionality
            print("   CUDA not available, will use CPU")
            print(
                f"   nvidia-smi: {'Available' if nvidia_smi_available else 'Not Available'}"
            )
            print(
                f"   CUDA compiler: {'Available' if cuda_compiler_available else 'Not Available'}"
            )
            print(
                f"   cuDNN libraries: {'Found' if cudnn_libraries_found else 'Not Found'}"
            )
            return False
        except Exception as e:
            print(f"   CUDA check failed: {e}")
            return False

    def _check_nvidia_smi(self) -> bool:
        """Check if nvidia-smi is available and working."""
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (
            subprocess.TimeoutExpired,
            subprocess.SubprocessError,
            FileNotFoundError,
        ):
            return False

    def _check_cuda_compiler(self) -> bool:
        """Check if CUDA compiler (nvcc) is available."""
        try:
            result = subprocess.run(
                ["nvcc", "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (
            subprocess.TimeoutExpired,
            subprocess.SubprocessError,
            FileNotFoundError,
        ):
            return False

    def _check_cudnn_libraries(self) -> bool:
        """Check if cuDNN library files are present in common locations."""
        cudnn_paths = [
            "/usr/lib/x86_64-linux-gnu/",
            "/usr/local/cuda/lib64/",
            "/opt/conda/lib/",
            "/usr/lib64/",
        ]

        for path in cudnn_paths:
            try:
                result = subprocess.run(
                    ["find", path, "-name", "*cudnn*"],
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
            print("   Running comprehensive CUDA diagnostics...")

            # Check nvidia-smi with detailed output
            nvidia_smi_success = self._check_nvidia_smi_detailed()

            # Check CUDA compiler with version info
            cuda_compiler_success = self._check_cuda_compiler_detailed()

            # Check cuDNN files with detailed locations
            cudnn_files_success = self._check_cudnn_files_detailed()

            # Run PyTorch CUDA diagnostics
            pytorch_cuda_success = self._check_pytorch_cuda_detailed()

            # Summary
            total_checks = 4
            passed_checks = sum(
                [
                    nvidia_smi_success,
                    cuda_compiler_success,
                    cudnn_files_success,
                    pytorch_cuda_success,
                ]
            )

            print(f"   CUDA diagnostics: {passed_checks}/{total_checks} checks passed")

            # Return True if at least PyTorch CUDA works (most important for our use case)
            return pytorch_cuda_success

        except Exception as e:
            print(f"   CUDA diagnostics failed: {e}")
            return False

    def _check_nvidia_smi_detailed(self) -> bool:
        """Check nvidia-smi with detailed output."""
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                # Extract first line with driver info
                lines = result.stdout.strip().split("\n")
                if lines:
                    print(f"     nvidia-smi: {lines[0]}")
                return True
            print("     nvidia-smi: Not available")
            return False
        except Exception:
            print("     nvidia-smi: Not available")
            return False

    def _check_cuda_compiler_detailed(self) -> bool:
        """Check CUDA compiler with version details."""
        try:
            result = subprocess.run(
                ["nvcc", "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                # Extract release line
                lines = result.stdout.strip().split("\n")
                release_line = next(
                    (line for line in lines if "release" in line.lower()), ""
                )
                print(f"     CUDA compiler: {release_line}")
                return True
            print("     CUDA compiler: Not available")
            return False
        except Exception:
            print("     CUDA compiler: Not available")
            return False

    def _check_cudnn_files_detailed(self) -> bool:
        """Check cuDNN files with detailed locations."""
        cudnn_paths = [
            "/usr/lib/x86_64-linux-gnu/",
            "/usr/local/cuda/lib64/",
            "/opt/conda/lib/",
            "/usr/lib64/",
        ]

        found_cudnn = False
        for path in cudnn_paths:
            try:
                result = subprocess.run(
                    ["find", path, "-name", "*cudnn*"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    libs = result.stdout.strip().split("\n")
                    if libs:
                        print(f"     cuDNN libraries in {path}: {len(libs)} files")
                        # Show first few libraries
                        for lib in libs[:3]:
                            print(f"       {os.path.basename(lib)}")
                        if len(libs) > 3:
                            print(f"       ... and {len(libs) - 3} more")
                        found_cudnn = True
                        break
            except Exception:
                continue

        if not found_cudnn:
            print("     cuDNN libraries: Not found in common locations")

        return found_cudnn

    def _check_pytorch_cuda_detailed(self) -> bool:
        """Check PyTorch CUDA functionality with detailed diagnostics."""
        try:
            import torch

            # Always show PyTorch version (valuable for debugging)
            print(f"     PyTorch: {torch.__version__}")

            # Show CUDA availability
            cuda_available = torch.cuda.is_available()
            print(f"     CUDA available: {cuda_available}")

            if not cuda_available:
                return False

            # Show CUDA version (from final-cuda-tests.py)
            cuda_version = torch.version.cuda
            print(f"     CUDA version: {cuda_version}")

            # Device info
            device_count = torch.cuda.device_count()
            print(f"     Device count: {device_count}")

            if device_count > 0:
                device_name = torch.cuda.get_device_name(0)
                memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"     Device name: {device_name}")
                print(f"     Device memory: {memory_gb:.1f}GB")

            # cuDNN check
            cudnn_enabled = torch.backends.cudnn.enabled
            if cudnn_enabled:
                cudnn_version = torch.backends.cudnn.version()
                print(f"     cuDNN version: {cudnn_version}")
            else:
                print("     cuDNN: Not available")

            # Test tensor operations (simplified like final-cuda-tests.py)
            try:
                # Simple tensor test
                x = torch.tensor([1.0]).cuda()
                y = x + 1  # Basic operation
                print("     ✅ CUDA tensor test passed")

                # More comprehensive test
                x_large = torch.randn(100, 100).cuda()
                y_large = torch.randn(100, 100).cuda()
                z = torch.mm(x_large, y_large)
                print("     ✅ CUDA matrix operations passed")

                # Test cuDNN-specific operations that might fail at runtime
                try:
                    # Test convolution operation that uses cuDNN
                    conv_input = torch.randn(1, 3, 32, 32).cuda()
                    conv_layer = torch.nn.Conv2d(3, 16, 3, padding=1).cuda()
                    conv_output = conv_layer(conv_input)
                    print("     ✅ cuDNN convolution operations passed")

                    # Test batch normalization that uses cuDNN
                    bn_input = torch.randn(1, 16, 32, 32).cuda()
                    bn_layer = torch.nn.BatchNorm2d(16).cuda()
                    bn_output = bn_layer(bn_input)
                    print("     ✅ cuDNN batch normalization passed")

                    return True

                except Exception as cudnn_error:
                    error_msg = str(cudnn_error).lower()
                    if "libcudnn" in error_msg or "cudnn" in error_msg:
                        print(f"     ⚠️  cuDNN runtime operations failed: {cudnn_error}")
                        print(
                            "     ℹ️  Basic CUDA works but cuDNN has issues - will use CPU fallback"
                        )
                        return False
                    else:
                        print(f"     ❌ CUDA operations failed: {cudnn_error}")
                        return False

            except Exception as tensor_error:
                print(f"     ❌ CUDA tensor operations failed: {tensor_error}")
                return False

        except Exception as e:
            print(f"     PyTorch CUDA check failed: {e}")
            return False

    def check_audio_processing(self) -> bool:
        """Check audio processing capabilities."""
        try:
            import torch
            import torchaudio

            # Test basic audio functionality
            sample_rate = 16000
            duration = 1.0
            waveform = torch.randn(1, int(sample_rate * duration))
            return True
        except Exception as e:
            print(f"   Audio processing test failed: {e}")
            return False

    def check_whisper_models(self) -> bool:
        """Check Whisper model availability."""
        try:
            from faster_whisper import WhisperModel

            # This doesn't actually load the model, just checks if we can import
            return True
        except Exception as e:
            print(f"   Whisper model check failed: {e}")
            return False

    def check_diarization_models(self) -> bool:
        """Check diarization model availability."""
        try:
            from pyannote.audio import Pipeline

            # Check if we can import the pipeline
            return True
        except Exception as e:
            print(f"   Diarization model check failed: {e}")
            return False

    def check_huggingface_token(self) -> bool:
        """Check HuggingFace token for diarization."""
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        if token:
            print(f"   HuggingFace token: {'*' * (len(token) - 4)}{token[-4:]}")
            return True
        print("   No HuggingFace token found (diarization may fail)")
        return False

    def check_file_permissions(self, config) -> bool:
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
                print(f"     {name} directory: {path}")

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
                        print(f"     {name} directory: Write access OK")
                    except Exception as e:
                        issues.append(f"{name} directory write failed: {e}")

            except Exception as e:
                issues.append(f"{name} directory: {e}")

        if issues:
            print(f"   Directory issues: {', '.join(issues)}")
            return False
        return True

    def check_config_file_permissions(self, config_path: str | None) -> bool:
        """Check configuration file read permissions."""
        try:
            # Determine config file path
            if config_path:
                config_file = Path(config_path)
            elif os.getenv("CONFIG_PATH"):
                config_file = Path(os.getenv("CONFIG_PATH"))
            else:
                # Check default locations
                default_configs = [Path("config.yaml"), Path("config.yml")]
                config_file = None
                for default_config in default_configs:
                    if default_config.exists():
                        config_file = default_config
                        break

                if not config_file:
                    print("   No config file found in default locations")
                    return True  # Not critical if using defaults

            if config_file and config_file.exists():
                print(f"     Config file: {config_file}")

                # Check if file is readable
                if not os.access(config_file, os.R_OK):
                    print(f"   Config file not readable: {config_file}")
                    return False

                # Try to actually read the file
                try:
                    with open(config_file, encoding="utf-8") as f:
                        content = f.read()
                        if len(content) == 0:
                            print(f"   Config file is empty: {config_file}")
                            return False

                    # Basic YAML syntax check
                    try:
                        import yaml

                        yaml.safe_load(content)
                        print(
                            f"     Config file: Read access OK ({len(content)} bytes, valid YAML)"
                        )
                    except ImportError:
                        print(
                            f"     Config file: Read access OK ({len(content)} bytes, YAML validation skipped)"
                        )
                    except yaml.YAMLError as yaml_err:
                        print(
                            f"   Warning: Config file has YAML syntax issues: {yaml_err}"
                        )
                        print(
                            f"     Config file: Read access OK ({len(content)} bytes, invalid YAML)"
                        )

                except Exception as e:
                    print(f"   Failed to read config file: {e}")
                    return False

                # Check parent directory write permissions (for potential config updates)
                parent_dir = config_file.parent
                if not os.access(parent_dir, os.W_OK):
                    print(f"   Warning: Config directory not writable: {parent_dir}")
                    # This is just a warning, not a failure
                else:
                    print("     Config directory: Write access OK")

            elif config_file:
                print(f"   Config file does not exist: {config_file}")
                return False

            return True

        except Exception as e:
            print(f"   Config file check failed: {e}")
            return False

    def check_logging_file_permissions(self, config) -> bool:
        """Check logging file permissions if file logging is enabled."""
        try:
            if not config.logging.file_enabled:
                print("     File logging disabled")
                return True

            log_file = Path(config.logging.file_path)
            print(f"     Log file: {log_file}")

            # Check if log file exists
            if log_file.exists():
                # Check if existing log file is writable
                if not os.access(log_file, os.W_OK):
                    print(f"   Existing log file not writable: {log_file}")
                    return False
                print("     Existing log file: Write access OK")
            else:
                # Check if parent directory exists and is writable
                parent_dir = log_file.parent

                # Try to create parent directory if it doesn't exist
                try:
                    parent_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    print(f"   Cannot create log directory: {parent_dir} - {e}")
                    return False

                # Check if parent directory is writable
                if not os.access(parent_dir, os.W_OK):
                    print(f"   Log directory not writable: {parent_dir}")
                    return False

                # Try to create and write to the log file
                try:
                    test_content = (
                        f"# Test log entry - {datetime.now(UTC).isoformat()}\n"
                    )
                    log_file.write_text(test_content, encoding="utf-8")

                    # Verify we can read it back
                    read_content = log_file.read_text(encoding="utf-8")
                    if read_content != test_content:
                        print("   Log file write/read verification failed")
                        return False

                    print("     Log file: Created and verified write access")

                    # Clean up test file if it was just created for testing
                    if log_file.stat().st_size == len(test_content):
                        log_file.unlink()
                        print("     Test log file cleaned up")

                except Exception as e:
                    print(f"   Cannot create/write log file: {log_file} - {e}")
                    return False

            # Check log file rotation permissions (if log gets large)
            parent_dir = log_file.parent
            if os.access(parent_dir, os.W_OK):
                print("     Log rotation: Directory writable for rotation")
            else:
                print(
                    f"   Warning: Log directory not writable for rotation: {parent_dir}"
                )
                # This is a warning, not a failure

            return True

        except Exception as e:
            print(f"   Log file permissions check failed: {e}")
            return False

    def check_disk_space(self, config) -> bool:
        """Check available disk space."""
        try:
            import psutil

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
                print(f"   Low disk space: {', '.join(low_space)}")
                return False
            return True
        except Exception as e:
            print(f"   Disk space check failed: {e}")
            return False

    def check_memory_availability(self) -> bool:
        """Check system memory availability."""
        try:
            import psutil

            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            available_gb = memory.available / (1024**3)

            print(
                f"   Memory: {available_gb:.1f}GB available / {memory_gb:.1f}GB total"
            )

            if available_gb < 2.0:
                print("   Warning: Low memory may cause processing issues")
                return False
            return True
        except Exception as e:
            print(f"   Memory check failed: {e}")
            return False

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
                print("     Not running in container")
                return True

            print("     Container environment detected")

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
                        stat_info = path.stat()
                        print(f"     {description}: {path} (mounted)")
                    except Exception as e:
                        mount_issues.append(f"{description} at {path}: {e}")
                else:
                    print(f"     {description}: {path} (not mounted)")

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
                    print(f"     {env_var}: {value}")
                else:
                    missing_env.append(env_var)

            if missing_env:
                print(
                    f"   Missing container environment variables: {', '.join(missing_env)}"
                )

            # This is informational, not a failure
            return True

        except Exception as e:
            print(f"   Container mount check failed: {e}")
            return False

    def check_network_connectivity(self) -> bool:
        """Check network connectivity for model downloads."""
        try:
            import socket

            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except Exception:
            print("   No internet connectivity (model downloads may fail)")
            return False

    def run_all_checks(self, config, config_path: str | None = None) -> bool:
        """Run all startup checks."""
        self.print_header()

        print("\n🔍 System Requirements:")
        self.check_item("Python Version (3.10+)", self.check_python_version)
        self.check_item("System Commands (ffmpeg, ffprobe)", self.check_system_commands)

        print("\n📦 Python Dependencies:")
        self.check_item("Critical Dependencies", self.check_python_dependencies)
        self.check_item(
            "Optional Dependencies", self.check_optional_dependencies, critical=False
        )

        print("\n🧠 ML Framework:")
        self.check_item("PyTorch Installation", self.check_torch_installation)
        self.check_item("CUDA Support", self.check_cuda_availability, critical=False)
        self.check_item("CUDA Diagnostics", self.check_cuda_diagnostics, critical=False)
        self.check_item("Audio Processing", self.check_audio_processing)

        print("\n🎯 AI Models:")
        self.check_item("Whisper Models", self.check_whisper_models)
        self.check_item("Diarization Models", self.check_diarization_models)
        self.check_item(
            "HuggingFace Token", self.check_huggingface_token, critical=False
        )

        print("\n📁 File & Directory Permissions:")
        self.check_item(
            "Configuration File Access",
            lambda: self.check_config_file_permissions(config_path),
        )
        self.check_item(
            "Logging File Permissions",
            lambda: self.check_logging_file_permissions(config),
        )
        self.check_item(
            "Working Directory Permissions", lambda: self.check_file_permissions(config)
        )
        self.check_item(
            "Container Volume Mounts", self.check_container_mounts, critical=False
        )

        print("\n💾 Storage & Resources:")
        self.check_item(
            "Disk Space", lambda: self.check_disk_space(config), critical=False
        )
        self.check_item(
            "Memory Availability", self.check_memory_availability, critical=False
        )

        print("\n🌐 Network:")
        self.check_item(
            "Internet Connectivity", self.check_network_connectivity, critical=False
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

                # Update stats
                self._processing_stats["files_queued"] += 1
                self._processing_stats["last_activity"] = datetime.now(UTC).isoformat()
                worker_stats["last_activity"] = time.time()

                # Process the file
                processing_start = time.time()
                try:

                    def progress_callback(progress: ProcessingProgress) -> None:
                        worker_logger.debug(
                            "Worker %s Progress: %s - %.1f%% - %s",
                            worker_id,
                            progress.stage,
                            progress.progress,
                            progress.message,
                        )

                    result = await self.batch_transcriber.process_file(
                        file_path=file_path,
                        progress_callback=progress_callback,
                    )

                    processing_time = time.time() - processing_start

                    if result.success:
                        worker_logger.info(
                            "✅ Worker %s successfully processed %s in %.2fs (transcription: %.2fs)",
                            worker_id,
                            file_path.name,
                            processing_time,
                            result.processing_time,
                        )

                        # Update statistics
                        worker_stats["files_processed"] += 1
                        worker_stats["total_processing_time"] += processing_time
                        self._processing_stats["files_completed"] += 1
                        self._processing_stats["total_processing_time"] += (
                            processing_time
                        )

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
                    else:
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

                except Exception as e:
                    processing_time = time.time() - processing_start
                    worker_logger.exception(
                        "💥 Worker %s unexpected error processing %s after %.2fs: %s",
                        worker_id,
                        file_path.name,
                        processing_time,
                        e,
                    )
                    worker_stats["files_failed"] += 1
                    self._processing_stats["files_failed"] += 1

                finally:
                    # Mark task as done
                    await self.processing_queue.mark_done(file_path)

            except asyncio.CancelledError:
                worker_logger.info("Worker %s cancelled", worker_id)
                break
            except Exception:
                worker_logger.exception("Error in worker %s", worker_id)
                await asyncio.sleep(1.0)  # Brief pause before continuing

        # Log final worker statistics
        worker_logger.info(
            "Processing worker %s stopped - Processed: %d, Failed: %d, Total time: %.2fs",
            worker_id,
            worker_stats["files_processed"],
            worker_stats["files_failed"],
            worker_stats["total_processing_time"],
        )

    def _on_file_detected(self, file_path: Path) -> None:
        """Handle callback when a new file is detected and ready for processing.

        Args:
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

            # Log queue status
            asyncio.create_task(self._log_queue_status())

        except Exception:
            self.logger.exception("❌ Failed to queue file %s", file_path)

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
                    "Health check server stopped in %.2fs", health_stop_time
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
                    "Cancelling %d worker tasks...", len(self.worker_tasks)
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
                    "Batch transcriber cleanup completed in %.2fs", cleanup_time
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
                "Periodic status logging enabled (interval: %ds)", args.status_interval
            )

        # Start the service
        await service.start()

        total_startup_time = time.time() - app_start_time
        logger.info("🎉 Application fully started in %.2fs", total_startup_time)

        # Wait for completion
        await service.wait_for_completion()

    except KeyboardInterrupt:
        logger.info("🛑 Received interrupt signal, shutting down...")
        print("\nReceived interrupt signal, shutting down...")

        # Cancel status logging
        if "status_task" in locals() and status_task:
            status_task.cancel()

        sys.exit(0)
    except Exception as e:
        # Log the exception details for debugging
        logger.exception(
            "💥 Fatal error occurred after %.2fs", time.time() - app_start_time
        )
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


async def _periodic_status_logger(service: TranscriptionService, interval: int) -> None:
    """Periodically log service status and statistics.

    Args:
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
                "=== Periodic Status Report (uptime: %.1f hours) ===", uptime / 3600
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
            try:
                import torch

                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        allocated = torch.cuda.memory_allocated(i) / (1024**3)
                        reserved = torch.cuda.memory_reserved(i) / (1024**3)
                        total = torch.cuda.get_device_properties(i).total_memory / (
                            1024**3
                        )
                        usage_percent = (reserved / total) * 100 if total > 0 else 0
                        logger.info(
                            "🚀 GPU %d: %.1fGB/%.1fGB (%.1f%%)",
                            i,
                            reserved,
                            total,
                            usage_percent,
                        )
            except ImportError:
                pass

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
        print("\nShutdown complete.")
        sys.exit(0)
