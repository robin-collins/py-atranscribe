"""Expose health check endpoints for container monitoring and orchestration.

Provides system status, resource monitoring, and service health information.
"""

import asyncio
import logging
import os
import time
from datetime import UTC, datetime
from typing import Any

import psutil
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse

from src.config import AppConfig
from src.utils.error_handling import error_tracker

# Import for type hint (avoid circular import)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.monitoring.file_monitor import ProcessingQueue


class HealthChecker:
    """Health checker that monitors system resources and service status.

    Provides health endpoints for container orchestration platforms.
    """

    def __init__(self, config: AppConfig) -> None:
        """Initialize HealthChecker.

        Args:
        ----
            config: Application configuration

        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.start_time = time.time()
        self._processing_queue_ref = None

        # Service status tracking
        self._service_status = {
            "status": "starting",
            "file_monitor_running": False,
            "transcriber_initialized": False,
            "last_activity": None,
        }

        # Create FastAPI app
        self.app = FastAPI(
            title="py-atranscribe Health Check",
            description="Health monitoring for automated transcription service",
            version="1.0.0",
        )

        # Register endpoints
        self._register_endpoints()

    def _register_endpoints(self) -> None:
        """Register health check endpoints."""
        self._register_health_endpoint()
        self._register_ready_endpoint()
        self._register_live_endpoint()
        self._register_detailed_endpoint()
        self._register_metrics_endpoint()

    def _register_health_endpoint(self) -> None:
        @self.app.get("/health")
        async def health_check() -> JSONResponse:
            """Perform basic health check.

            Return 200 if service is healthy, 503 if unhealthy.
            Returns simple healthy/unhealthy status.
            """
            try:
                health_data = await self.get_health_status()

                # Simple response for basic health check
                simple_response = {
                    "status": health_data["overall_status"],
                    "timestamp": health_data["timestamp"],
                    "uptime_seconds": health_data["uptime_seconds"],
                }

                if health_data["overall_status"] == "healthy":
                    return JSONResponse(status_code=200, content=simple_response)
                # Include basic error info for unhealthy status
                simple_response["issues"] = []
                for check_name, check_status in health_data["checks"].items():
                    if check_status not in ["healthy", "running", "initialized"]:
                        simple_response["issues"].append(
                            f"{check_name}: {check_status}",
                        )

                return JSONResponse(status_code=503, content=simple_response)

            except Exception as exc:
                self.logger.exception("Health check failed")
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "unhealthy",
                        "error": str(exc),
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                )

    def _register_ready_endpoint(self) -> None:
        @self.app.get("/health/ready")
        async def readiness_check() -> dict[str, Any]:
            """Check Kubernetes readiness.

            Return 200 when service is ready to handle requests.
            """
            if self._service_status["status"] == "ready":
                return {"status": "ready", "timestamp": datetime.now(UTC).isoformat()}
            raise HTTPException(
                status_code=503,
                detail=f"Service not ready: {self._service_status['status']}",
            )

    def _register_live_endpoint(self) -> None:
        @self.app.get("/health/live")
        async def liveness_check() -> dict[str, Any]:
            """Check Kubernetes liveness.

            Return 200 if service is alive and not deadlocked.
            """
            # Simple liveness check - if we can respond, we're alive
            return {"status": "alive", "timestamp": datetime.now(UTC).isoformat()}

    def _register_detailed_endpoint(self) -> None:
        @self.app.get("/health/detailed")
        async def detailed_health() -> JSONResponse:
            """Return comprehensive health information including resource usage and statistics."""
            try:
                detailed_health_info = await self.get_detailed_health()
                return JSONResponse(content=detailed_health_info)
            except Exception as exc:
                self.logger.exception("Detailed health check failed")
                raise HTTPException(status_code=500, detail=str(exc)) from exc

    def _register_metrics_endpoint(self) -> None:
        @self.app.get("/metrics")
        async def metrics() -> Response:
            """Expose Prometheus-style metrics endpoint."""
            try:
                metrics_data = await self.get_metrics()
                prometheus_metrics = self._format_prometheus_metrics(metrics_data)
                return Response(content=prometheus_metrics, media_type="text/plain")
            except Exception as exc:
                self.logger.exception("Metrics endpoint failed")
                raise HTTPException(status_code=500, detail=str(exc)) from exc

    async def get_health_status(self) -> dict[str, Any]:
        """Get overall health status.

        Returns
        -------
            Dict with health status information

        """
        # Check system resources
        disk_healthy = self.check_disk_space()
        memory_healthy = self.check_memory_usage()
        queue_healthy = await self.check_processing_queue()

        # Determine overall health
        overall_healthy = (
            disk_healthy
            and memory_healthy
            and queue_healthy
            and self._service_status["status"] in ["ready", "running"]
        )

        return {
            "overall_status": "healthy" if overall_healthy else "unhealthy",
            "service_status": self._service_status["status"],
            "checks": {
                "disk_space": "healthy" if disk_healthy else "unhealthy",
                "memory_usage": "healthy" if memory_healthy else "unhealthy",
                "processing_queue": "healthy" if queue_healthy else "unhealthy",
                "file_monitor": "running"
                if self._service_status["file_monitor_running"]
                else "stopped",
                "transcriber": "initialized"
                if self._service_status["transcriber_initialized"]
                else "not_initialized",
            },
            "uptime_seconds": time.time() - self.start_time,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    async def get_detailed_health(self) -> dict[str, Any]:
        """Get detailed health information including resource usage.

        Returns
        -------
            Dict with detailed health and system information

        """
        # Get basic health
        basic_health = await self.get_health_status()

        # Add detailed system information
        memory_info = psutil.virtual_memory()
        disk_info = psutil.disk_usage(str(self.config.directories.input))

        detailed = {
            **basic_health,
            "system": {
                "cpu_count": psutil.cpu_count(),
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory": {
                    "total_gb": round(memory_info.total / (1024**3), 2),
                    "available_gb": round(memory_info.available / (1024**3), 2),
                    "used_gb": round(memory_info.used / (1024**3), 2),
                    "used_percent": round(memory_info.percent, 1),
                    "free_gb": round(
                        (memory_info.total - memory_info.used) / (1024**3),
                        2,
                    ),
                },
                "disk": {
                    "total_gb": round(disk_info.total / (1024**3), 2),
                    "free_gb": round(disk_info.free / (1024**3), 2),
                    "used_gb": round(disk_info.used / (1024**3), 2),
                    "used_percent": round((disk_info.used / disk_info.total) * 100, 1),
                },
                "load_average": list(psutil.getloadavg())
                if hasattr(psutil, "getloadavg")
                else None,
                "boot_time": psutil.boot_time(),
            },
            "configuration": {
                "input_directory": str(self.config.directories.input),
                "output_directory": str(self.config.directories.output),
                "backup_directory": str(self.config.directories.backup),
                "temp_directory": str(self.config.directories.temp),
                "supported_formats": self.config.monitoring.supported_formats,
                "whisper_model": self.config.transcription.whisper.model_size,
                "whisper_device": self.config.transcription.whisper.device,
                "whisper_compute_type": self.config.transcription.whisper.compute_type,
                "diarization_enabled": self.config.diarization.enabled,
                "diarization_model": self.config.diarization.model
                if self.config.diarization.enabled
                else None,
                "output_formats": self.config.transcription.output_formats,
                "stability_delay": self.config.monitoring.stability_delay,
                "poll_interval": self.config.monitoring.poll_interval,
                "max_concurrent_files": self.config.performance.max_concurrent_files,
                "post_processing_action": self.config.post_processing.action,
                "logging_level": self.config.logging.level,
                "health_check_enabled": self.config.health_check.enabled,
            },
            "error_statistics": error_tracker.get_error_stats(),
            "process_info": {
                "pid": os.getpid(),
                "parent_pid": os.getppid(),
                "cpu_percent": psutil.Process().cpu_percent(),
                "memory_info": {
                    "rss_mb": round(psutil.Process().memory_info().rss / (1024**2), 1),
                    "vms_mb": round(psutil.Process().memory_info().vms / (1024**2), 1),
                    "percent": round(psutil.Process().memory_percent(), 1),
                },
                "num_threads": psutil.Process().num_threads(),
                "create_time": psutil.Process().create_time(),
            },
        }

        # Add detailed directory information
        detailed["directories"] = {}
        for name, path in [
            ("input", self.config.directories.input),
            ("output", self.config.directories.output),
            ("backup", self.config.directories.backup),
            ("temp", self.config.directories.temp),
        ]:
            if path.exists():
                try:
                    disk_usage = psutil.disk_usage(str(path))
                    file_count = len(list(path.iterdir())) if path.is_dir() else 0
                    detailed["directories"][name] = {
                        "path": str(path),
                        "exists": True,
                        "is_directory": path.is_dir(),
                        "file_count": file_count,
                        "disk_usage": {
                            "total_gb": round(disk_usage.total / (1024**3), 2),
                            "free_gb": round(disk_usage.free / (1024**3), 2),
                            "used_percent": round(
                                (disk_usage.used / disk_usage.total) * 100,
                                1,
                            ),
                        },
                    }
                except (OSError, PermissionError, FileNotFoundError) as e:
                    detailed["directories"][name] = {
                        "path": str(path),
                        "exists": True,
                        "error": str(e),
                    }
            else:
                detailed["directories"][name] = {
                    "path": str(path),
                    "exists": False,
                }

        # Add GPU information if available
        try:
            import torch

            if torch.cuda.is_available():
                gpu_info = {
                    "available": True,
                    "device_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "devices": [],
                }

                for i in range(torch.cuda.device_count()):
                    device_props = torch.cuda.get_device_properties(i)
                    gpu_info["devices"].append(
                        {
                            "id": i,
                            "name": device_props.name,
                            "total_memory_gb": round(
                                device_props.total_memory / (1024**3),
                                2,
                            ),
                            "memory_allocated_gb": round(
                                torch.cuda.memory_allocated(i) / (1024**3),
                                2,
                            ),
                            "memory_reserved_gb": round(
                                torch.cuda.memory_reserved(i) / (1024**3),
                                2,
                            ),
                            "memory_usage_percent": round(
                                (
                                    torch.cuda.memory_allocated(i)
                                    / device_props.total_memory
                                )
                                * 100,
                                1,
                            ),
                            "major": device_props.major,
                            "minor": device_props.minor,
                            "multi_processor_count": device_props.multi_processor_count,
                        },
                    )

                detailed["system"]["gpu"] = gpu_info
            else:
                detailed["system"]["gpu"] = {
                    "available": False,
                    "reason": "CUDA not available",
                }
        except ImportError:
            detailed["system"]["gpu"] = {
                "available": False,
                "error": "PyTorch not available",
            }
        except (RuntimeError, AttributeError) as e:
            detailed["system"]["gpu"] = {
                "available": False,
                "error": str(e),
            }

        # Add queue status if available
        if self._processing_queue_ref:
            try:
                queue_status = await self._processing_queue_ref.get_status()
                detailed["processing_queue"] = {
                    "queued": queue_status.get("queued", 0),
                    "processing": queue_status.get("processing", 0),
                    "max_size": queue_status.get("max_size", 0),
                    "total_items": queue_status.get("queued", 0)
                    + queue_status.get("processing", 0),
                }
            except (AttributeError, RuntimeError, ConnectionError) as e:
                detailed["processing_queue"] = {
                    "error": str(e),
                }
        else:
            detailed["processing_queue"] = {
                "status": "not_connected",
            }

        return detailed

    async def get_metrics(self) -> dict[str, Any]:
        """Get metrics for monitoring systems.

        Returns
        -------
            Dict with metrics data

        """
        memory_info = psutil.virtual_memory()
        disk_info = psutil.disk_usage(str(self.config.directories.input))
        error_stats = error_tracker.get_error_stats()

        metrics = {
            "system_cpu_percent": psutil.cpu_percent(),
            "system_memory_total_bytes": memory_info.total,
            "system_memory_available_bytes": memory_info.available,
            "system_memory_used_percent": memory_info.percent,
            "system_disk_total_bytes": disk_info.total,
            "system_disk_free_bytes": disk_info.free,
            "system_disk_used_percent": (disk_info.used / disk_info.total) * 100,
            "service_uptime_seconds": time.time() - self.start_time,
            "service_status": 1 if self._service_status["status"] == "ready" else 0,
            "file_monitor_running": 1
            if self._service_status["file_monitor_running"]
            else 0,
            "transcriber_initialized": 1
            if self._service_status["transcriber_initialized"]
            else 0,
            "error_total": error_stats.get("total_errors", 0),
            "error_recent": error_stats.get("recent_errors", 0),
            "error_rate_per_minute": error_stats.get("error_rate", 0),
        }

        # Add GPU metrics if available
        try:
            import torch

            if torch.cuda.is_available():
                metrics.update(
                    {
                        "gpu_available": 1,
                        "gpu_memory_allocated_bytes": torch.cuda.memory_allocated(0),
                        "gpu_memory_reserved_bytes": torch.cuda.memory_reserved(0),
                    },
                )
            else:
                metrics["gpu_available"] = 0
        except ImportError:
            metrics["gpu_available"] = 0

        return metrics

    def _format_prometheus_metrics(self, metrics: dict[str, Any]) -> str:
        """Format metrics in Prometheus exposition format."""
        lines = []

        for key, value in metrics.items():
            if isinstance(value, int | float):
                lines.append(f"# HELP {key} Metric from py-atranscribe")
                lines.append(f"# TYPE {key} gauge")
                lines.append(f"{key} {value}")
                lines.append("")

        return "\n".join(lines)

    def check_disk_space(self) -> bool:
        """Check if sufficient disk space is available.

        Returns
        -------
            bool: True if disk space is sufficient

        """
        try:
            for directory in [
                self.config.directories.input,
                self.config.directories.output,
                self.config.directories.backup,
            ]:
                if directory.exists():
                    disk_usage = psutil.disk_usage(str(directory))
                    free_gb = disk_usage.free / (1024**3)

                    if free_gb < self.config.health_check.disk_space_min_gb:
                        self.logger.warning(
                            "Low disk space in %s: %.2fGB",
                            directory,
                            free_gb,
                        )
                        return False
        except Exception:
            self.logger.exception("Error checking disk space")
            return False
        else:
            return True

    def check_memory_usage(self) -> bool:
        """Check if memory usage is within acceptable limits.

        Returns
        -------
            bool: True if memory usage is acceptable

        """
        try:
            memory_info = psutil.virtual_memory()

            for _ in range(1):
                if (
                    memory_info.percent
                    > self.config.health_check.memory_usage_max_percent
                ):
                    self.logger.warning(
                        "High memory usage: %.1f%%",
                        memory_info.percent,
                    )
                    return False
        except Exception:
            self.logger.exception("Error checking memory usage")
            return False
        else:
            return True

    async def check_processing_queue(self) -> bool:
        """Check if processing queue is healthy."""
        try:
            if self._processing_queue_ref:
                queue_status = await self._processing_queue_ref.get_status()
                queue_size = queue_status.get("queued", 0) + queue_status.get(
                    "processing",
                    0,
                )

                # Check if queue is too large
                if queue_size > self.config.health_check.queue_size_max:
                    self.logger.warning(
                        "Processing queue too large: %d items",
                        queue_size,
                    )
                    return False

                return True

            # If no queue reference, assume healthy
        except (AttributeError, RuntimeError, ConnectionError) as e:
            self.logger.warning("Error checking processing queue: %s", e)
            return False
        else:
            return True

    def set_processing_queue_ref(self, processing_queue: "ProcessingQueue") -> None:
        """Set reference to processing queue for health checks."""
        self._processing_queue_ref = processing_queue

    def update_service_status(self, status: str, **kwargs: object) -> None:
        """Update service status.

        Args:
        ----
            status: New service status
            **kwargs: Additional status fields to update

        """
        self._service_status["status"] = status
        self._service_status["last_activity"] = datetime.now(UTC).isoformat()

        for key, value in kwargs.items():
            self._service_status[key] = value

        self.logger.debug("Service status updated: %s", status)

    async def start_server(self) -> None:
        """Start the health check server."""
        if not self.config.health_check.enabled:
            self.logger.info("Health check server disabled")
            return

        try:
            config = uvicorn.Config(
                app=self.app,
                host=self.config.health_check.host,
                port=self.config.health_check.port,
                log_level="warning",  # Reduce uvicorn log verbosity
                access_log=False,  # Disable access logs for health checks
            )

            server = uvicorn.Server(config)

            self.logger.info(
                "Starting health check server on %s:%d",
                self.config.health_check.host,
                self.config.health_check.port,
            )

            # Start server in background task and store the task reference
            self._server_task = asyncio.create_task(server.serve())

        except Exception:
            self.logger.exception("Failed to start health check server")
            raise

    async def stop_server(self) -> None:
        """Stop the health check server."""
        # Update status to indicate shutdown
        self.update_service_status("stopping")
        self.logger.info("Health check server stopped")
