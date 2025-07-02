"""Health check endpoint for container monitoring and orchestration.
Provides system status, resource monitoring, and service health information.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any

import psutil
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse

from src.config import AppConfig
from src.utils.error_handling import error_tracker


class HealthChecker:
    """Health checker that monitors system resources and service status.
    Provides health endpoints for container orchestration platforms.
    """

    def __init__(self, config: AppConfig) -> None:
        """Initialize HealthChecker.

        Args:
            config: Application configuration

        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.start_time = time.time()

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

        @self.app.get("/health")
        async def health_check():
            """Basic health check endpoint.
            Returns 200 if service is healthy, 503 if unhealthy.
            """
            try:
                health_data = await self.get_health_status()

                if health_data["overall_status"] == "healthy":
                    return JSONResponse(status_code=200, content=health_data)
                return JSONResponse(status_code=503, content=health_data)

            except Exception as e:
                self.logger.exception(f"Health check failed: {e}")
                return JSONResponse(
                    status_code=503,
                    content={
                        "overall_status": "unhealthy",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    },
                )

        @self.app.get("/health/ready")
        async def readiness_check():
            """Kubernetes readiness probe endpoint.
            Returns 200 when service is ready to handle requests.
            """
            if self._service_status["status"] == "ready":
                return {"status": "ready", "timestamp": datetime.now().isoformat()}
            raise HTTPException(
                status_code=503,
                detail=f"Service not ready: {self._service_status['status']}",
            )

        @self.app.get("/health/live")
        async def liveness_check():
            """Kubernetes liveness probe endpoint.
            Returns 200 if service is alive and not deadlocked.
            """
            # Simple liveness check - if we can respond, we're alive
            return {"status": "alive", "timestamp": datetime.now().isoformat()}

        @self.app.get("/health/detailed")
        async def detailed_health():
            """Detailed health information including resource usage and statistics."""
            try:
                detailed_health = await self.get_detailed_health()
                return JSONResponse(content=detailed_health)
            except Exception as e:
                self.logger.exception(f"Detailed health check failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/metrics")
        async def metrics():
            """Prometheus-style metrics endpoint."""
            try:
                metrics_data = await self.get_metrics()

                # Convert to Prometheus format
                prometheus_metrics = self._format_prometheus_metrics(metrics_data)

                return Response(content=prometheus_metrics, media_type="text/plain")
            except Exception as e:
                self.logger.exception(f"Metrics endpoint failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    async def get_health_status(self) -> dict[str, Any]:
        """Get overall health status.

        Returns:
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
            "timestamp": datetime.now().isoformat(),
        }

    async def get_detailed_health(self) -> dict[str, Any]:
        """Get detailed health information including resource usage.

        Returns:
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
                    "total_gb": memory_info.total / (1024**3),
                    "available_gb": memory_info.available / (1024**3),
                    "used_percent": memory_info.percent,
                },
                "disk": {
                    "total_gb": disk_info.total / (1024**3),
                    "free_gb": disk_info.free / (1024**3),
                    "used_percent": (disk_info.used / disk_info.total) * 100,
                },
            },
            "configuration": {
                "input_directory": str(self.config.directories.input),
                "output_directory": str(self.config.directories.output),
                "supported_formats": self.config.monitoring.supported_formats,
                "whisper_model": self.config.transcription.whisper.model_size,
                "diarization_enabled": self.config.diarization.enabled,
                "output_formats": self.config.transcription.output_formats,
            },
            "error_statistics": error_tracker.get_error_stats(),
        }

        # Add GPU information if available
        try:
            import torch

            if torch.cuda.is_available():
                detailed["system"]["gpu"] = {
                    "available": True,
                    "device_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "device_name": torch.cuda.get_device_name(0),
                    "memory_allocated_gb": torch.cuda.memory_allocated(0) / (1024**3),
                    "memory_reserved_gb": torch.cuda.memory_reserved(0) / (1024**3),
                }
            else:
                detailed["system"]["gpu"] = {"available": False}
        except ImportError:
            detailed["system"]["gpu"] = {
                "available": False,
                "error": "PyTorch not available",
            }

        return detailed

    async def get_metrics(self) -> dict[str, Any]:
        """Get metrics for monitoring systems.

        Returns:
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

        Returns:
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
                            f"Low disk space in {directory}: {free_gb:.2f}GB",
                        )
                        return False

            return True

        except Exception as e:
            self.logger.exception(f"Error checking disk space: {e}")
            return False

    def check_memory_usage(self) -> bool:
        """Check if memory usage is within acceptable limits.

        Returns:
            bool: True if memory usage is acceptable

        """
        try:
            memory_info = psutil.virtual_memory()

            if memory_info.percent > self.config.health_check.memory_usage_max_percent:
                self.logger.warning("High memory usage: %.1f%%", memory_info.percent)
                return False

            return True

        except Exception as e:
            self.logger.exception(f"Error checking memory usage: {e}")
            return False

    async def check_processing_queue(self) -> bool:
        """Check if processing queue is healthy.

        Returns:
            bool: True if queue is healthy

        """
        # This would need to be connected to the actual processing queue
        # For now, return True as a placeholder
        return True

    def update_service_status(self, status: str, **kwargs) -> None:
        """Update service status.

        Args:
            status: New service status
            **kwargs: Additional status fields to update

        """
        self._service_status["status"] = status
        self._service_status["last_activity"] = datetime.now().isoformat()

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
                f"Starting health check server on "
                f"{self.config.health_check.host}:{self.config.health_check.port}",
            )

            # Start server in background task
            asyncio.create_task(server.serve())

        except Exception as e:
            self.logger.exception(f"Failed to start health check server: {e}")
            raise

    async def stop_server(self) -> None:
        """Stop the health check server."""
        # Update status to indicate shutdown
        self.update_service_status("stopping")
        self.logger.info("Health check server stopped")
