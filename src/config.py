"""Configuration management for the automated transcription application.

Handles loading and validation of configuration from YAML files and environment
variables.
"""

import os
import re
import tempfile
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DirectoriesConfig(BaseModel):
    """Directory configuration for input, output, backup, and temporary files."""

    input: Path = Field(
        default=Path("/data/in"),
        description="Input directory for audio files",
    )
    output: Path = Field(
        default=Path("/data/out"),
        description="Output directory for transcripts",
    )
    backup: Path = Field(
        default=Path("/data/backup"),
        description="Backup directory for processed files",
    )
    temp: Path = Field(
        default=Path(tempfile.gettempdir()) / "transcribe",
        description="Temporary directory for processing",
    )


class FileMonitoringConfig(BaseModel):
    """File monitoring configuration."""

    supported_formats: list[str] = Field(
        default=[
            ".wav",
            ".mp3",
            ".flac",
            ".m4a",
            ".aac",
            ".ogg",
            ".opus",
            ".wma",
            ".aiff",
            ".au",
            ".mp4",
            ".mkv",
            ".avi",
            ".mov",
            ".wmv",
            ".flv",
            ".webm",
            ".3gp",
            ".m4v",
            ".f4v",
            ".asf",
            ".rm",
            ".rmvb",
            ".vob",
            ".ts",
            ".mts",
            ".m2ts",
            ".divx",
            ".xvid",
            ".dv",
            ".f4a",
            ".f4b",
        ],
        description="Supported audio/video file formats",
    )
    stability_delay: float = Field(
        default=5.0,
        ge=0,
        description="Seconds to wait before processing new files",
    )
    poll_interval: float = Field(
        default=1.0,
        ge=0.1,
        description="File monitoring poll interval in seconds",
    )


class WhisperConfig(BaseModel):
    """Whisper transcription model configuration."""

    model_size: str = Field(default="medium", description="Whisper model size")
    device: str = Field(
        default="auto",
        description="Device for inference (auto, cpu, cuda)",
    )
    compute_type: str = Field(
        default="auto",
        description="Compute type (auto, int8, int16, float16, float32)",
    )
    cpu_threads: int = Field(
        default=0,
        ge=0,
        description="Number of CPU threads (0 = auto)",
    )
    num_workers: int = Field(default=1, ge=1, description="Number of parallel workers")
    initial_prompt: str | None = Field(
        default=None,
        description="Initial prompt to guide transcription",
    )

    @field_validator("model_size")
    @classmethod
    def validate_model_size(cls, v: str) -> str:
        """Validate the model size."""
        valid_sizes = [
            "tiny",
            "base",
            "small",
            "medium",
            "large-v1",
            "large-v2",
            "large-v3",
        ]
        if v not in valid_sizes:
            msg = f"model_size must be one of {valid_sizes}"
            raise ValueError(msg)
        return v

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate the device."""
        valid_devices = ["auto", "cpu", "cuda"]
        if v not in valid_devices:
            msg = f"device must be one of {valid_devices}"
            raise ValueError(msg)
        return v


class PreprocessingConfig(BaseModel):
    """Audio preprocessing configuration."""

    enable_vad: bool = Field(
        default=True,
        description="Enable Voice Activity Detection",
    )
    vad_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="VAD confidence threshold",
    )
    enable_bgm_separation: bool = Field(
        default=False,
        description="Enable background music separation",
    )
    chunk_length_s: int = Field(
        default=30,
        ge=1,
        description="Audio chunk duration for processing",
    )


class TranscriptionConfig(BaseModel):
    """Transcription pipeline configuration."""

    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    language: str = Field(
        default="auto",
        description="Language for transcription (auto or language code)",
    )
    output_formats: list[str] = Field(
        default=["srt", "webvtt", "txt", "json", "tsv", "lrc"],
        description="Output formats to generate",
    )


class DiarizationConfig(BaseModel):
    """Speaker diarization configuration."""

    enabled: bool = Field(default=True, description="Enable speaker diarization")
    hf_token: str | None = Field(
        default=None,
        description="HuggingFace token for model access",
    )
    model: str = Field(
        default="pyannote/speaker-diarization-3.1",
        description="Diarization model",
    )
    device: str = Field(
        default="auto",
        description="Device for diarization (auto, cpu, cuda)",
    )
    min_speakers: int = Field(default=1, ge=1, description="Minimum number of speakers")
    max_speakers: int = Field(
        default=10,
        ge=1,
        description="Maximum number of speakers",
    )
    embedding_model: str = Field(
        default="pyannote/wespeaker-voxceleb-resnet34-LM",
        description="Speaker embedding model",
    )


class RetryConfig(BaseModel):
    """Retry configuration for error handling."""

    max_attempts: int = Field(default=3, ge=1, description="Maximum retry attempts")
    base_delay: float = Field(
        default=1.0,
        ge=0,
        description="Base delay between retries",
    )
    max_delay: float = Field(
        default=60.0,
        ge=0,
        description="Maximum delay between retries",
    )
    exponential_base: float = Field(
        default=2.0,
        ge=1.0,
        description="Exponential backoff base",
    )
    jitter: bool = Field(default=True, description="Add random jitter to retry delays")


class PerformanceConfig(BaseModel):
    """Performance optimization configuration."""

    max_memory_usage_gb: float = Field(
        default=8.0,
        ge=0.5,
        description="Maximum memory usage limit",
    )
    enable_model_offload: bool = Field(
        default=True,
        description="Enable model offloading to save GPU memory",
    )
    gpu_memory_fraction: float = Field(
        default=0.8,
        ge=0.1,
        le=1.0,
        description="Fraction of GPU memory to use",
    )
    max_concurrent_files: int = Field(
        default=2,
        ge=1,
        description="Maximum concurrent file processing",
    )
    batch_size: int = Field(
        default=16,
        ge=1,
        description="Batch size for model inference",
    )
    retry: RetryConfig = Field(default_factory=RetryConfig)


class PostProcessingConfig(BaseModel):
    """Post-processing configuration for handled files."""

    action: str = Field(
        default="move",
        description="Action to take after processing (move, delete, keep)",
    )
    backup_structure: str = Field(
        default="date",
        description="Backup directory structure (flat, date, original)",
    )

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        """Validate the post-processing action."""
        valid_actions = ["move", "delete", "keep"]
        if v not in valid_actions:
            msg = f"action must be one of {valid_actions}"
            raise ValueError(msg)
        return v

    @field_validator("backup_structure")
    @classmethod
    def validate_backup_structure(cls, v: str) -> str:
        """Validate the backup structure."""
        valid_structures = ["flat", "date", "original"]
        if v not in valid_structures:
            msg = f"backup_structure must be one of {valid_structures}"
            raise ValueError(msg)
        return v


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="structured",
        description="Log format (structured or plain)",
    )
    file_enabled: bool = Field(default=False, description="Enable file logging")
    file_path: Path = Field(
        default=Path("/var/log/transcribe.log"),
        description="Log file path",
    )

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        """Validate the logging level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            msg = f"level must be one of {valid_levels}"
            raise ValueError(msg)
        return v.upper()


class HealthCheckConfig(BaseModel):
    """Health check endpoint configuration."""

    enabled: bool = Field(default=True, description="Enable health check endpoint")
    port: int = Field(default=8000, ge=1, le=65535, description="Health check port")
    host: str = Field(default="127.0.0.1", description="Health check host")
    disk_space_min_gb: float = Field(
        default=1.0,
        ge=0,
        description="Minimum disk space threshold",
    )
    memory_usage_max_percent: float = Field(
        default=90.0,
        ge=0,
        le=100,
        description="Maximum memory usage threshold",
    )
    queue_size_max: int = Field(
        default=100,
        ge=1,
        description="Maximum processing queue size",
    )


class MetricsConfig(BaseModel):
    """Prometheus metrics configuration."""

    enabled: bool = Field(default=False, description="Enable Prometheus metrics")
    port: int = Field(default=9090, ge=1, le=65535, description="Metrics port")
    system_metrics_interval: int = Field(
        default=30,
        ge=1,
        description="System metrics collection interval",
    )
    processing_metrics_interval: int = Field(
        default=10,
        ge=1,
        description="Processing metrics collection interval",
    )


class AppConfig(BaseSettings):
    """Main application configuration."""

    model_config = SettingsConfigDict(
        env_prefix="",
        env_nested_delimiter="__",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra environment variables that don't match model fields
    )

    directories: DirectoriesConfig = Field(default_factory=DirectoriesConfig)
    monitoring: FileMonitoringConfig = Field(default_factory=FileMonitoringConfig)
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    post_processing: PostProcessingConfig = Field(default_factory=PostProcessingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    health_check: HealthCheckConfig = Field(default_factory=HealthCheckConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)


def expand_environment_variables(value: str | dict | list) -> str | dict | list:
    """Recursively expand environment variables in configuration values.

    Supports ${VAR_NAME} and ${VAR_NAME:-default_value} syntax.
    """
    if isinstance(value, str):
        # Pattern to match ${VAR_NAME} or ${VAR_NAME:-default}
        pattern = r"\$\{([^}:]+)(?::-(.*?))?\}"

        def replace_var(match: re.Match) -> str:
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else ""
            return os.getenv(var_name, default_value)

        return re.sub(pattern, replace_var, value)
    if isinstance(value, dict):
        return {k: expand_environment_variables(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_environment_variables(item) for item in value]
    return value


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load configuration from YAML file and environment variables.

    Args:
    ----
        config_path: Path to configuration file. If None, looks for CONFIG_PATH
            environment variable or uses default config.yaml

    Returns:
    -------
        AppConfig: Loaded and validated configuration

    Raises:
    ------
        FileNotFoundError: If configuration file is not found
        yaml.YAMLError: If configuration file is invalid YAML
        ValueError: If configuration validation fails

    """
    # Determine config file path
    if config_path is None:
        config_path = os.getenv("CONFIG_PATH", "config.yaml")

    config_path = Path(config_path)

    # Load YAML configuration if file exists
    config_data = {}
    if config_path.exists():
        try:
            with config_path.open(encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            msg = f"Invalid YAML in configuration file {config_path}: {e}"
            raise yaml.YAMLError(msg) from e

    # Expand environment variables in loaded data
    config_data = expand_environment_variables(config_data)

    # Create configuration object with environment variable overrides
    try:
        config = AppConfig(**config_data)
    except (TypeError, ValueError) as e:
        msg = f"Configuration validation failed: {e}"
        raise ValueError(msg) from e

    return config


def create_directories(config: AppConfig) -> None:
    """Create necessary directories if they don't exist.

    Args:
    ----
        config: Application configuration

    Raises:
    ------
        PermissionError: If unable to create directories due to permissions
        OSError: If unable to create directories due to other OS errors

    """
    directories_to_create = [
        config.directories.input,
        config.directories.output,
        config.directories.backup,
        config.directories.temp,
    ]

    for directory in directories_to_create:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as e:
            msg = f"Failed to create directory {directory}: {e}"
            raise OSError(msg) from e


def validate_config(config: AppConfig) -> list[str]:
    """Validate configuration for common issues and return list of warnings.

    Args:
    ----
        config: Application configuration to validate

    Returns:
    -------
        List[str]: List of validation warnings

    """
    warnings = []

    # Check HuggingFace token if diarization is enabled
    if config.diarization.enabled and not config.diarization.hf_token:
        warnings.append(
            "Diarization is enabled but HF_TOKEN is not set. "
            "Speaker diarization will fail without a valid HuggingFace token.",
        )

    # Check directory permissions
    for name, path in [
        ("input", config.directories.input),
        ("output", config.directories.output),
        ("backup", config.directories.backup),
        ("temp", config.directories.temp),
    ]:
        if path.exists() and not os.access(path, os.R_OK | os.W_OK):
            warnings.append(
                f"{name.capitalize()} directory {path} is not readable/writable",
            )

    # Check memory settings
    min_memory_gb = 2.0
    if config.performance.max_memory_usage_gb < min_memory_gb:
        warnings.append(
            "max_memory_usage_gb is set below 2GB, this may cause processing failures",
        )

    # Check concurrent processing settings
    max_concurrent_files = 4
    if config.performance.max_concurrent_files > max_concurrent_files:
        warnings.append(
            "max_concurrent_files is set above 4, this may cause resource exhaustion",
        )

    return warnings
