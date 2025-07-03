"""Unit tests for configuration management functionality."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from src.config import (
    AppConfig,
    DiarizationConfig,
    DirectoriesConfig,
    WhisperConfig,
    create_directories,
    expand_environment_variables,
    load_config,
    validate_config,
)

# Expected default values for configuration tests
DEFAULT_MIN_SPEAKERS = 1
DEFAULT_MAX_SPEAKERS = 10
DEFAULT_MAX_CONCURRENT_FILES = 2
DEFAULT_MAX_CONCURRENT_FILES_HIGH_PERFORMANCE = 4


class TestExpandEnvironmentVariables:
    """Test environment variable expansion functionality."""

    def test_simple_variable_expansion(self) -> None:
        """Test simple ${VAR} expansion."""
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            result = expand_environment_variables("${TEST_VAR}")
            assert result == "test_value"

    def test_variable_with_default(self) -> None:
        """Test ${VAR:-default} expansion."""
        result = expand_environment_variables("${NONEXISTENT_VAR:-default_value}")
        assert result == "default_value"

    def test_nested_dict_expansion(self) -> None:
        """Test expansion in nested dictionaries."""
        with patch.dict(os.environ, {"DB_HOST": "localhost", "DB_PORT": "5432"}):
            data = {
                "database": {
                    "host": "${DB_HOST}",
                    "port": "${DB_PORT:-3306}",
                },
            }
            result = expand_environment_variables(data)
            assert result["database"]["host"] == "localhost"
            assert result["database"]["port"] == "5432"

    def test_list_expansion(self) -> None:
        """Test expansion in lists."""
        with patch.dict(os.environ, {"ITEM1": "first", "ITEM2": "second"}):
            data = ["${ITEM1}", "${ITEM2}", "static"]
            result = expand_environment_variables(data)
            assert result == ["first", "second", "static"]


class TestDirectoriesConfig:
    """Test directories configuration."""

    def test_default_directories(self) -> None:
        """Test default directory configuration."""
        config = DirectoriesConfig()
        assert config.input == Path("/data/in")
        assert config.output == Path("/data/out")
        assert config.backup == Path("/data/backup")
        assert config.temp == Path("/tmp/transcribe")  # noqa: S108

    def test_custom_directories(self) -> None:
        """Test custom directory configuration."""
        config = DirectoriesConfig(
            input=Path("/custom/input"),
            output=Path("/custom/output"),
        )
        assert config.input == Path("/custom/input")
        assert config.output == Path("/custom/output")


class TestWhisperConfig:
    """Test Whisper configuration validation."""

    def test_valid_model_sizes(self) -> None:
        """Test valid model size validation."""
        valid_sizes = [
            "tiny",
            "base",
            "small",
            "medium",
            "large-v1",
            "large-v2",
            "large-v3",
        ]
        for size in valid_sizes:
            config = WhisperConfig(model_size=size)
            assert config.model_size == size

    def test_invalid_model_size(self) -> None:
        """Test invalid model size raises error."""
        with pytest.raises(ValueError, match="model_size must be one of"):
            WhisperConfig(model_size="invalid")

    def test_valid_devices(self) -> None:
        """Test valid device validation."""
        valid_devices = ["auto", "cpu", "cuda"]
        for device in valid_devices:
            config = WhisperConfig(device=device)
            assert config.device == device

    def test_invalid_device(self) -> None:
        """Test invalid device raises error."""
        with pytest.raises(ValueError, match="device must be one of"):
            WhisperConfig(device="invalid")


class TestDiarizationConfig:
    """Test diarization configuration."""

    def test_default_config(self) -> None:
        """Test default diarization configuration."""
        config = DiarizationConfig()
        assert config.enabled is True
        assert config.min_speakers == DEFAULT_MIN_SPEAKERS
        assert config.max_speakers == DEFAULT_MAX_SPEAKERS
        assert config.model == "pyannote/speaker-diarization-3.1"

    def test_with_token(self) -> None:
        """Test configuration with HuggingFace token."""
        config = DiarizationConfig(hf_token="test_token")  # noqa: S106
        assert config.hf_token == "test_token"  # noqa: S105


class TestLoadConfig:
    """Test configuration loading functionality."""

    def test_load_from_yaml_file(self) -> None:
        """Test loading configuration from YAML file."""
        config_data = {
            "transcription": {
                "whisper": {
                    "model_size": "small",
                },
            },
            "diarization": {
                "enabled": False,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = load_config(temp_path)
            assert config.transcription.whisper.model_size == "small"
            assert config.diarization.enabled is False
        finally:
            Path(temp_path).unlink()

    def test_load_nonexistent_file(self) -> None:
        """Test loading with nonexistent file falls back to defaults."""
        config = load_config("nonexistent_file.yaml")
        assert isinstance(config, AppConfig)
        assert config.transcription.whisper.model_size == "medium"  # Default value

    def test_environment_variable_override(self) -> None:
        """Test environment variable overrides."""
        with patch.dict(
            os.environ,
            {
                "TRANSCRIPTION__WHISPER__MODEL_SIZE": "large-v3",
                "DIARIZATION__ENABLED": "false",
            },
        ):
            config = load_config(None)
            assert config.transcription.whisper.model_size == "large-v3"
            assert config.diarization.enabled is False

    def test_invalid_yaml(self) -> None:
        """Test loading invalid YAML raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: }")
            temp_path = f.name

        try:
            with pytest.raises(yaml.YAMLError):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()


class TestCreateDirectories:
    """Test directory creation functionality."""

    def test_create_directories_success(self) -> None:
        """Test successful directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = AppConfig(
                directories=DirectoriesConfig(
                    input=temp_path / "input",
                    output=temp_path / "output",
                    backup=temp_path / "backup",
                    temp=temp_path / "temp",
                ),
            )

            create_directories(config)

            assert (temp_path / "input").exists()
            assert (temp_path / "output").exists()
            assert (temp_path / "backup").exists()
            assert (temp_path / "temp").exists()

    def test_create_directories_existing(self) -> None:
        """Test directory creation with existing directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            # Pre-create one directory
            (temp_path / "input").mkdir()

            config = AppConfig(
                directories=DirectoriesConfig(
                    input=temp_path / "input",
                    output=temp_path / "output",
                ),
            )

            # Should not raise error for existing directory
            create_directories(config)

            assert (temp_path / "input").exists()
            assert (temp_path / "output").exists()


class TestValidateConfig:
    """Test configuration validation."""

    def test_valid_config(self) -> None:
        """Test validation of valid configuration."""
        config = AppConfig()
        warnings = validate_config(config)
        # Should have warning about missing HF_TOKEN
        assert len(warnings) >= 1
        assert any("HF_TOKEN" in warning for warning in warnings)

    def test_disabled_diarization(self) -> None:
        """Test validation with disabled diarization."""
        config = AppConfig(diarization=DiarizationConfig(enabled=False))
        warnings = validate_config(config)
        # Should not have HF_TOKEN warning when diarization is disabled
        assert not any("HF_TOKEN" in warning for warning in warnings)

    def test_low_memory_warning(self) -> None:
        """Test warning for low memory configuration."""
        from src.config import PerformanceConfig

        config = AppConfig(performance=PerformanceConfig(max_memory_usage_gb=1.0))
        warnings = validate_config(config)
        assert any("2GB" in warning for warning in warnings)

    def test_high_concurrent_files_warning(self) -> None:
        """Test warning for high concurrent files."""
        from src.config import PerformanceConfig

        config = AppConfig(performance=PerformanceConfig(max_concurrent_files=5))
        warnings = validate_config(config)
        assert any("resource exhaustion" in warning for warning in warnings)


class TestAppConfig:
    """Test main application configuration."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = AppConfig()
        assert config.transcription.whisper.model_size == "medium"
        assert config.diarization.enabled is True
        assert config.performance.max_concurrent_files == DEFAULT_MAX_CONCURRENT_FILES
        assert config.logging.level == "INFO"

    def test_nested_config_override(self) -> None:
        """Test nested configuration override via environment."""
        with patch.dict(
            os.environ,
            {
                "TRANSCRIPTION__WHISPER__MODEL_SIZE": "large-v3",
                "PERFORMANCE__MAX_CONCURRENT_FILES": "4",
                "LOGGING__LEVEL": "DEBUG",
            },
        ):
            config = AppConfig()
            assert config.transcription.whisper.model_size == "large-v3"
            assert (
                config.performance.max_concurrent_files
                == DEFAULT_MAX_CONCURRENT_FILES_HIGH_PERFORMANCE
            )
            assert config.logging.level == "DEBUG"
