"""Unit tests for file handling utilities."""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from src.utils.error_handling import FileSystemError
from src.utils.file_handler import FileHandler

# Expected test values
EXPECTED_MEDIA_FILES_COUNT = 2
EXPECTED_RECURSIVE_FILES_COUNT = 3
EXPECTED_FILE_SIZE_BYTES = 10


class TestFileHandler:
    """Test the FileHandler class."""

    def setup_method(self) -> None:
        """Set up a temporary directory for testing."""
        self.test_dir = tempfile.mkdtemp()
        self.handler = FileHandler()

    def teardown_method(self) -> None:
        """Clean up the temporary directory."""
        shutil.rmtree(self.test_dir)

    def create_test_file(self, name: str, content: str = "test") -> Path:
        """Helper to create a test file."""
        path = Path(self.test_dir) / name
        path.write_text(content)
        return path

    def test_is_media_file(self) -> None:
        """Test media file detection."""
        assert self.handler.is_media_file("test.mp3") is True
        assert self.handler.is_media_file("test.mp4") is True
        assert self.handler.is_media_file("test.txt") is False
        assert self.handler.is_audio_file("test.wav") is True
        assert self.handler.is_video_file("test.mov") is True
        assert self.handler.is_audio_file("test.mov") is False

    def test_get_media_files(self) -> None:
        """Test getting media files from a directory."""
        self.create_test_file("audio.mp3")
        self.create_test_file("video.mkv")
        self.create_test_file("document.txt")

        # Test non-recursive
        files = self.handler.get_media_files(self.test_dir)
        assert len(files) == EXPECTED_MEDIA_FILES_COUNT
        assert Path(self.test_dir) / "audio.mp3" in files
        assert Path(self.test_dir) / "video.mkv" in files

        # Test recursive
        sub_dir = Path(self.test_dir) / "subdir"
        sub_dir.mkdir()
        (sub_dir / "sub_audio.flac").write_text("test")
        files = self.handler.get_media_files(self.test_dir, include_subdirectories=True)
        assert len(files) == EXPECTED_RECURSIVE_FILES_COUNT
        assert sub_dir / "sub_audio.flac" in files

    def test_validate_file_path(self) -> None:
        """Test file path validation."""
        test_file = self.create_test_file("valid.txt")

        # Valid file
        assert self.handler.validate_file_path(test_file) == test_file.resolve()

        # Non-existent file
        with pytest.raises(FileSystemError, match="File not found"):
            self.handler.validate_file_path("nonexistent.txt")

        # Path is a directory
        with pytest.raises(FileSystemError, match="Path is not a file"):
            self.handler.validate_file_path(self.test_dir)

    def test_ensure_directory(self) -> None:
        """Test directory creation and validation."""
        new_dir = Path(self.test_dir) / "new_dir"

        # Create new directory
        self.handler.ensure_directory(new_dir)
        assert new_dir.exists()
        assert new_dir.is_dir()

        # Ensure existing directory
        self.handler.ensure_directory(new_dir)
        assert new_dir.exists()

    def test_safe_move_file(self) -> None:
        """Test safe file moving."""
        src_file = self.create_test_file("source.txt")
        dest_dir = Path(self.test_dir) / "dest"
        dest_file = dest_dir / "source.txt"

        self.handler.safe_move_file(src_file, dest_file)

        assert not src_file.exists()
        assert dest_file.exists()

    def test_safe_move_with_conflict(self) -> None:
        """Test safe file moving with a filename conflict."""
        src_file = self.create_test_file("source.txt")
        dest_dir = Path(self.test_dir) / "dest"
        dest_dir.mkdir()
        # Create a conflicting file at the destination
        (dest_dir / "source.txt").write_text("conflict")

        final_path = self.handler.safe_move_file(src_file, dest_dir / "source.txt")

        assert not src_file.exists()
        assert final_path.name == "source_1.txt"
        assert final_path.exists()

    def test_safe_copy_file(self) -> None:
        """Test safe file copying."""
        src_file = self.create_test_file("source.txt")
        dest_dir = Path(self.test_dir) / "dest"
        dest_file = dest_dir / "source.txt"

        self.handler.safe_copy_file(src_file, dest_file)

        assert src_file.exists()
        assert dest_file.exists()
        assert src_file.read_text() == dest_file.read_text()

    def test_safe_delete_file(self) -> None:
        """Test safe file deletion."""
        file_to_delete = self.create_test_file("deleteme.txt")

        assert self.handler.safe_delete_file(file_to_delete) is True
        assert not file_to_delete.exists()

        # Deleting non-existent file
        assert self.handler.safe_delete_file("nonexistent.txt") is False

    def test_get_file_info(self) -> None:
        """Test getting file information."""
        test_file = self.create_test_file("info.mp3", content="audio data")
        info = self.handler.get_file_info(test_file)

        assert info["name"] == "info.mp3"
        assert info["size_bytes"] == EXPECTED_FILE_SIZE_BYTES
        assert info["is_audio"] is True
        assert info["is_video"] is False

    def test_write_and_read_text_file(self) -> None:
        """Test writing and reading a text file."""
        file_path = Path(self.test_dir) / "text_file.txt"
        content = "Hello, this is a test."

        self.handler.write_text_file(file_path, content)
        read_content = self.handler.read_text_file(file_path)

        assert content == read_content
