"""File handling utilities for audio and video processing.

This module provides comprehensive file management capabilities including:
- Media file type detection and validation
- Path operations and directory management
- File system operations with error handling
- Audio/video file format support

Based on patterns from Whisper-WebUI reference implementation.
"""

import fnmatch
import logging
import os
import shutil
from pathlib import Path
from typing import Any, List, Optional

from src.utils.error_handling import FileSystemError


class FileHandler:
    """Comprehensive file handling utility for media processing operations.

    Provides file type detection, path management, and file system operations
    with proper error handling and logging.
    """

    # Supported audio file extensions
    AUDIO_EXTENSIONS = [
        ".mp3",
        ".wav",
        ".wma",
        ".aac",
        ".flac",
        ".ogg",
        ".m4a",
        ".aiff",
        ".alac",
        ".opus",
        ".webm",
        ".ac3",
        ".amr",
        ".au",
        ".mid",
        ".midi",
        ".mka",
    ]

    # Supported video file extensions
    VIDEO_EXTENSIONS = [
        ".mp4",
        ".mkv",
        ".flv",
        ".avi",
        ".mov",
        ".wmv",
        ".webm",
        ".m4v",
        ".mpeg",
        ".mpg",
        ".3gp",
        ".f4v",
        ".ogv",
        ".vob",
        ".mts",
        ".m2ts",
        ".divx",
        ".mxf",
        ".rm",
        ".rmvb",
        ".ts",
    ]

    # All supported media extensions
    MEDIA_EXTENSIONS = AUDIO_EXTENSIONS + VIDEO_EXTENSIONS

    def __init__(self) -> None:
        """Initialize FileHandler with logging."""
        self.logger = logging.getLogger(__name__)

    @classmethod
    def is_audio_file(cls, file_path: Path | str) -> bool:
        """Check if file is a supported audio format.

        Args:
            file_path: Path to the file to check

        Returns:
            True if file is a supported audio format
        """
        extension = Path(file_path).suffix.lower()
        return extension in cls.AUDIO_EXTENSIONS

    @classmethod
    def is_video_file(cls, file_path: Path | str) -> bool:
        """Check if file is a supported video format.

        Args:
            file_path: Path to the file to check

        Returns:
            True if file is a supported video format
        """
        extension = Path(file_path).suffix.lower()
        return extension in cls.VIDEO_EXTENSIONS

    @classmethod
    def is_media_file(cls, file_path: Path | str) -> bool:
        """Check if file is a supported media format (audio or video).

        Args:
            file_path: Path to the file to check

        Returns:
            True if file is a supported media format
        """
        extension = Path(file_path).suffix.lower()
        return extension in cls.MEDIA_EXTENSIONS

    @classmethod
    def get_media_files(
        cls,
        folder_path: Path | str,
        include_subdirectories: bool = False,
        extensions: Optional[List[str]] = None,
    ) -> List[Path]:
        """Get all media files from a directory.

        Args:
            folder_path: Directory to search
            include_subdirectories: Whether to search subdirectories recursively
            extensions: Optional list of specific extensions to search for

        Returns:
            List of Path objects for found media files
        """
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            return []

        # Use provided extensions or default media extensions
        search_extensions = extensions or cls.MEDIA_EXTENSIONS
        media_patterns = [f"*{ext}" for ext in search_extensions]
        media_files = []

        try:
            if include_subdirectories:
                # Recursive search using os.walk for better performance
                for root, _, files in os.walk(folder):
                    root_path = Path(root)
                    for pattern in media_patterns:
                        for file in fnmatch.filter(files, pattern):
                            file_path = root_path / file
                            if file_path.exists() and file_path.is_file():
                                media_files.append(file_path)
            else:
                # Search only in the specified directory
                for pattern in media_patterns:
                    for file_path in folder.glob(pattern):
                        if file_path.is_file():
                            media_files.append(file_path)

        except (OSError, PermissionError) as e:
            logging.getLogger(__name__).warning(
                "Error accessing directory %s: %s", folder, e
            )

        return sorted(media_files)  # Sort for consistent ordering

    def validate_file_path(self, file_path: Path | str) -> Path:
        """Validate and normalize file path.

        Args:
            file_path: Path to validate

        Returns:
            Validated Path object

        Raises:
            FileSystemError: If file doesn't exist or is not accessible
        """
        path = Path(file_path)

        if not path.exists():
            raise FileSystemError(f"File not found: {path}")

        if not path.is_file():
            raise FileSystemError(f"Path is not a file: {path}")

        if not os.access(path, os.R_OK):
            raise FileSystemError(f"File is not readable: {path}")

        return path.resolve()

    def ensure_directory(self, directory: Path | str) -> Path:
        """Ensure directory exists, creating it if necessary.

        Args:
            directory: Directory path to ensure

        Returns:
            Path object for the directory

        Raises:
            FileSystemError: If directory cannot be created or accessed
        """
        dir_path = Path(directory)

        try:
            dir_path.mkdir(parents=True, exist_ok=True)

            # Verify directory is writable
            if not os.access(dir_path, os.W_OK):
                raise FileSystemError(f"Directory is not writable: {dir_path}")

            return dir_path.resolve()

        except OSError as e:
            raise FileSystemError(f"Cannot create directory {dir_path}: {e}") from e

    def safe_move_file(self, source: Path | str, destination: Path | str) -> Path:
        """Safely move a file, handling conflicts and errors.

        Args:
            source: Source file path
            destination: Destination file path

        Returns:
            Path object for the final destination

        Raises:
            FileSystemError: If move operation fails
        """
        src_path = self.validate_file_path(source)
        dest_path = Path(destination)

        # Ensure destination directory exists
        self.ensure_directory(dest_path.parent)

        # Handle filename conflicts
        final_dest = self._resolve_filename_conflict(dest_path)

        try:
            shutil.move(str(src_path), str(final_dest))
            self.logger.info("Moved file: %s -> %s", src_path, final_dest)
            return final_dest

        except OSError as e:
            raise FileSystemError(
                f"Cannot move file {src_path} to {final_dest}: {e}"
            ) from e

    def safe_copy_file(self, source: Path | str, destination: Path | str) -> Path:
        """Safely copy a file, handling conflicts and errors.

        Args:
            source: Source file path
            destination: Destination file path

        Returns:
            Path object for the final destination

        Raises:
            FileSystemError: If copy operation fails
        """
        src_path = self.validate_file_path(source)
        dest_path = Path(destination)

        # Ensure destination directory exists
        self.ensure_directory(dest_path.parent)

        # Handle filename conflicts
        final_dest = self._resolve_filename_conflict(dest_path)

        try:
            shutil.copy2(str(src_path), str(final_dest))
            self.logger.info("Copied file: %s -> %s", src_path, final_dest)
            return final_dest

        except OSError as e:
            raise FileSystemError(
                f"Cannot copy file {src_path} to {final_dest}: {e}"
            ) from e

    def safe_delete_file(self, file_path: Path | str) -> bool:
        """Safely delete a file.

        Args:
            file_path: Path to file to delete

        Returns:
            True if file was deleted, False if it didn't exist

        Raises:
            FileSystemError: If deletion fails
        """
        path = Path(file_path)

        if not path.exists():
            return False

        if not path.is_file():
            raise FileSystemError(f"Path is not a file: {path}")

        try:
            path.unlink()
            self.logger.info("Deleted file: %s", path)
            return True

        except OSError as e:
            raise FileSystemError(f"Cannot delete file {path}: {e}") from e

    def get_file_info(self, file_path: Path | str) -> dict[str, Any]:
        """Get comprehensive file information.

        Args:
            file_path: Path to analyze

        Returns:
            Dictionary with file information

        Raises:
            FileSystemError: If file cannot be accessed
        """
        path = self.validate_file_path(file_path)

        try:
            stat = path.stat()

            return {
                "path": str(path),
                "name": path.name,
                "stem": path.stem,
                "suffix": path.suffix,
                "size_bytes": stat.st_size,
                "size_mb": stat.st_size / (1024 * 1024),
                "modified_time": stat.st_mtime,
                "is_audio": self.is_audio_file(path),
                "is_video": self.is_video_file(path),
                "is_media": self.is_media_file(path),
                "readable": os.access(path, os.R_OK),
                "writable": os.access(path, os.W_OK),
            }

        except OSError as e:
            raise FileSystemError(f"Cannot get file info for {path}: {e}") from e

    def read_text_file(self, file_path: Path | str, encoding: str = "utf-8") -> str:
        """Read text content from a file.

        Args:
            file_path: Path to text file
            encoding: Text encoding to use

        Returns:
            File content as string

        Raises:
            FileSystemError: If file cannot be read
        """
        path = self.validate_file_path(file_path)

        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read()

        except (OSError, UnicodeDecodeError) as e:
            raise FileSystemError(f"Cannot read text file {path}: {e}") from e

    def write_text_file(
        self,
        file_path: Path | str,
        content: str,
        encoding: str = "utf-8",
        create_dirs: bool = True,
    ) -> Path:
        """Write text content to a file.

        Args:
            file_path: Path to write to
            content: Text content to write
            encoding: Text encoding to use
            create_dirs: Whether to create parent directories

        Returns:
            Path object for the written file

        Raises:
            FileSystemError: If file cannot be written
        """
        path = Path(file_path)

        if create_dirs:
            self.ensure_directory(path.parent)

        try:
            with open(path, "w", encoding=encoding) as f:
                f.write(content)

            self.logger.debug("Wrote text file: %s (%d chars)", path, len(content))
            return path.resolve()

        except (OSError, UnicodeEncodeError) as e:
            raise FileSystemError(f"Cannot write text file {path}: {e}") from e

    def _resolve_filename_conflict(self, file_path: Path) -> Path:
        """Resolve filename conflicts by adding a counter.

        Args:
            file_path: Original file path

        Returns:
            Path that doesn't conflict with existing files
        """
        if not file_path.exists():
            return file_path

        counter = 1
        original_stem = file_path.stem
        suffix = file_path.suffix
        parent = file_path.parent

        while True:
            new_name = f"{original_stem}_{counter}{suffix}"
            new_path = parent / new_name

            if not new_path.exists():
                return new_path

            counter += 1

            # Prevent infinite loops
            if counter > 9999:
                raise FileSystemError(
                    f"Cannot resolve filename conflict for {file_path}"
                )

    @classmethod
    def get_supported_extensions(cls) -> dict[str, List[str]]:
        """Get all supported file extensions by category.

        Returns:
            Dictionary with extension categories
        """
        return {
            "audio": cls.AUDIO_EXTENSIONS,
            "video": cls.VIDEO_EXTENSIONS,
            "media": cls.MEDIA_EXTENSIONS,
        }

    def cleanup_temp_files(self, temp_dir: Path | str, pattern: str = "*") -> int:
        """Clean up temporary files in a directory.

        Args:
            temp_dir: Directory to clean
            pattern: File pattern to match (default: all files)

        Returns:
            Number of files deleted
        """
        temp_path = Path(temp_dir)
        if not temp_path.exists() or not temp_path.is_dir():
            return 0

        deleted_count = 0

        try:
            for file_path in temp_path.glob(pattern):
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        deleted_count += 1
                    except OSError:
                        self.logger.warning("Could not delete temp file: %s", file_path)

            if deleted_count > 0:
                self.logger.info(
                    "Cleaned up %d temporary files from %s", deleted_count, temp_path
                )

        except OSError:
            self.logger.warning("Error accessing temp directory: %s", temp_path)

        return deleted_count
