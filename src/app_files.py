from __future__ import annotations

from pathlib import Path


def safe_filename(name: str) -> str:
    """Sanitize a filename to reduce filesystem issues."""
    return "".join(c for c in name if c.isalnum() or c in ("-", "_", ".", " ")).strip()


def save_bytes_to_file(data: bytes | memoryview, dest_path: Path) -> Path:
    """Save uploaded bytes to disk."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, memoryview):
        data = data.tobytes()
    dest_path.write_bytes(data)
    return dest_path


def validate_extension(file_path: Path, supported_ext: set[str]) -> bool:
    """Check whether a file extension is supported."""
    return file_path.suffix.lower() in supported_ext


def clear_workspace(work_dir: Path) -> None:
    """Best-effort cleanup of the app workspace."""
    if not work_dir.exists():
        return

    for path in sorted(work_dir.rglob("*"), reverse=True):
        try:
            if path.is_file():
                path.unlink()
            else:
                path.rmdir()
        except OSError:
            pass


def read_text_file(path: Path) -> str:
    """Read a UTF-8 text file from disk."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path.read_text(encoding="utf-8")
