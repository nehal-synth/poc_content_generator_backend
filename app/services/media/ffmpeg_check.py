import shutil

FFMPEG_INSTALL_HINT = (
    "Install ffmpeg (includes ffprobe): "
    "sudo apt install ffmpeg  (Ubuntu/Debian)  or  brew install ffmpeg  (macOS)"
)


def missing_ffmpeg_tools() -> list[str]:
    return [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]


def ffmpeg_unavailable_message() -> str | None:
    missing = missing_ffmpeg_tools()
    if not missing:
        return None
    tools = ", ".join(missing)
    return f"{tools} not found on PATH. {FFMPEG_INSTALL_HINT}"


def ensure_ffmpeg_available() -> None:
    msg = ffmpeg_unavailable_message()
    if msg:
        raise RuntimeError(msg)
