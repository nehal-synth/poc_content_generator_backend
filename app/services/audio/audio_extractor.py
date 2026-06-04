import subprocess
from pathlib import Path

from app.services.media.ffmpeg_check import FFMPEG_INSTALL_HINT, ensure_ffmpeg_available


class AudioExtractionError(Exception):
    pass


def extract_audio(video_path: Path, output_path: Path | None = None) -> Path:
    ensure_ffmpeg_available()
    if output_path is None:
        output_path = video_path.with_suffix(".mp3")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "libmp3lame",
        "-q:a",
        "4",
        str(output_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise AudioExtractionError(f"ffmpeg not found. {FFMPEG_INSTALL_HINT}") from exc
    if result.returncode != 0:
        raise AudioExtractionError(result.stderr or "ffmpeg failed to extract audio")
    return output_path
