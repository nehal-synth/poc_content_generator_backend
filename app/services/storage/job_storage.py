import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings
from app.models.schemas import JobRecord, JobStatus, JobSummary


def _jobs_meta_dir() -> Path:
    path = settings.jobs_dir
    path.mkdir(parents=True, exist_ok=True)
    return path


def _job_dir(job_id: str) -> Path:
    path = settings.upload_dir / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _job_meta_path(job_id: str) -> Path:
    return _jobs_meta_dir() / f"{job_id}.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_job(filename: str, content_type: str, size_bytes: int, is_video: bool) -> JobRecord:
    job_id = str(uuid.uuid4())
    now = _now_iso()
    job = JobRecord(
        id=job_id,
        status=JobStatus.UPLOADED,
        status_message="File uploaded successfully",
        file={
            "filename": filename,
            "content_type": content_type,
            "size_bytes": size_bytes,
            "is_video": is_video,
        },
        created_at=now,
        updated_at=now,
    )
    save_job(job)
    _job_dir(job_id)
    return job


def save_job(job: JobRecord) -> None:
    job.updated_at = _now_iso()
    _job_meta_path(job.id).write_text(job.model_dump_json(indent=2), encoding="utf-8")


def get_job(job_id: str) -> JobRecord | None:
    path = _job_meta_path(job_id)
    if not path.exists():
        return None
    return JobRecord.model_validate_json(path.read_text(encoding="utf-8"))


def list_jobs(limit: int = 50) -> list[JobSummary]:
    meta_dir = _jobs_meta_dir()
    all_summaries: list[JobSummary] = []
    for path in meta_dir.glob("*.json"):
        try:
            job = JobRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        all_summaries.append(
            JobSummary(
                id=job.id,
                status=job.status,
                filename=job.file.filename,
                pillar=job.classification.pillar if job.classification else None,
                format=job.classification.format if job.classification else None,
                created_at=job.created_at,
            )
        )

    # Sort by the created_at field stored inside the JSON (stable, accurate order).
    # st_mtime changes on every pipeline write, so it cannot be trusted for ordering.
    all_summaries.sort(key=lambda j: j.created_at, reverse=True)
    return all_summaries[:limit]


def save_upload_file(job_id: str, filename: str, data: bytes) -> Path:
    dest = _job_dir(job_id) / filename
    dest.write_bytes(data)
    return dest


def get_job_file_path(job_id: str, filename: str) -> Path | None:
    path = _job_dir(job_id) / filename
    return path if path.exists() else None


def update_job_status(job: JobRecord, status: JobStatus, message: str = "") -> JobRecord:
    job.status = status
    job.status_message = message
    save_job(job)
    return job


def delete_job(job_id: str) -> bool:
    meta = _job_meta_path(job_id)
    if not meta.exists():
        return False
    meta.unlink(missing_ok=True)
    job_dir = settings.upload_dir / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)
    return True
