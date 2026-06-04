from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from app.config import settings
from app.models.schemas import HealthResponse, JobListResponse, JobRecord, JobSummary, UploadResponse
from app.services.export.export_service import export_docx, export_pdf
from app.services.media.ffmpeg_check import ffmpeg_unavailable_message
from app.services.pipeline.job_processor import process_job
from app.services.storage.job_storage import create_job, delete_job, get_job, list_jobs, save_job, save_upload_file

router = APIRouter(prefix="/api")


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        openai_configured=settings.openai_configured,
    )


@router.get("/jobs", response_model=JobListResponse)
async def get_jobs(limit: int = 50) -> JobListResponse:
    jobs = list_jobs(limit=limit)
    return JobListResponse(jobs=jobs, total=len(jobs))


@router.get("/jobs/{job_id}", response_model=JobRecord)
async def get_job_by_id(job_id: str) -> JobRecord:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.delete("/jobs/{job_id}")
async def remove_job(job_id: str) -> dict[str, str]:
    if not delete_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    return {"message": "Job deleted"}


@router.post("/jobs/upload", response_model=UploadResponse)
async def upload_content(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    companion_file: UploadFile | None = File(None),
    notes: str = Form(""),
) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    if not settings.openai_configured:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not configured. Add it to backend/.env before uploading.",
        )

    suffix = file.filename.rsplit(".", 1)[-1].lower()
    allowed = {"mp4", "mov", "avi", "pdf", "docx", "doc", "txt"}
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: .{suffix}")

    data = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_size_mb}MB limit")

    is_video = suffix in {"mp4", "mov", "avi"}
    if is_video:
        ffmpeg_msg = ffmpeg_unavailable_message()
        if ffmpeg_msg:
            raise HTTPException(status_code=503, detail=ffmpeg_msg)

    job = create_job(
        filename=file.filename,
        content_type=file.content_type or "application/octet-stream",
        size_bytes=len(data),
        is_video=is_video,
    )
    save_upload_file(job.id, file.filename, data)

    companion_text = notes.strip()
    if companion_file and companion_file.filename:
        companion_data = await companion_file.read()
        companion_path = save_upload_file(job.id, companion_file.filename, companion_data)
        try:
            from app.services.document.document_parser import extract_text_from_document

            companion_text = (
                f"{companion_text}\n\n{extract_text_from_document(companion_path)}".strip()
                if companion_text
                else extract_text_from_document(companion_path)
            )
        except Exception:
            pass

    if companion_text:
        job.companion_text = companion_text
        save_job(job)

    background_tasks.add_task(process_job, job.id)

    return UploadResponse(
        job_id=job.id,
        status=job.status,
        message="Upload received. Processing started.",
    )


@router.get("/jobs/{job_id}/export")
async def export_job(job_id: str, format: str = "pdf") -> Response:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Job is not completed yet")

    base_name = job.file.filename.rsplit(".", 1)[0]

    if format == "docx":
        content = export_docx(job)
        return Response(
            content=content,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f'attachment; filename="{base_name}-content-kit.docx"'},
        )
    if format == "pdf":
        content = export_pdf(job)
        return Response(
            content=content,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{base_name}-content-kit.pdf"'},
        )
    raise HTTPException(status_code=400, detail="Format must be pdf or docx")
