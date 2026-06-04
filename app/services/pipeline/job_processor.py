import asyncio
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

from app.models.schemas import JobStatus, LLMStepUsage, WhisperUsage
from app.services.analysis.content_analyzer import (
    InsufficientContentError,
    analyze_content,
    describe_video_frames,
)
from app.services.audio.audio_extractor import AudioExtractionError, extract_audio
from app.services.document.document_parser import extract_text_from_document
from app.services.generation.content_generator import generate_assets, generate_publishing_recommendation
from app.services.storage.job_storage import get_job, get_job_file_path, save_job, update_job_status
from app.services.strategy.classifier import classify_content
from app.services.transcription.transcript_corrector import align_and_correct_transcript
from app.services.transcription.whisper_service import TranscriptionError, transcribe_raw_audio
from app.services.video.frame_extractor import extract_key_frames

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}
DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}

# ---------------------------------------------------------------------------
# GPT-4o pricing constants (per token)
# ---------------------------------------------------------------------------
_GPT4O_INPUT_RATE = 0.0000025     # $2.50 / 1M tokens  — non-cached input
_GPT4O_CACHED_RATE = 0.00000125   # $1.25 / 1M tokens  — cached input
_GPT4O_OUTPUT_RATE = 0.0000100    # $10.00 / 1M tokens — output

# Whisper pricing
_WHISPER_RATE_PER_SECOND = 0.0001  # $0.006 / min ≡ $0.0001 / sec


def _is_video(filename: str) -> bool:
    return Path(filename).suffix.lower() in VIDEO_EXTENSIONS


def _is_document(filename: str) -> bool:
    return Path(filename).suffix.lower() in DOCUMENT_EXTENSIONS


def _record_metric(job, key: str, elapsed: float) -> None:
    """Store a rounded timing metric and immediately persist to disk."""
    job.profiling_metrics[key] = round(elapsed, 4)
    save_job(job)


def _calc_llm_cost(usage: dict) -> float:
    """Return total GPT-4o cost for a single API call given a usage dict."""
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    cached_tokens = usage.get("cached_tokens", 0)
    non_cached = max(prompt_tokens - cached_tokens, 0)
    return (
        non_cached * _GPT4O_INPUT_RATE
        + cached_tokens * _GPT4O_CACHED_RATE
        + completion_tokens * _GPT4O_OUTPUT_RATE
    )


def _llm_step_usage(usage: dict) -> LLMStepUsage:
    """Build a LLMStepUsage model from a raw usage dict and calculate cost."""
    return LLMStepUsage(
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        cached_tokens=usage.get("cached_tokens", 0),
        cost=round(_calc_llm_cost(usage), 8),
    )


def _merge_llm_usage(a: dict, b: dict) -> dict:
    """Combine two usage dicts by summing each token field."""
    return {
        "prompt_tokens": a.get("prompt_tokens", 0) + b.get("prompt_tokens", 0),
        "completion_tokens": a.get("completion_tokens", 0) + b.get("completion_tokens", 0),
        "cached_tokens": a.get("cached_tokens", 0) + b.get("cached_tokens", 0),
    }


async def process_job(job_id: str) -> None:
    pipeline_start = time.perf_counter()
    wall_start = time.time()

    job = get_job(job_id)
    if not job:
        return

    file_path = get_job_file_path(job_id, job.file.filename)
    if not file_path:
        job.status = JobStatus.FAILED
        job.error = "Uploaded file not found"
        save_job(job)
        return

    try:
        source_text = ""
        visual_description = ""

        if _is_video(job.file.filename):
            # ── Step 1: Extract audio ──────────────────────────────────────────
            update_job_status(job, JobStatus.EXTRACTING_AUDIO, "Extracting audio from video")
            job = get_job(job_id) or job
            audio_path = file_path.with_suffix(".mp3")

            t0 = time.perf_counter()
            try:
                await asyncio.to_thread(extract_audio, file_path, audio_path)
            except AudioExtractionError as exc:
                if "not found" in str(exc).lower():
                    raise
                audio_path = file_path
            finally:
                _record_metric(job, "audio_extraction_time", time.perf_counter() - t0)

            # ── Step 2: Extract frames + visual analysis ───────────────────────
            update_job_status(job, JobStatus.ANALYZING, "Extracting visual context from video frames")
            job = get_job(job_id) or job
            frames_dir = file_path.parent / "frames"

            t0 = time.perf_counter()
            try:
                frames = await asyncio.to_thread(extract_key_frames, file_path, frames_dir)
            except Exception as exc:
                logger.warning(
                    "Frame extraction failed for job %s — skipping visual analysis: %s",
                    job_id, str(exc),
                )
                frames = []
            finally:
                _record_metric(job, "frame_extraction_time", time.perf_counter() - t0)

            t0 = time.perf_counter()
            if frames:
                try:
                    visual_description, vision_usage = await asyncio.to_thread(
                        describe_video_frames, frames
                    )
                    # Record vision_analysis cost
                    job = get_job(job_id) or job
                    job.cost_and_usage_metrics.vision_analysis = _llm_step_usage(vision_usage)
                    job.cost_and_usage_metrics.total_cumulative_cost = round(
                        job.cost_and_usage_metrics.total_cumulative_cost
                        + job.cost_and_usage_metrics.vision_analysis.cost,
                        8,
                    )
                except Exception as exc:
                    logger.warning(
                        "Vision analysis failed for job %s — continuing transcript-only: %s",
                        job_id, str(exc),
                    )
                    visual_description = ""
            else:
                logger.info("No frames extracted for job %s — skipping visual analysis.", job_id)
                visual_description = ""
            _record_metric(job, "vision_analysis_time", time.perf_counter() - t0)

            # ── Step 3: Stage 1 — Raw Whisper transcription ───────────────────
            update_job_status(job, JobStatus.TRANSCRIBING, "Transcribing audio with Whisper (Stage 1)")
            job = get_job(job_id) or job

            t0 = time.perf_counter()
            try:
                raw_transcript, audio_duration = await asyncio.to_thread(
                    transcribe_raw_audio, audio_path
                )
                if not raw_transcript.strip():
                    raise TranscriptionError("Whisper returned an empty transcript.")

                # Record Whisper cost
                whisper_cost = round(audio_duration * _WHISPER_RATE_PER_SECOND, 8)
                job.cost_and_usage_metrics.whisper = WhisperUsage(
                    duration_seconds=round(audio_duration, 3),
                    cost=whisper_cost,
                )
                job.cost_and_usage_metrics.total_cumulative_cost = round(
                    job.cost_and_usage_metrics.total_cumulative_cost + whisper_cost, 8
                )

            except (TranscriptionError, Exception) as exc:
                _record_metric(job, "transcription_time", time.perf_counter() - t0)
                logger.error(
                    "Transcription failed for job %s — aborting pipeline: %s",
                    job_id, str(exc), exc_info=True,
                )
                job = get_job(job_id) or job
                job.status = JobStatus.FAILED
                job.error = f"Audio transcription failed: {exc}"
                job.status_message = "Transcription failed"
                job.processing_seconds = round(time.time() - wall_start, 2)
                job.profiling_metrics["total_pipeline_time"] = round(time.perf_counter() - pipeline_start, 4)
                save_job(job)
                return
            _record_metric(job, "transcription_time", time.perf_counter() - t0)

            # ── Step 4: Stage 2 — LLM phonetic correction ─────────────────────
            update_job_status(job, JobStatus.TRANSCRIBING, "Aligning transcript with visual context (Stage 2)")
            job = get_job(job_id) or job

            t0 = time.perf_counter()
            source_text, correction_usage = await asyncio.to_thread(
                align_and_correct_transcript,
                raw_transcript,
                visual_description,
            )
            # Record transcript_correction cost
            job.cost_and_usage_metrics.transcript_correction = _llm_step_usage(correction_usage)
            job.cost_and_usage_metrics.total_cumulative_cost = round(
                job.cost_and_usage_metrics.total_cumulative_cost
                + job.cost_and_usage_metrics.transcript_correction.cost,
                8,
            )
            _record_metric(job, "correction_time", time.perf_counter() - t0)

            if job.companion_text:
                source_text = f"{source_text}\n\nCompanion notes:\n{job.companion_text}".strip()

        elif _is_document(job.file.filename):
            update_job_status(job, JobStatus.ANALYZING, "Extracting document text")
            job = get_job(job_id) or job
            source_text = await asyncio.to_thread(extract_text_from_document, file_path)
            if job.companion_text:
                source_text = f"{source_text}\n\nCompanion notes:\n{job.companion_text}".strip()
        else:
            raise ValueError(f"Unsupported file type: {job.file.filename}")

        # ── Step 5: Content analysis ───────────────────────────────────────────
        update_job_status(job, JobStatus.ANALYZING, "Analyzing content with AI")
        job = get_job(job_id) or job

        t0 = time.perf_counter()
        analysis, analysis_usage = await asyncio.to_thread(
            analyze_content,
            source_text,
            job.file.filename,
            visual_description,
        )
        job.analysis = analysis
        # Note: analyze_content uses chat_json (no vision) — maps to vision_analysis bucket
        # if visual_description was empty, otherwise it's a separate analysis step.
        # We accumulate into vision_analysis only if it was a pure text call (no frames);
        # for document/text-only jobs this is the first LLM call so we store in vision_analysis.
        # For video jobs with frames, vision_analysis was already populated above; here we
        # fold the content-analysis usage into its own notional key via total_cumulative_cost.
        _content_analysis_step = _llm_step_usage(analysis_usage)
        job.cost_and_usage_metrics.total_cumulative_cost = round(
            job.cost_and_usage_metrics.total_cumulative_cost + _content_analysis_step.cost, 8
        )
        # Persist content-analysis tokens into vision_analysis for video jobs (additive merge)
        # or set it directly for document jobs where no frame analysis occurred.
        if job.cost_and_usage_metrics.vision_analysis.prompt_tokens == 0:
            job.cost_and_usage_metrics.vision_analysis = _content_analysis_step
        else:
            merged = _merge_llm_usage(
                {
                    "prompt_tokens": job.cost_and_usage_metrics.vision_analysis.prompt_tokens,
                    "completion_tokens": job.cost_and_usage_metrics.vision_analysis.completion_tokens,
                    "cached_tokens": job.cost_and_usage_metrics.vision_analysis.cached_tokens,
                },
                analysis_usage,
            )
            job.cost_and_usage_metrics.vision_analysis = _llm_step_usage(merged)
        _record_metric(job, "content_analysis_time", time.perf_counter() - t0)

        # ── Step 6: Classification ─────────────────────────────────────────────
        update_job_status(job, JobStatus.CLASSIFYING, "Classifying pillar and format")
        job = get_job(job_id) or job

        t0 = time.perf_counter()
        classification, classification_usage = await asyncio.to_thread(
            classify_content,
            analysis,
            job.file.filename,
        )
        job.classification = classification
        job.cost_and_usage_metrics.classification = _llm_step_usage(classification_usage)
        job.cost_and_usage_metrics.total_cumulative_cost = round(
            job.cost_and_usage_metrics.total_cumulative_cost
            + job.cost_and_usage_metrics.classification.cost,
            8,
        )
        _record_metric(job, "classification_time", time.perf_counter() - t0)

        # ── Step 7: Asset generation ───────────────────────────────────────────
        update_job_status(job, JobStatus.GENERATING, "Generating Instagram assets")
        job = get_job(job_id) or job

        t0 = time.perf_counter()
        assets, assets_usage = await asyncio.to_thread(
            generate_assets,
            analysis,
            classification,
            job.file.filename,
        )
        recommendation, rec_usage = await asyncio.to_thread(
            generate_publishing_recommendation,
            analysis,
            classification,
        )
        job.assets = assets
        job.recommendation = recommendation

        # Merge assets + recommendation usage into the single "generation" bucket
        combined_gen_usage = _merge_llm_usage(assets_usage, rec_usage)
        job.cost_and_usage_metrics.generation = _llm_step_usage(combined_gen_usage)
        job.cost_and_usage_metrics.total_cumulative_cost = round(
            job.cost_and_usage_metrics.total_cumulative_cost
            + job.cost_and_usage_metrics.generation.cost,
            8,
        )
        _record_metric(job, "asset_generation_time", time.perf_counter() - t0)

        # ── Finalise ───────────────────────────────────────────────────────────
        job.processing_seconds = round(time.time() - wall_start, 2)
        job.profiling_metrics["total_pipeline_time"] = round(time.perf_counter() - pipeline_start, 4)
        update_job_status(job, JobStatus.COMPLETED, "Processing complete")

    except Exception as exc:
        job = get_job(job_id) or job
        job.status = JobStatus.FAILED
        job.error = str(exc)
        job.status_message = "Processing failed"
        job.processing_seconds = round(time.time() - wall_start, 2)
        job.profiling_metrics["total_pipeline_time"] = round(time.perf_counter() - pipeline_start, 4)
        save_job(job)


def schedule_job_processing(job_id: str) -> None:
    asyncio.create_task(process_job(job_id))
