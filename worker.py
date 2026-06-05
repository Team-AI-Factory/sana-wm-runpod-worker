import os
import json
import time
import glob
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timezone

import boto3
import numpy as np
from PIL import Image
from botocore.client import Config


def env(name, default=""):
    return str(os.environ.get(name, default)).strip()


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    print(str(message), flush=True)


R2_BUCKET = env("R2_BUCKET", "wankelpie")
R2_ACCOUNT_ID = env("R2_ACCOUNT_ID")
R2_ENDPOINT = env("R2_ENDPOINT") or (
    f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com" if R2_ACCOUNT_ID else ""
)
R2_ACCESS_KEY_ID = env("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = env("R2_SECRET_ACCESS_KEY")

R2_ACTIVE_PREFIX = env("R2_ACTIVE_PREFIX", "jobs/active/").strip("/") + "/"
R2_RUNNING_PREFIX = env("R2_RUNNING_PREFIX", "jobs/running/").strip("/") + "/"
R2_DONE_PREFIX = env("R2_DONE_PREFIX", "jobs/done/").strip("/") + "/"
R2_FAILED_PREFIX = env("R2_FAILED_PREFIX", "jobs/failed/").strip("/") + "/"
R2_OUTPUT_PREFIX = env("R2_OUTPUT_PREFIX", "videos/sana-wm/outputs/").strip("/") + "/"

POLL_SECONDS = int(env("POLL_SECONDS", "30"))

WORKSPACE = Path(env("WORKSPACE", "/workspace"))
SANA_DIR = WORKSPACE / "Sana"
JOB_INPUT_DIR = WORKSPACE / "job-input"
RESULTS_DIR = WORKSPACE / "results"


def make_valid_sana_frames(value):
    """
    SANA-WM / LTX VAE wants frame counts shaped like 8k+1.
    Examples: 321, 641, 1041.
    If user accidentally sets 1040, this safely changes it to 1041.
    """
    try:
        frames = int(value)
    except Exception:
        frames = 321

    if frames < 9:
        frames = 321

    remainder = (frames - 1) % 8

    if remainder == 0:
        return frames

    fixed = frames + (8 - remainder)
    log(f"SANA_FRAMES_AUTO_FIXED: {frames} -> {fixed} because SANA-WM requires 8k+1 frames")
    return fixed


SANA_FRAMES = make_valid_sana_frames(env("SANA_FRAMES", "321"))
SANA_STEPS = int(env("SANA_STEPS", "20"))
SANA_NO_REFINER = env("SANA_NO_REFINER", "false").lower() == "true"
NO_ACTION_OVERLAY = env("NO_ACTION_OVERLAY", "true").lower() != "false"


def make_action_for_frames(frames):
    """
    Action segments must add up to frames - 1.
    Old default was 80+40+40+60+100 = 320, giving 321 frames.
    This scales that same movement pattern to the requested length.
    """
    total = frames - 1

    if total <= 0:
        total = 320

    weights = [80, 40, 40, 60, 100]
    labels = ["w", "jw", "w", "lw", "w"]
    base = sum(weights)

    parts = []
    used = 0

    for index, weight in enumerate(weights):
        if index == len(weights) - 1:
            count = total - used
        else:
            count = max(1, round(total * weight / base))
            used += count

        parts.append(f"{labels[index]}-{count}")

    return ",".join(parts)


def parse_action_total(action):
    total = 0

    for part in str(action or "").split(","):
        if "-" not in part:
            continue

        try:
            total += int(part.split("-")[-1])
        except Exception:
            pass

    return total


def get_sana_action():
    raw_action = env("SANA_ACTION", "")

    if not raw_action or raw_action.lower() == "auto":
        action = make_action_for_frames(SANA_FRAMES)
        log(f"SANA_ACTION_AUTO_CREATED: {action}")
        return action

    total = parse_action_total(raw_action)
    expected = SANA_FRAMES - 1

    if total != expected:
        action = make_action_for_frames(SANA_FRAMES)
        log(f"SANA_ACTION_AUTO_FIXED: old total {total}, expected {expected}. New action: {action}")
        return action

    return raw_action


SANA_ACTION = get_sana_action()


def make_s3_client():
    if not R2_ENDPOINT:
        raise RuntimeError("Missing R2_ENDPOINT or R2_ACCOUNT_ID.")

    if not R2_ACCESS_KEY_ID:
        raise RuntimeError("Missing R2_ACCESS_KEY_ID.")

    if not R2_SECRET_ACCESS_KEY:
        raise RuntimeError("Missing R2_SECRET_ACCESS_KEY.")

    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )


s3 = make_s3_client()


def list_named_active_jobs():
    paginator = s3.get_paginator("list_objects_v2")
    jobs = []

    for page in paginator.paginate(Bucket=R2_BUCKET, Prefix=R2_ACTIVE_PREFIX):
        for item in page.get("Contents", []):
            key = item.get("Key", "")
            filename = key.split("/")[-1]

            if not key.endswith(".json"):
                continue

            if filename == "current.json":
                continue

            if not filename.startswith("sana-wm-"):
                continue

            jobs.append(
                {
                    "key": key,
                    "modified": item.get("LastModified"),
                }
            )

    jobs.sort(
        key=lambda item: item["modified"]
        or datetime.min.replace(tzinfo=timezone.utc)
    )

    return [item["key"] for item in jobs]


def get_json_from_r2(key):
    response = s3.get_object(Bucket=R2_BUCKET, Key=key)
    body = response["Body"].read().decode("utf-8")
    return json.loads(body)


def put_json_to_r2(key, data):
    s3.put_object(
        Bucket=R2_BUCKET,
        Key=key,
        Body=json.dumps(data, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    log(f"wrote {key}")


def delete_r2_key(key):
    s3.delete_object(Bucket=R2_BUCKET, Key=key)
    log(f"deleted {key}")


def upload_file_to_r2(local_path, key, content_type):
    with open(local_path, "rb") as file:
        s3.put_object(
            Bucket=R2_BUCKET,
            Key=key,
            Body=file,
            ContentType=content_type,
        )

    log(f"uploaded {key}")


def download_file_from_r2(key, local_path):
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(R2_BUCKET, key, str(local_path))


def create_default_reference_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        return

    image = Image.new("RGB", (1280, 704), (35, 38, 42))
    image.save(path)


def create_default_intrinsics(path):
    """
    Important fix:
    SANA accepts intrinsics shape (4,), (3,3), or (F,3,3).
    The old worker created (1040,4), which caused the crash.
    This creates simple shape (4,), which SANA accepts.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    intrinsics = np.array([900.0, 900.0, 640.0, 352.0], dtype=np.float32)
    np.save(path, intrinsics)


def prepare_job_input(job):
    shutil.rmtree(JOB_INPUT_DIR, ignore_errors=True)
    JOB_INPUT_DIR.mkdir(parents=True, exist_ok=True)

    prompt = job.get("video_prompt") or job.get("prompt") or ""
    prompt_path = JOB_INPUT_DIR / "prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    image_path = JOB_INPUT_DIR / "start.png"
    image_key = (
        job.get("reference_image_key")
        or job.get("image_key")
        or job.get("r2_reference_image_key")
    )

    if image_key:
        download_file_from_r2(image_key, image_path)
    else:
        create_default_reference_image(image_path)

    intrinsics_path = JOB_INPUT_DIR / "intrinsics.npy"
    intrinsics_key = job.get("intrinsics_key") or job.get("r2_intrinsics_key")

    if intrinsics_key:
        download_file_from_r2(intrinsics_key, intrinsics_path)
    else:
        create_default_intrinsics(intrinsics_path)

    return prompt_path, image_path, intrinsics_path


def find_generated_mp4(job_id):
    patterns = [
        str(RESULTS_DIR / f"{job_id}_generated.mp4"),
        str(RESULTS_DIR / "*.mp4"),
    ]

    files = []

    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))

    if not files:
        return ""

    files.sort(key=lambda path: os.path.getsize(path), reverse=True)

    return files[0]


def clean_old_results():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for old_file in glob.glob(str(RESULTS_DIR / "*.mp4")):
        try:
            os.remove(old_file)
        except Exception:
            pass

    for old_file in glob.glob(str(RESULTS_DIR / "*.log")):
        try:
            os.remove(old_file)
        except Exception:
            pass


def run_sana_wm(job):
    job_id = job["job_id"]

    clean_old_results()

    prompt_path, image_path, intrinsics_path = prepare_job_input(job)

    command = [
        "python",
        "inference_video_scripts/inference_sana_wm.py",
        "--image",
        str(image_path),
        "--prompt",
        str(prompt_path),
        "--intrinsics",
        str(intrinsics_path),
        "--action",
        SANA_ACTION,
        "--num_frames",
        str(SANA_FRAMES),
        "--step",
        str(SANA_STEPS),
        "--output_dir",
        str(RESULTS_DIR),
        "--name",
        job_id,
    ]

    if NO_ACTION_OVERLAY:
        command.append("--no_action_overlay")

    if SANA_NO_REFINER:
        command.append("--no_refiner")

    log("SANA_GENERATION_STARTED: started Starting official SANA-WM inference")
    log("COMMAND: " + " ".join(command))

    result = subprocess.run(
        command,
        cwd=str(SANA_DIR),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    log(result.stdout)

    log_path = RESULTS_DIR / f"{job_id}_generation.log"
    log_path.write_text(result.stdout, encoding="utf-8")

    if result.returncode != 0:
        raise RuntimeError(f"SANA-WM failed with exit code {result.returncode}")

    mp4_path = find_generated_mp4(job_id)

    if not mp4_path:
        raise RuntimeError("SANA-WM finished but no MP4 was found.")

    log("SANA_GENERATION_DONE: done SANA-WM generated raw MP4")
    return Path(mp4_path), log_path


def process_one_job(active_job_key):
    job = get_json_from_r2(active_job_key)

    job_id = job.get("job_id") or Path(active_job_key).stem
    job["job_id"] = job_id

    running_key = R2_RUNNING_PREFIX + f"{job_id}.json"
    done_key = R2_DONE_PREFIX + f"{job_id}.json"
    failed_key = R2_FAILED_PREFIX + f"{job_id}.json"

    output_folder = R2_OUTPUT_PREFIX + f"{job_id}/"
    output_video_key = output_folder + "video.mp4"
    output_log_key = output_folder + "generation.log"
    output_job_key = output_folder + "job.json"

    running_job = dict(job)
    running_job["status"] = "running"
    running_job["active_job_key"] = active_job_key
    running_job["worker_started_at"] = now_iso()
    running_job["no_action_overlay"] = NO_ACTION_OVERLAY
    running_job["sana_frames"] = SANA_FRAMES
    running_job["sana_action"] = SANA_ACTION
    running_job["sana_steps"] = SANA_STEPS

    put_json_to_r2(running_key, running_job)

    try:
        mp4_path, log_path = run_sana_wm(job)

        log("R2_UPLOAD_STARTED: started Uploading raw MP4 to R2")
        upload_file_to_r2(mp4_path, output_video_key, "video/mp4")
        upload_file_to_r2(log_path, output_log_key, "text/plain")

        done_job = dict(job)
        done_job["status"] = "done"
        done_job["active_job_key"] = active_job_key
        done_job["worker_finished_at"] = now_iso()
        done_job["output_video_key"] = output_video_key
        done_job["output_folder"] = output_folder
        done_job["no_action_overlay"] = NO_ACTION_OVERLAY
        done_job["sana_frames"] = SANA_FRAMES
        done_job["sana_action"] = SANA_ACTION
        done_job["sana_steps"] = SANA_STEPS

        put_json_to_r2(done_key, done_job)
        put_json_to_r2(output_job_key, done_job)

        delete_r2_key(active_job_key)

        temp_link = s3.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": R2_BUCKET,
                "Key": output_video_key,
            },
            ExpiresIn=604800,
        )

        log("SANA_WM_JOB_SUCCESS")
        log("R2_UPLOAD_SUCCESS")
        log(f"R2_VIDEO_KEY={output_video_key}")
        log(f"R2_TEMP_VIDEO_LINK={temp_link}")
        log(f"ACTIVE_JOB_DELETED={active_job_key}")

    except Exception as error:
        failed_job = dict(job)
        failed_job["status"] = "failed"
        failed_job["active_job_key"] = active_job_key
        failed_job["worker_failed_at"] = now_iso()
        failed_job["error"] = str(error)
        failed_job["no_action_overlay"] = NO_ACTION_OVERLAY
        failed_job["sana_frames"] = SANA_FRAMES
        failed_job["sana_action"] = SANA_ACTION
        failed_job["sana_steps"] = SANA_STEPS

        put_json_to_r2(failed_key, failed_job)

        # Important: remove the exact failed active job so the worker does not retry
        # the same broken job forever every 30 seconds.
        delete_r2_key(active_job_key)

        log(f"SANA_WM_JOB_FAILED: {error}")
        log(f"ACTIVE_JOB_DELETED_AFTER_FAILURE={active_job_key}")


def main():
    log("WORKER_LOOP_STARTED")
    log(f"Polling every {POLL_SECONDS} seconds")
    log(f"Queue folder: {R2_BUCKET}/{R2_ACTIVE_PREFIX}")
    log("Queue mode: named job files only, example sana-wm-xxxx.json")
    log(f"No action overlay: {NO_ACTION_OVERLAY}")
    log(f"SANA frames: {SANA_FRAMES}")
    log(f"SANA action: {SANA_ACTION}")
    log(f"SANA steps: {SANA_STEPS}")

    while True:
        try:
            log(f"JOB_SCAN_STARTED: scanning {R2_ACTIVE_PREFIX}")

            active_jobs = list_named_active_jobs()

            if not active_jobs:
                log(f"NO_JOB_FOUND waiting {POLL_SECONDS}s")
                time.sleep(POLL_SECONDS)
                continue

            active_job_key = active_jobs[0]

            log(f"JOB_FOUND: {active_job_key}")
            process_one_job(active_job_key)
            log(f"JOB_DONE waiting {POLL_SECONDS}s for next job")

        except Exception as error:
            log(f"WORKER_LOOP_ERROR: {error}")

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
