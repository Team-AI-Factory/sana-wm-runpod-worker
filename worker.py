import os
import json
import time
import glob
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timezone

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from PIL import Image
import numpy as np


def env(name, default=""):
    return str(os.environ.get(name, default)).strip()


def now():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    print(f"{message}", flush=True)


R2_BUCKET = env("R2_BUCKET", "wankelpie")
R2_ACCOUNT_ID = env("R2_ACCOUNT_ID")
R2_ENDPOINT = env("R2_ENDPOINT") or (
    f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com" if R2_ACCOUNT_ID else ""
)
R2_ACCESS_KEY_ID = env("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = env("R2_SECRET_ACCESS_KEY")

ACTIVE_PREFIX = env("R2_ACTIVE_PREFIX", "jobs/active/").strip("/") + "/"
RUNNING_PREFIX = env("R2_RUNNING_PREFIX", "jobs/running/").strip("/") + "/"
DONE_PREFIX = env("R2_DONE_PREFIX", "jobs/done/").strip("/") + "/"
FAILED_PREFIX = env("R2_FAILED_PREFIX", "jobs/failed/").strip("/") + "/"
OUTPUT_PREFIX = env("R2_OUTPUT_PREFIX", "videos/sana-wm/outputs/").strip("/") + "/"

POLL_SECONDS = int(env("POLL_SECONDS", "30"))
WORKSPACE = Path(env("WORKSPACE", "/workspace"))
SANA_DIR = WORKSPACE / "Sana"
JOB_INPUT_DIR = WORKSPACE / "job-input"
RESULTS_DIR = WORKSPACE / "results"

SANA_FRAMES = int(env("SANA_FRAMES", "321"))
SANA_STEPS = int(env("SANA_STEPS", "20"))
SANA_ACTION = env("SANA_ACTION", "w-80,jw-40,w-40,lw-60,w-100")

NO_ACTION_OVERLAY = env("NO_ACTION_OVERLAY", "true").lower() != "false"


def s3_client():
    if not R2_ENDPOINT or not R2_ACCESS_KEY_ID or not R2_SECRET_ACCESS_KEY:
        raise RuntimeError("Missing R2 endpoint or R2 access keys.")

    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )


s3 = s3_client()


def list_active_job_keys():
    paginator = s3.get_paginator("list_objects_v2")
    keys = []

    for page in paginator.paginate(Bucket=R2_BUCKET, Prefix=ACTIVE_PREFIX):
        for item in page.get("Contents", []):
            key = item.get("Key", "")
            name = key.split("/")[-1]

            if not key.endswith(".json"):
                continue

            if name == "current.json":
                continue

            if not name.startswith("sana-wm-"):
                continue

            keys.append(
                {
                    "key": key,
                    "modified": item.get("LastModified"),
                }
            )

    keys.sort(key=lambda item: item["modified"] or datetime.min.replace(tzinfo=timezone.utc))
    return [item["key"] for item in keys]


def get_json(key):
    response = s3.get_object(Bucket=R2_BUCKET, Key=key)
    body = response["Body"].read().decode("utf-8")
    return json.loads(body)


def put_json(key, data):
    s3.put_object(
        Bucket=R2_BUCKET,
        Key=key,
        Body=json.dumps(data, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    log(f"wrote {key}")


def delete_key(key):
    s3.delete_object(Bucket=R2_BUCKET, Key=key)
    log(f"deleted {key}")


def upload_file(local_path, key, content_type):
    with open(local_path, "rb") as file:
        s3.put_object(
            Bucket=R2_BUCKET,
            Key=key,
            Body=file,
            ContentType=content_type,
        )
    log(f"uploaded {key}")


def download_file(key, local_path):
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(R2_BUCKET, key, str(local_path))


def ensure_default_reference_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        return

    image = Image.new("RGB", (1280, 704), (35, 38, 42))
    image.save(path)


def ensure_intrinsics(path, frames):
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        return

    intrinsics = np.array([900.0, 900.0, 640.0, 352.0], dtype=np.float32)
    intrinsics = np.broadcast_to(intrinsics, (frames, 4)).copy()
    np.save(path, intrinsics)


def prepare_job_input(job):
    shutil.rmtree(JOB_INPUT_DIR, ignore_errors=True)
    JOB_INPUT_DIR.mkdir(parents=True, exist_ok=True)

    prompt = job.get("video_prompt") or job.get("prompt") or ""
    prompt_path = JOB_INPUT_DIR / "prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    image_path = JOB_INPUT_DIR / "start.png"
    reference_image_key = (
        job.get("reference_image_key")
        or job.get("image_key")
        or job.get("r2_reference_image_key")
    )

    if reference_image_key:
        download_file(reference_image_key, image_path)
    else:
        ensure_default_reference_image(image_path)

    intrinsics_path = JOB_INPUT_DIR / "intrinsics.npy"
    intrinsics_key = job.get("intrinsics_key") or job.get("r2_intrinsics_key")

    if intrinsics_key:
        download_file(intrinsics_key, intrinsics_path)
    else:
        ensure_intrinsics(intrinsics_path, SANA_FRAMES)

    return prompt_path, image_path, intrinsics_path


def find_output_mp4(job_id):
    patterns = [
        str(RESULTS_DIR / f"{job_id}_generated.mp4"),
        str(RESULTS_DIR / "*.mp4"),
        str(WORKSPACE / "outputs" / "**" / "*.mp4"),
    ]

    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))

    if not files:
        return ""

    files.sort(key=lambda path: os.path.getsize(path), reverse=True)
    return files[0]


def run_sana(job):
    job_id = job["job_id"]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for old_mp4 in glob.glob(str(RESULTS_DIR / "*.mp4")):
        try:
            os.remove(old_mp4)
        except Exception:
            pass

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

    mp4_path = find_output_mp4(job_id)

    if not mp4_path:
        raise RuntimeError("SANA-WM finished but no MP4 was found.")

    log("SANA_GENERATION_DONE: done SANA-WM generated MP4")
    return Path(mp4_path), log_path


def process_job(active_key):
    job = get_json(active_key)
    job_id = job.get("job_id") or Path(active_key).stem
    job["job_id"] = job_id

    running_key = RUNNING_PREFIX + f"{job_id}.json"
    done_key = DONE_PREFIX + f"{job_id}.json"
    failed_key = FAILED_PREFIX + f"{job_id}.json"
    output_video_key = OUTPUT_PREFIX + f"{job_id}/video.mp4"
    output_log_key = OUTPUT_PREFIX + f"{job_id}/generation.log"
    output_job_key = OUTPUT_PREFIX + f"{job_id}/job.json"

    running_job = dict(job)
    running_job["status"] = "running"
    running_job["active_job_key"] = active_key
    running_job["worker_started_at"] = now()
    running_job["no_action_overlay"] = NO_ACTION_OVERLAY

    put_json(running_key, running_job)

    try:
        mp4_path, log_path = run_sana(job)

        log("R2_UPLOAD_STARTED: started Uploading video to R2")
        upload_file(mp4_path, output_video_key, "video/mp4")
        upload_file(log_path, output_log_key, "text/plain")

        done_job = dict(job)
        done_job["status"] = "done"
        done_job["active_job_key"] = active_key
        done_job["worker_finished_at"] = now()
        done_job["output_video_key"] = output_video_key
        done_job["output_folder"] = OUTPUT_PREFIX + f"{job_id}/"
        done_job["no_action_overlay"] = NO_ACTION_OVERLAY

        put_json(done_key, done_job)
        put_json(output_job_key, done_job)

        delete_key(active_key)

        log("SANA_WM_JOB_SUCCESS")
        log("R2_UPLOAD_SUCCESS")
        log(f"R2_VIDEO_KEY={output_video_key}")
        log(f"ACTIVE_JOB_DELETED={active_key}")

    except Exception as error:
        failed_job = dict(job)
        failed_job["status"] = "failed"
        failed_job["active_job_key"] = active_key
        failed_job["worker_failed_at"] = now()
        failed_job["error"] = str(error)
        failed_job["no_action_overlay"] = NO_ACTION_OVERLAY

        put_json(failed_key, failed_job)

        log(f"SANA_WM_JOB_FAILED: {error}")
        log(f"ACTIVE_JOB_NOT_DELETED={active_key}")


def main():
    log("WORKER_LOOP_STARTED")
    log(f"Polling every {POLL_SECONDS} seconds for jobs under: {ACTIVE_PREFIX}")
    log("Queue mode: named sana-wm-xxxx.json files only")
    log(f"No action overlay: {NO_ACTION_OVERLAY}")

    while True:
        try:
            log(f"JOB_SCAN_STARTED: reading {ACTIVE_PREFIX}*.json")
            keys = list_active_job_keys()

            if not keys:
                log(f"NO_JOB_FOUND waiting {POLL_SECONDS}s")
                time.sleep(POLL_SECONDS)
                continue

            active_key = keys[0]
            log(f"JOB_FOUND: {active_key}")
            process_job(active_key)
            log(f"JOB_DONE waiting {POLL_SECONDS}s for next job")

        except Exception as error:
            log(f"WORKER_LOOP_ERROR: {error}")

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
