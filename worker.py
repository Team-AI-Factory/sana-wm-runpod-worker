import json
import os
import subprocess
import time
from pathlib import Path

import boto3
import numpy as np
from botocore.client import Config
from PIL import Image, ImageDraw


WORKSPACE = Path(os.environ.get("WORKSPACE_DIR", "/workspace"))
SANA_HOME = Path(os.environ.get("SANA_HOME", "/workspace/Sana"))
JOB_INPUT_DIR = WORKSPACE / "job-input"
RESULTS_DIR = WORKSPACE / "results"


def now_id():
    return time.strftime("%Y%m%d-%H%M%S")


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def upload_json(s3, bucket, key, data):
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data, indent=2),
        ContentType="application/json",
    )


def stage(s3, bucket, job_id, stage_name, status="started", message="", data=None):
    data = data or {}
    payload = {
        "job_id": job_id,
        "stage": stage_name,
        "status": status,
        "message": message,
        "data": data,
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    print(f"{stage_name}: {status} {message}", flush=True)

    safe_stage = stage_name.lower().replace(" ", "_")
    key_base = f"learning/job-history/{job_id}/stages"
    upload_json(s3, bucket, f"{key_base}/{now_id()}-{safe_stage}.json", payload)
    upload_json(s3, bucket, f"learning/job-history/{job_id}/latest-stage.json", payload)


def load_job(s3, bucket, job_key):
    result = s3.get_object(Bucket=bucket, Key=job_key)
    raw = result["Body"].read().decode("utf-8")
    return json.loads(raw)


def create_fallback_reference_image(prompt, output_path):
    width, height = 1280, 704
    text = prompt.lower()

    img = Image.new("RGB", (width, height), (38, 84, 130))
    draw = ImageDraw.Draw(img)

    for y in range(height):
        r = int(28 + y * 0.06)
        g = int(78 + y * 0.05)
        b = int(125 + y * 0.03)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    if "sun" in text or "sunrise" in text:
        draw.ellipse((920, 70, 1080, 230), fill=(255, 200, 90))

    draw.rectangle((0, 430, width, height), fill=(25, 95, 135))

    for y in range(455, height, 30):
        draw.line([(0, y), (width, y + 8)], fill=(120, 175, 205), width=2)

    draw.polygon([(0, 430), (220, 190), (470, 430)], fill=(55, 75, 84))
    draw.polygon([(320, 430), (590, 130), (900, 430)], fill=(45, 68, 80))
    draw.polygon([(720, 430), (1000, 210), (1280, 430)], fill=(50, 78, 88))

    draw.text((40, 40), "SANA-WM reference frame", fill=(255, 255, 255))
    img.save(output_path)


def download_reference_image_if_available(s3, bucket, job, output_path):
    reference_key = job.get("reference_image_key")
    if not reference_key:
        return False

    print(f"Downloading reference image: {reference_key}", flush=True)
    obj = s3.get_object(Bucket=bucket, Key=reference_key)
    output_path.write_bytes(obj["Body"].read())
    return True


def choose_settings(job):
    duration = int(job.get("duration_seconds", 60))

    if duration >= 60:
        return {
            "num_frames": int(os.environ.get("FULL_NUM_FRAMES", "321")),
            "step": int(os.environ.get("FULL_STEP", "20")),
            "use_refiner": os.environ.get("USE_REFINER", "true").lower() == "true",
        }

    return {
        "num_frames": int(os.environ.get("TINY_NUM_FRAMES", "9")),
        "step": int(os.environ.get("TINY_STEP", "2")),
        "use_refiner": False,
    }


def run_command(command, cwd):
    print("COMMAND:", " ".join(command), flush=True)
    process = subprocess.run(command, cwd=str(cwd))
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}")


def find_mp4():
    files = sorted(RESULTS_DIR.rglob("*.mp4"))
    if not files:
        raise RuntimeError("No MP4 file found in results directory")
    return files[0]


def main():
    bucket = os.environ["R2_BUCKET"]
    job_key = os.environ.get("JOB_KEY", "jobs/sana-wm/active/current.json")
    s3 = s3_client()

    startup_job_id = "runpod-startup-" + now_id()

    try:
        stage(s3, bucket, startup_job_id, "JOB_LOAD_STARTED", "started", f"Reading {job_key}")
        job = load_job(s3, bucket, job_key)

        job_id = job.get("job_id") or ("sana-wm-runpod-" + now_id())
        prompt = job.get("prompt") or job.get("video_prompt") or "A peaceful cinematic nature scene."
        action = job.get("action") or "w-80,jw-40,w-40,lw-60,w-100"

        stage(s3, bucket, job_id, "JOB_LOAD_DONE", "done", "Job loaded from R2", {
            "job_key": job_key,
            "prompt": prompt,
            "mode": os.environ.get("SANA_WM_MODE"),
        })

        JOB_INPUT_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        start_image = JOB_INPUT_DIR / "start.png"
        prompt_file = JOB_INPUT_DIR / "prompt.txt"
        intrinsics_file = JOB_INPUT_DIR / "intrinsics.npy"

        stage(s3, bucket, job_id, "REFERENCE_IMAGE_STARTED", "started", "Preparing reference image")

        used_reference = download_reference_image_if_available(s3, bucket, job, start_image)

        if not used_reference:
            create_fallback_reference_image(prompt, start_image)

        prompt_file.write_text(prompt, encoding="utf-8")
        np.save(intrinsics_file, np.array([1000.0, 1000.0, 640.0, 352.0], dtype=np.float32))

        stage(s3, bucket, job_id, "REFERENCE_IMAGE_READY", "done", "Reference image ready", {
            "used_reference_image": used_reference
        })

        settings = choose_settings(job)

        stage(s3, bucket, job_id, "SANA_GENERATION_STARTED", "started", "Starting official SANA-WM inference", {
            "settings": settings,
            "action": action,
            "mode": os.environ.get("SANA_WM_MODE"),
        })

        command = [
            "python",
            "inference_video_scripts/inference_sana_wm.py",
            "--image",
            str(start_image),
            "--prompt",
            str(prompt_file),
            "--intrinsics",
            str(intrinsics_file),
            "--action",
            action,
            "--num_frames",
            str(settings["num_frames"]),
            "--step",
            str(settings["step"]),
            "--output_dir",
            str(RESULTS_DIR),
            "--name",
            job_id,
        ]

        if not settings["use_refiner"]:
            command.append("--no_refiner")

        run_command(command, SANA_HOME)

        mp4_file = find_mp4()

        stage(s3, bucket, job_id, "SANA_GENERATION_DONE", "done", "SANA-WM generated MP4", {
            "local_mp4": str(mp4_file)
        })

        output_prefix = job.get("output_prefix")
        if not output_prefix:
            output_prefix = f"videos/sana-wm/outputs/{job_id}/"

        if not output_prefix.endswith("/"):
            output_prefix += "/"

        video_key = output_prefix + "video.mp4"
        metadata_key = job.get("metadata_key") or f"learning/job-history/{job_id}/metadata.json"
        result_key = f"learning/job-history/{job_id}/result.json"

        stage(s3, bucket, job_id, "R2_UPLOAD_STARTED", "started", "Uploading video to R2", {
            "video_key": video_key
        })

        s3.upload_file(
            str(mp4_file),
            bucket,
            video_key,
            ExtraArgs={"ContentType": "video/mp4"},
        )

        temp_link = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": video_key},
            ExpiresIn=604800,
        )

        completed = {
            **job,
            "job_id": job_id,
            "status": "completed",
            "model": "sana-wm",
            "gpu_provider": "runpod",
            "sana_wm_mode": os.environ.get("SANA_WM_MODE"),
            "video_key": video_key,
            "metadata_key": metadata_key,
            "result_key": result_key,
            "temporary_video_link": temp_link,
            "settings": settings,
            "used_reference_image": used_reference,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        upload_json(s3, bucket, metadata_key, completed)
        upload_json(s3, bucket, result_key, completed)

        stage(s3, bucket, job_id, "R2_UPLOAD_DONE", "done", "Video uploaded to R2", {
            "video_key": video_key,
            "temporary_video_link": temp_link,
        })

        print("SANA_WM_JOB_SUCCESS", flush=True)
        print("R2_UPLOAD_SUCCESS", flush=True)
        print("R2_VIDEO_KEY=" + video_key, flush=True)
        print("R2_TEMP_VIDEO_LINK=" + temp_link, flush=True)

    except Exception as exc:
        failed_job_id = locals().get("job_id", startup_job_id)
        try:
            stage(s3, bucket, failed_job_id, "FAILED_AT_STAGE", "failed", str(exc), {
                "job_key": job_key
            })
            upload_json(s3, bucket, f"learning/mistakes/{failed_job_id}/error.json", {
                "job_id": failed_job_id,
                "error": str(exc),
                "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
        except Exception:
            pass

        print("SANA_WM_JOB_FAILED", flush=True)
        print(str(exc), flush=True)
        raise


if __name__ == "__main__":
    main()
