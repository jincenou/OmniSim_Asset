"""Local OpenAI-compatible Qwen video-frame description; no motion generation."""
import asyncio
import base64
import hashlib
import json
import os
import time
from pathlib import Path
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "text_motion/vlm"
STATE = {"state": "starting", "cases": {}}
PROMPT = """You describe the observed motion of one marked target person in a short driving video.
You receive EVERY native frame in chronological order, each with elapsed seconds and target bbox.
These are moving camera, moving crop observations. Pixel displacement, apparent scale changes and
the car overtaking a person DO NOT establish walking direction or body motion. The camera motion
summary is context, not a human world-trajectory label. Use visible limb/body articulation and
consistent temporal evidence. Keep occluded joints UNKNOWN; never invent hidden foot phase.
Report observed walking, standing, turning, or other behavior, including changes over the clip.
Do not force everyone to walk forward or stand still. Dark clothing and occlusion are not motion.
The desired downstream output is a short English motion prompt, at most 12 words, describing ONLY
supported body behavior. It is for a motion prior, not image generation. Avoid color, camera,
scenery, names, body-shape guesses, and instructions such as 'fill the gap'. If unsupported,
condmdi_prompt must be null. Do not use hidden-body claims as evidence.
Return a JSON object with exactly these keys:
action: one of walking, standing, running, turning, stepping, other, unknown;
body_relative_direction: one of forward, backward, sideways, in_place, unknown;
condmdi_prompt: English string of at most 12 words, or null;
confidence: number 0 to 1 (subjective, not calibrated);
evidence: list of {start_seconds, end_seconds, observation}, grounded in visible frames;
occlusion_intervals: list of {start_seconds, end_seconds, parts}, parts a list of body-part strings;
uncertainties: list of strings.
The complete clip can include multiple actions; action is the dominant supported category and
the prompt may mention a transition only when visible evidence supports it. Return JSON only."""


def save():
    tmp = OUT / "status.writing"
    tmp.write_text(json.dumps(STATE, indent=2) + "\n")
    tmp.replace(OUT / "status.json")


def content_for(case, record):
    source = ROOT.parent / "frontend_quality_20261020/motion_unmasked" / case
    native = json.loads((source / "native.json").read_text())
    t0 = native["frames"][0]["image_center_timestamp_s"]
    context = dict(sample=case, frames=len(native["frames"]),
                   duration_seconds=native["frames"][-1]["image_center_timestamp_s"] - t0,
                   ego_path_m=record["ego_path_m"], ego_rotation_deg=record["ego_rotation_deg"],
                   target_identity_scope=record["identity_scope"],
                   input_scope="unsegmented, undistorted moving crop; native observed RGB, no generated/hidden reference")
    parts = [{"type": "text", "text": json.dumps(context)}]
    hashes = []
    for f in native["frames"]:
        path = Path(f["path"])
        assert path.is_relative_to(source / "native")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert digest == f["sha256"]
        stamp = f["image_center_timestamp_s"] - t0
        parts.append({"type": "text", "text": json.dumps(dict(native_index=f["native_index"],
                       elapsed_seconds=stamp, target_bbox_xyxy=f["bbox_input_xyxy"]))})
        parts.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(raw).decode()}})
        hashes.append(dict(index=f["native_index"], seconds=stamp, sha256=digest))
    return parts, context, hashes


def validate(value, duration):
    assert set(value) == {"action", "body_relative_direction", "condmdi_prompt", "confidence",
                          "evidence", "occlusion_intervals", "uncertainties"}
    assert value["action"] in ["walking", "standing", "running", "turning", "stepping", "other", "unknown"]
    assert value["body_relative_direction"] in ["forward", "backward", "sideways", "in_place", "unknown"]
    assert isinstance(value["confidence"], (float, int)) and 0 <= value["confidence"] <= 1
    prompt = value["condmdi_prompt"]
    assert prompt is None or isinstance(prompt, str) and len(prompt.split()) <= 12
    assert isinstance(value["uncertainties"], list) and all(isinstance(s, str) for s in value["uncertainties"])
    for key in ["evidence", "occlusion_intervals"]:
        assert isinstance(value[key], list)
        for event in value[key]:
            assert 0 <= event["start_seconds"] <= event["end_seconds"] <= duration + .051
            if key == "evidence":
                assert isinstance(event["observation"], str)
            else:
                assert isinstance(event["parts"], list)


async def main():
    assert not (OUT / "status.json").exists()
    protocol = json.loads((OUT / "PROTOCOL.json").read_text())
    assert hashlib.sha256((OUT / "PROTOCOL.json").read_bytes()).hexdigest() == (OUT / "PROTOCOL.sha256").read_text().strip()
    client = AsyncOpenAI(base_url="http://127.0.0.1:18493/v1", api_key="local-only",
                         timeout=240, max_retries=0)
    device_check = await asyncio.create_subprocess_exec(
        "nvidia-smi", "--id=0", "--query-gpu=uuid", "--format=csv,noheader",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    device_out, device_err = await device_check.communicate()
    assert device_check.returncode == 0, device_err.decode()
    assert device_out.decode().strip() == "GPU-0db9aa59-0293-a3ea-06a0-395913880fbd"
    # vLLM 0.11's NVML adapter requires numeric CUDA_VISIBLE_DEVICES.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0",
               PATH="/mnt/ojc/miniconda3/envs/vllm_vlm/bin:" + os.environ.get("PATH", ""),
               VLLM_USE_FLASHINFER_SAMPLER="0",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONNOUSERSITE="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_HOME=str(OUT / "cache/hf"), VLLM_CACHE_ROOT=str(OUT / "cache/vllm"),
               TORCHINDUCTOR_CACHE_DIR=str(OUT / "cache/inductor"),
               TRITON_CACHE_DIR=str(OUT / "cache/triton"), XDG_CACHE_HOME=str(OUT / "cache/xdg"))
    argv = [
        "/mnt/ojc/miniconda3/envs/vllm_vlm/bin/python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "/mnt/ojc/models/qwen3_vl/Qwen3-VL-8B-Instruct",
        "--served-model-name", "fd-motion-describer", "--host", "127.0.0.1", "--port", "18493",
        "--dtype", "bfloat16", "--max-model-len", "16384", "--max-num-seqs", "2",
        "--gpu-memory-utilization", "0.60", "--enforce-eager",
        "--limit-mm-per-prompt", '{"image":64,"video":0}', "--mm-processor-kwargs", '{"max_pixels":147456}',
    ]
    with (OUT / "server.log").open("x") as log:
        server = await asyncio.create_subprocess_exec(*argv, env=env, stdout=log, stderr=log)
        STATE.update(state="loading_local_model", server_pid=server.pid, started=time.time())
        save()
        try:
            deadline = time.time() + 900
            while True:
                if server.returncode is not None:
                    raise RuntimeError(f"Local model server exited: {server.returncode}")
                try:
                    models = await client.models.list()
                except Exception as error:
                    if time.time() >= deadline:
                        raise RuntimeError("Local server readiness deadline exceeded") from error
                    await asyncio.sleep(3)
                    continue
                assert any(m.id == "fd-motion-describer" for m in models.data)
                break
            cohort = {x["sample_id"]: x for x in json.loads((ROOT / "cohort_v1.json").read_text())["dev6"]}
            semaphore = asyncio.Semaphore(2)
            STATE["state"] = "describing"
            save()

            async def describe(case):
                async with semaphore:
                    parts, context, hashes = await asyncio.to_thread(content_for, case, cohort[case])
                    STATE["cases"][case] = dict(state="running", started=time.time())
                    save()
                    try:
                        response = await client.chat.completions.create(
                            model="fd-motion-describer", messages=[{"role": "system", "content": PROMPT},
                                                                   {"role": "user", "content": parts}],
                            temperature=0, seed=42, max_tokens=1200, response_format={"type": "json_object"})
                        text = response.choices[0].message.content
                        raw = dict(response_text=text, finish_reason=response.choices[0].finish_reason,
                                   usage=response.usage.model_dump(), input=context, frames=hashes,
                                   request_protocol_sha256=hashlib.sha256((OUT / "PROTOCOL.json").read_bytes()).hexdigest())
                        (OUT / "cases" / (case + ".json")).write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n")
                        assert response.choices[0].finish_reason == "stop", response.choices[0].finish_reason
                        parsed = json.loads(text)
                        validate(parsed, context["duration_seconds"])
                        (OUT / "cases" / (case + "_parsed.json")).write_text(json.dumps(parsed, indent=2, ensure_ascii=False) + "\n")
                        STATE["cases"][case].update(state="described_needs_evidence_review",
                                                     prompt=parsed["condmdi_prompt"], completed=time.time())
                    except Exception as error:
                        STATE["cases"][case].update(state="failed", error=str(error), completed=time.time())
                    save()

            await asyncio.gather(*(describe(c) for c in protocol["cases"]))
            STATE["state"] = "finished_needs_evidence_review" if all(v["state"] != "failed" for v in STATE["cases"].values()) else "completed_with_failures"
        except Exception as error:
            STATE.update(state="failed", error=str(error))
        finally:
            await client.close()
            if server.returncode is None:
                server.terminate()
                try:
                    await asyncio.wait_for(server.wait(), 45)
                except asyncio.TimeoutError:
                    server.kill()
                    await server.wait()
            STATE.update(server_exit_code=server.returncode, finished=time.time(), server_stopped=True)
            save()


if __name__ == "__main__":
    asyncio.run(main())
