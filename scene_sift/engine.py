"""Grounding DINO object detection and Gemma 4 reasoning for SceneSift."""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from functools import lru_cache
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps


MODEL_REPO = "IDEA-Research/grounding-dino-base"
GEMMA_MODEL = "gemma-4-31b-it"
GEMMA_FALLBACK_MODEL = "gemma-4-26b-a4b-it"
GEMMA_RETRY_DELAY_SECONDS = 10
DETECTION_GROUPS = (
    ("knife", "machete", "cleaver", "box cutter", "razor blade", "scissors",
     "axe", "sword", "broken glass", "sharp metal shard", "needle", "syringe"),
    ("handgun", "pistol", "revolver", "rifle", "shotgun", "grenade",
     "explosive device"),
    ("hammer", "baseball bat", "crowbar", "metal pipe", "rope",
     "electrical cord", "chemical container", "gas canister"),
)


def configured_api_key() -> str:
    """Read the local key without displaying it or requiring another package."""
    value = os.getenv("GEMINI_API_KEY", "").strip()
    if value:
        return value
    dotenv = Path(__file__).resolve().parents[1] / ".env"
    if dotenv.is_file():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"\'')
    return ""


def open_image(raw: bytes) -> Image.Image:
    if len(raw) > 15 * 1024 * 1024:
        raise ValueError("Please use an image under 15 MB.")
    image = Image.open(io.BytesIO(raw))
    image.load()
    return ImageOps.exif_transpose(image).convert("RGB")


@lru_cache(maxsize=1)
def detector():
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    model_path = Path(os.getenv("GROUNDING_DINO_PATH", r"C:\Detections\models\grounding-dino-base"))
    source = str(model_path) if model_path.is_dir() else MODEL_REPO
    processor = AutoProcessor.from_pretrained(source, local_files_only=model_path.is_dir())
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        source, local_files_only=model_path.is_dir()
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return processor, model, device


def _overlap(a: list[int], b: list[int]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / max(1, area_a + area_b - intersection)


def detect_objects(image: Image.Image, threshold: float = 0.35,
                   extra_terms: str = "") -> list[dict]:
    import torch

    processor, model, device = detector()
    custom = []
    known = {term for group in DETECTION_GROUPS for term in group}
    for term in extra_terms.split(",")[:8]:
        clean = re.sub(r"[^a-z0-9 -]", "", term.lower()).strip(" -")[:40]
        if clean and clean not in known and clean not in custom:
            custom.append(clean)
    groups = (*DETECTION_GROUPS, tuple(custom)) if custom else DETECTION_GROUPS
    candidates = []
    for group in groups:
        prompt = " . ".join(group) + " ."
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        result = processor.post_process_grounded_object_detection(
            outputs=outputs, input_ids=inputs.input_ids,
            threshold=threshold, text_threshold=0.30,
            target_sizes=[(image.height, image.width)],
        )[0]
        labels = result.get("text_labels", result.get("labels", []))
        for box, score, raw_label in zip(result["boxes"], result["scores"], labels):
            phrase = str(raw_label).lower().strip(" .")
            if not phrase:
                continue
            x1, y1, x2, y2 = [float(value) for value in box.tolist()]
            clipped = [max(0, min(image.width, round(x1))),
                       max(0, min(image.height, round(y1))),
                       max(0, min(image.width, round(x2))),
                       max(0, min(image.height, round(y2)))]
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                continue
            candidates.append({"label": phrase.capitalize(),
                               "confidence": round(float(score), 3),
                               "box": clipped})

    found = []
    for candidate in sorted(candidates, key=lambda item: item["confidence"], reverse=True):
        if any(_overlap(candidate["box"], item["box"]) > 0.60 for item in found):
            continue
        x1, y1, x2, y2 = candidate["box"]
        candidate.update({
            "id": f"OBS-{len(found) + 1:03d}",
            "box_1000": [round(y1 / image.height * 1000), round(x1 / image.width * 1000),
                         round(y2 / image.height * 1000), round(x2 / image.width * 1000)],
            "observation": "", "context": "", "limitation": "",
            "model_verdict": "unverified",
        })
        found.append(candidate)
        if len(found) >= 24:
            break
    return found


def crop_observation(image: Image.Image, item: dict) -> Image.Image:
    return image.crop(tuple(item["box"]))


def annotated_image(image: Image.Image, items: list[dict]) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    line = max(2, image.width // 400)
    for item in items:
        x1, y1, x2, y2 = item["box"]
        color = "#00d8e8" if item.get("source") == "Gemma 4" else "#ffb500"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line)
        score = item.get("confidence")
        label = f"{item['id']} {item['label']}" + (f" {score:.0%}" if score is not None else "")
        text_box = draw.textbbox((x1, y1), label)
        tw, th = text_box[2] - text_box[0] + 8, text_box[3] - text_box[1] + 6
        top = max(0, y1 - th)
        draw.rectangle((x1, top, x1 + tw, top + th), fill=color)
        draw.text((x1 + 4, top + 2), label, fill="#111827")
        if item.get("gemma_box"):
            draw.rectangle(tuple(item["gemma_box"]), outline="#00d8e8", width=line)
    return output


def _image_part(image: Image.Image) -> dict:
    reduced = image.copy()
    reduced.thumbnail((768, 768), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    reduced.save(buffer, format="JPEG", quality=75, optimize=True)
    return {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(buffer.getvalue()).decode("ascii")}}


def _gemma(parts: list[dict], key: str, json_output: bool = False,
           models: tuple[str, ...] | None = None, return_model: bool = False):
    if not key:
        raise ValueError("Set GEMINI_API_KEY in the local .env file to use Gemma 4 reasoning.")
    payload = {"contents": [{"role": "user", "parts": parts}]}
    response = None
    last_error = ""
    # The 26B MoE model is materially faster for image requests. Try it twice
    # before the larger 31B model so ordinary scans do not time out first.
    model_sequence = models or (GEMMA_FALLBACK_MODEL, GEMMA_FALLBACK_MODEL, GEMMA_MODEL)
    for attempt, model_name in enumerate(model_sequence):
        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent",
                headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                json=payload, timeout=(15, 100),
            )
        except requests.RequestException as exc:
            last_error = f"{model_name}: {type(exc).__name__}"
        else:
            if response.ok:
                break
            try:
                message = response.json()["error"]["message"]
            except (ValueError, KeyError, TypeError):
                message = response.text[:200]
            last_error = f"{model_name} returned {response.status_code}: {message}"
            if response.status_code not in {408, 429, 500, 502, 503, 504}:
                raise RuntimeError(f"Gemma API error: {last_error}")
        if attempt < len(model_sequence) - 1:
            retry_after = 0
            if response is not None:
                try:
                    retry_after = int(response.headers.get("Retry-After", "0"))
                except (TypeError, ValueError):
                    retry_after = 0
            time.sleep(max(GEMMA_RETRY_DELAY_SECONDS, retry_after))
    else:
        raise RuntimeError(f"Gemma API remained unavailable after retries. {last_error}")
    try:
        returned_parts = response.json()["candidates"][0]["content"]["parts"]
        text = "\n".join(p.get("text", "") for p in returned_parts if not p.get("thought"))
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Gemma returned no readable response.") from exc
    if not text.strip():
        raise RuntimeError("Gemma returned no final answer.")
    if json_output:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text):
            try:
                parsed, _ = decoder.raw_decode(text[match.start():])
            except ValueError:
                continue
            if isinstance(parsed, dict):
                result = json.dumps(parsed)
                return (result, model_name) if return_model else result
        raise RuntimeError("Gemma did not return JSON. Try again.")
    result = text.strip()
    return (result, model_name) if return_model else result


def explain_scene(image: Image.Image, items: list[dict], key: str) -> dict:
    item_list = [{"id": item["id"], "label": item["label"],
                  "source": item.get("source", "Grounding DINO"),
                  "box_1000": item["box_1000"],
                  "existing_detail": item.get("observation", "")} for item in items]
    prompt = (
        "You are assisting a human reviewer of a photograph. Grounding DINO and Gemma 4 proposed "
        "the following visual items. They may be false positives. Inspect the image and each box. "
        "Describe only what is visually apparent. Do not infer intent, events before the photo, "
        "identity, guilt, or that an object is definitely dangerous. "
        "Return only a JSON object with scene_summary and items. Each item has id, observation "
        "(what is visible), context (nearby visible context), limitation (what the image cannot establish), "
        "and model_verdict (one of likely_match, likely_false_positive, uncertain). "
        "Use likely_false_positive when a proposed grenade is visibly a vegetable, for example. "
        "Use one item per supplied ID; do not add IDs or other evidence. "
        f"Visual proposals: {json.dumps(item_list)}"
    )
    result = json.loads(_gemma([{"text": prompt}, _image_part(image)], key, json_output=True))
    allowed = {item["id"] for item in items}
    entries = result.get("items", [])
    if not isinstance(entries, list):
        entries = []
    explanations = {entry.get("id"): entry for entry in entries
                    if isinstance(entry, dict) and entry.get("id") in allowed}
    for item in items:
        entry = explanations.get(item["id"], {})
        for field in ("observation", "context", "limitation"):
            item[field] = str(entry.get(field, item.get(field, "")))[:500]
        verdict = entry.get("model_verdict", "uncertain")
        item["model_verdict"] = verdict if verdict in {"likely_match", "likely_false_positive", "uncertain"} else "uncertain"
    return {"scene_summary": str(result.get("scene_summary", ""))[:800], "items": items}


def ask_question(image: Image.Image, items: list[dict], question: str, history: list[dict], key: str) -> str:
    context = [{field: item.get(field) for field in ("id", "label", "observation", "context", "limitation")}
               for item in items]
    prompt = (
        "Answer a question about this photograph for a human reviewer. Ground your answer in visible "
        "details and cite observation IDs where relevant. Distinguish observed details from possibilities. "
        "If the image cannot answer, say so. Do not assert identity, intent, guilt, or a sequence of events. "
        f"Observations: {json.dumps(context)}. Recent conversation: {json.dumps(history[-6:])}. "
        f"Question: {question[:800]}"
    )
    return _gemma([{"text": prompt}, _image_part(image)], key)


def review_statement(image: Image.Image, items: list[dict], statement: str, key: str) -> dict:
    context = [{field: item.get(field) for field in ("id", "label", "observation", "context", "limitation")}
               for item in items]
    prompt = (
        "Compare this statement only with what is visible in the attached photograph. "
        "Return only JSON: assessment (one of direct_visual_conflict, no_direct_conflict, "
        "cannot_determine), explanation, cited_ids (array of observation IDs). A direct visual conflict "
        "requires the image itself to plainly show the opposite of a concrete claim. Do not treat "
        "absence from the image as proof of absence. Do not infer where a person was, who used an item, "
        "intent, guilt, or timing. If unsure, use cannot_determine. "
        f"Observations: {json.dumps(context)}. Statement: {statement[:1500]}"
    )
    result = json.loads(_gemma([{"text": prompt}, _image_part(image)], key, json_output=True))
    assessment = result.get("assessment", "cannot_determine")
    if assessment not in {"direct_visual_conflict", "no_direct_conflict", "cannot_determine"}:
        assessment = "cannot_determine"
    allowed = {item["id"] for item in items}
    return {"assessment": assessment, "explanation": str(result.get("explanation", ""))[:1200],
            "cited_ids": [value for value in result.get("cited_ids", []) if value in allowed]}
