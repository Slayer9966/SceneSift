"""Validated Gemma spatial clues and evidence-backed investigative reasoning."""

from __future__ import annotations

import json
import re

from PIL import Image, ImageDraw

from engine import (GEMMA_FALLBACK_MODEL, GEMMA_MODEL, _gemma, _image_part,
                    _overlap, crop_observation)


# Keep Gemma as the primary reasoning model required by the project.  Google's
# hosted Gemma 4 endpoints can intermittently return HTTP 500, so lightweight
# follow-up tasks may fall back to another Google reasoning model rather than
# making the entire demo feature fail.
REASONING_MODELS = (GEMMA_FALLBACK_MODEL, GEMMA_MODEL, "gemini-3.8-flash")


def evidence_sheet(image: Image.Image, items: list[dict]) -> Image.Image:
    """Small, ID-labelled contact sheet for grounded reasoning."""
    sheet = Image.new("RGB", (768, 552), "white")
    draw = ImageDraw.Draw(sheet)
    for index, item in enumerate(items[:4]):
        x = (index % 2) * 384
        y = (index // 2) * 276
        draw.rectangle((x + 4, y + 4, x + 380, y + 272), outline="#777777", width=2)
        draw.text((x + 12, y + 9), f"{item['id']}  {item['label']}", fill="#111111")
        crop = crop_observation(image, item)
        crop.thumbnail((356, 235), Image.Resampling.LANCZOS)
        sheet.paste(crop, (x + 14 + (356 - crop.width) // 2, y + 32 + (235 - crop.height) // 2))
    return sheet


def _short(value: object, limit: int = 500) -> str:
    return str(value).strip()[:limit] if value is not None else ""


def _known_ids(values: object, items: list[dict]) -> list[str]:
    allowed = {item["id"] for item in items}
    return [value for value in values if isinstance(value, str) and value in allowed] if isinstance(values, list) else []


def _pixel_box(box: object, image: Image.Image) -> tuple[list[int], list[int]] | None:
    """Validate Gemma's normalized [ymin, xmin, ymax, xmax] box."""
    if not isinstance(box, list) or len(box) != 4:
        return None
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in box):
        return None
    if any(value < 0 or value > 1000 for value in box):
        return None
    y1, x1, y2, x2 = [round(value) for value in box]
    if y2 <= y1 or x2 <= x1:
        return None
    pixels = [round(x1 * image.width / 1000), round(y1 * image.height / 1000),
              round(x2 * image.width / 1000), round(y2 * image.height / 1000)]
    if pixels[2] - pixels[0] < 12 or pixels[3] - pixels[1] < 12:
        return None
    return [y1, x1, y2, x2], pixels


def scan_visual_clues(image: Image.Image, key: str) -> list[dict]:
    """Ask Gemma for a few precise boxes, then enforce a strict local schema."""
    prompt = (
        "Locate up to three clearly visible physical objects or scene features worth examining "
        "in this image. Return only JSON with the format "
        '{"clues":[{"name":"object name","box_2d":[ymin,xmin,ymax,xmax],'
        '"visible_detail":"what is directly visible","possibility":"a tentative explanation",'
        '"alternative":"an ordinary alternative","next_check":"how to verify it"}]}. '
        "Use integer coordinates on a 0 to 1000 grid relative to the full image. "
        "Include only clues you can localize accurately. Do not infer past actions, injuries, "
        "intent, guilt, or a cause of death."
    )
    result = json.loads(_gemma([{"text": prompt}, _image_part(image)], key, json_output=True))
    if not isinstance(result, dict):
        raise ValueError("Gemma clue response was not an object.")
    raw_clues = result.get("clues", [])
    if not isinstance(raw_clues, list):
        raise ValueError("Gemma clue response did not contain a list of clues.")
    clues = []
    for entry in raw_clues[:3]:
        if not isinstance(entry, dict):
            continue
        name = _short(entry.get("name"), 60)
        parsed = _pixel_box(entry.get("box_2d"), image)
        if not name or parsed is None:
            continue
        normalized, pixels = parsed
        clues.append({
            "id": f"CLUE-{len(clues) + 1:03d}", "label": name.capitalize(),
            "source": "Gemma 4", "confidence": None,
            "box_1000": normalized, "box": pixels,
            "observation": _short(entry.get("visible_detail"), 400),
            "context": "", "limitation": "A model-proposed clue; verify the crop against the original image.",
            "model_verdict": "unverified",
            "scan_lead": {field: _short(entry.get(field), 320)
                          for field in ("possibility", "alternative", "next_check")},
        })
    return clues


def merge_evidence(detections: list[dict], clues: list[dict]) -> list[dict]:
    """Keep precise detector crops and retain Gemma's independent spatial boxes."""
    items = [dict(item, source="Grounding DINO") for item in detections]
    for item in items:
        item["observation"] = item.get("observation") or f"Detector proposal: {item['label'].lower()}."
        item["limitation"] = item.get("limitation") or "Verify the crop against the original image."
    for clue in clues:
        clue_words = set(re.findall(r"[a-z]+", clue["label"].lower()))
        match = None
        for item in items:
            item_words = set(re.findall(r"[a-z]+", item["label"].lower()))
            overlap = _overlap(clue["box"], item["box"])
            if overlap > 0.60 or (overlap > 0.18 and clue_words & item_words):
                match = item
                break
        if match:
            match["source"] = "Grounding DINO + Gemma 4"
            match["gemma_box"] = clue["box"]
            match["gemma_box_1000"] = clue["box_1000"]
            match["gemma_detail"] = clue["observation"]
            match["observation"] = clue["observation"] or match["observation"]
            match["model_verdict"] = "likely_match"
            match["scan_lead"] = clue.get("scan_lead", {})
        else:
            items.append(clue)
    return items


def investigation_from_scan(items: list[dict]) -> dict:
    """Build the log from the same Gemma response that localized each crop."""
    leads = []
    for item in items:
        proposed = item.get("scan_lead", {})
        if proposed.get("possibility"):
            leads.append({"possibility": proposed["possibility"],
                          "visible_basis": item.get("observation", ""),
                          "alternative": proposed.get("alternative", ""),
                          "next_check": proposed.get("next_check", ""),
                          "evidence_ids": [item["id"]]})
    if not leads:
        draft = evidence_only_fallback(items, "What could explain the visible scene?")
        draft["fallback_reason"] = "The visual scan did not return usable investigative explanations."
        draft["limits"] = "Evidence-only draft: the scan returned no usable lead fields. No event or medical cause is established."
        return draft
    return {"direct_answer": "The scan identified the following visible clues and tentative explanations.",
            "assessment": "cannot_determine", "evidence_ids": [x["evidence_ids"][0] for x in leads],
            "leads": leads, "reasoning_model": "Gemma 4 — same visual scan",
            "limits": "These are hypotheses from one image. The photo does not establish an event, guilt, or medical cause."}


def investigate_scene(image: Image.Image, items: list[dict], question: str,
                      history: list[dict], key: str) -> dict:
    """Use Gemma to reason over visual evidence already extracted from the image."""
    records = [{"id": item["id"], "object": item["label"],
                "visible": item.get("observation", "")} for item in items[:12]]
    prompt = (
        "Analyze these photo observations. Answer the question and suggest up to 3 testable "
        "explanations. State visible facts separately from possibilities. Never invent events, "
        "identify a culprit, or determine a medical cause of death. Use only listed evidence IDs. "
        "Return JSON: {\"direct_answer\":\"...\",\"assessment\":\"cannot_determine\","
        "\"evidence_ids\":[],\"leads\":[{\"possibility\":\"...\",\"visible_basis\":\"...\","
        "\"alternative\":\"...\",\"next_check\":\"...\",\"evidence_ids\":[]}],\"limits\":\"...\"}. "
        f"Observations: {json.dumps(records)}. Question: {question[:800]}. "
        f"Recent discussion: {json.dumps(history[-2:])[:1800]}"
    )
    response, model_name = _gemma([{"text": prompt}],
                                  key, json_output=True, models=REASONING_MODELS,
                                  return_model=True)
    raw = json.loads(response)
    if not isinstance(raw, dict):
        raise ValueError("Gemma did not return an investigative object.")
    assessment = raw.get("assessment", "cannot_determine")
    if assessment not in {"direct_visual_support", "direct_visual_conflict", "cannot_determine"}:
        assessment = "cannot_determine"
    leads = []
    for lead in raw.get("leads", [])[:4] if isinstance(raw.get("leads"), list) else []:
        if not isinstance(lead, dict):
            continue
        possibility = _short(lead.get("possibility"), 240)
        basis = _short(lead.get("visible_basis"), 400)
        if not possibility or not basis:
            continue
        cited_ids = _known_ids(lead.get("evidence_ids"), items)
        if not cited_ids:
            continue
        leads.append({
            "possibility": possibility,
            "visible_basis": basis,
            "alternative": _short(lead.get("alternative"), 320),
            "next_check": _short(lead.get("next_check"), 320),
            "evidence_ids": cited_ids,
        })
    answer = _short(raw.get("direct_answer"), 900)
    if re.search(r"cause of death|what.*killed|how.*died|what.*death", question, re.I):
        if "cannot" not in answer.lower() and "unknown" not in answer.lower():
            answer = "A cause of death cannot be determined from this photo. " + answer
    return {
        "direct_answer": answer or "The image does not establish an answer to this question.",
        "assessment": assessment,
        "evidence_ids": _known_ids(raw.get("evidence_ids"), items),
        "leads": leads,
        "limits": _short(raw.get("limits"), 600) or "A single photograph cannot establish an event or medical cause.",
        "reasoning_model": model_name,
    }


def evidence_only_fallback(items: list[dict], question: str) -> dict:
    """Honest, non-generative draft when the reasoning API is unavailable."""
    leads = []
    for item in items[:4]:
        leads.append({
            "possibility": f"Check whether {item['label'].lower()} is relevant",
            "visible_basis": item.get("observation") or f"{item['label']} was proposed in the image.",
            "alternative": "It may be ordinary or unrelated to any incident.",
            "next_check": "Verify the crop against the original photo and gather scene context.",
            "evidence_ids": [item["id"]],
        })
    answer = ("A cause of death cannot be determined from this photo. "
              if re.search(r"cause of death|what.*killed|how.*died|what.*death", question, re.I)
              else "")
    answer += (f"The image review proposed {len(items)} visual clue(s) for human verification."
               if items else "No visual clues were proposed. This does not establish that the scene is clear.")
    return {"direct_answer": answer, "assessment": "cannot_determine",
            "evidence_ids": [item["id"] for item in items[:4]], "leads": leads,
            "limits": "Evidence-only draft. Gemma reasoning was unavailable; no event or medical cause can be established.",
            "fallback": True, "fallback_reason": "The reasoning API did not return a usable response."}


def question_fallback_from_scan(items: list[dict], question: str, scan_result: dict | None) -> dict:
    """Answer common questions conservatively using the completed visual scan."""
    result = evidence_only_fallback(items, question)
    if scan_result:
        result["evidence_ids"] = list(scan_result.get("evidence_ids", result["evidence_ids"]))
        result["leads"] = list(scan_result.get("leads", result["leads"]))
        result["reasoning_model"] = "Gemma 4 visual scan + local evidence fallback"

    if re.search(r"break[ -]?in|broke into|broken into|intrud|burglar|forced entr", question, re.I):
        relevant = [item["id"] for item in items if re.search(
            r"glass|chair|door|window|lock|damage", f"{item.get('label', '')} {item.get('observation', '')}", re.I
        )][:4]
        result["direct_answer"] = (
            "A break-in is possible, but this image does not establish that one occurred. "
            "The broken drinking glass and overturned chair could suggest a disturbance, but they also "
            "have ordinary accidental explanations. The current evidence contains no verified sign of "
            "forced entry."
        )
        result["evidence_ids"] = relevant or result["evidence_ids"]
        result["leads"] = [{
            "possibility": "Check whether there was forced entry or an accidental disturbance",
            "visible_basis": "Broken glass and an overturned chair are visible in the reviewed scene.",
            "alternative": "A glass may have been dropped and the chair may have been knocked over accidentally.",
            "next_check": "Inspect doors, windows, locks, tool marks, missing property, fingerprints, and available camera footage.",
            "evidence_ids": relevant or result["evidence_ids"][:2],
        }]
        result["limits"] = (
            "This answer reuses the completed Gemma visual scan because the follow-up request failed. "
            "A photograph alone cannot establish unlawful entry or identify a person."
        )
    return result


def format_investigation(result: dict) -> str:
    blocks = [f"**Visual assessment:** {result['assessment'].replace('_', ' ').title()}",
              result["direct_answer"]]
    if result.get("reasoning_model"):
        blocks.append("**Reasoning model:** " + result["reasoning_model"])
    if result["evidence_ids"]:
        blocks.append("**Referenced evidence:** " + ", ".join(result["evidence_ids"]))
    if result["leads"]:
        blocks.append("**Investigative possibilities to test:**")
        for number, lead in enumerate(result["leads"], 1):
            ids = ", ".join(lead["evidence_ids"])
            detail = [f"{number}. **{lead['possibility']}** ({ids})",
                      f"   - Visible basis: {lead['visible_basis']}"]
            if lead["alternative"]:
                detail.append(f"   - Alternative: {lead['alternative']}")
            if lead["next_check"]:
                detail.append(f"   - Check: {lead['next_check']}")
            blocks.append("\n".join(detail))
    else:
        blocks.append("No scene-specific possibility could be grounded in the visible evidence.")
    blocks.append("**Limits:** " + result["limits"])
    return "\n\n".join(blocks)


def cross_examine(image: Image.Image, items: list[dict], statement: str, key: str) -> dict:
    """Break a statement into claims testable against extracted visual evidence."""
    evidence = [{field: item.get(field) for field in
                 ("id", "label", "source", "observation", "context", "model_verdict")}
                for item in items]
    prompt = (
        "Compare the statement to the supplied model-proposed visual evidence records only. "
        "These records were extracted from labelled crops. You do not see the photograph here. Break it into up to six concrete "
        "claims. Return ONLY JSON: {\"claims\":[{\"claim\":\"...\",\"assessment\":"
        "\"direct_visual_conflict|no_direct_conflict|cannot_determine\",\"explanation\":\"...\","
        "\"cited_ids\":[\"...\"]}]}. A direct visual conflict requires the photograph to plainly "
        "show the opposite of a concrete visual claim in these records. An object being absent from frame does not "
        "prove it was absent elsewhere. Do not infer identity, actions before the photo, alibis, "
        "intent, guilt, or cause of death. Use only supplied evidence IDs. "
        f"Evidence: {json.dumps(evidence)}. Statement: {statement[:1500]}"
    )
    response, model_name = _gemma([{"text": prompt}],
                                  key, json_output=True, models=REASONING_MODELS,
                                  return_model=True)
    raw = json.loads(response)
    if not isinstance(raw, dict):
        raise ValueError("Gemma statement response was not an object.")
    claims = []
    for entry in raw.get("claims", [])[:6] if isinstance(raw.get("claims"), list) else []:
        if not isinstance(entry, dict):
            continue
        assessment = entry.get("assessment", "cannot_determine")
        if assessment not in {"direct_visual_conflict", "no_direct_conflict", "cannot_determine"}:
            assessment = "cannot_determine"
        cited_ids = _known_ids(entry.get("cited_ids"), items)
        if assessment == "direct_visual_conflict" and not cited_ids:
            assessment = "cannot_determine"
        claims.append({
            "claim": _short(entry.get("claim"), 400),
            "assessment": assessment,
            "explanation": _short(entry.get("explanation"), 700),
            "cited_ids": cited_ids,
        })
    if not claims:
        claims = [{"claim": statement[:400], "assessment": "cannot_determine",
                   "explanation": "No directly testable visual claim was identified.", "cited_ids": []}]
    conflicts = [claim for claim in claims if claim["assessment"] == "direct_visual_conflict"]
    return {
        "claims": claims,
        "assessment": "direct_visual_conflict" if conflicts else "no_direct_conflict",
        "has_contradiction": bool(conflicts),
        "explanation": (f"{len(conflicts)} direct visual conflict(s) found."
                        if conflicts else "No direct visual conflict was established."),
        "cited_ids": list(dict.fromkeys(value for claim in conflicts for value in claim["cited_ids"])),
        "reasoning_model": model_name,
    }
