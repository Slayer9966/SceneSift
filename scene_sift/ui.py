"""SceneSift Streamlit interface, launched through the repository's root app.py."""

from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st

from engine import (DETECTION_GROUPS, GEMMA_MODEL, MODEL_REPO, annotated_image,
                    configured_api_key, crop_observation, detect_objects, open_image)
from investigator import (cross_examine, evidence_only_fallback, format_investigation, investigate_scene,
                          investigation_from_scan, merge_evidence, question_fallback_from_scan,
                          scan_visual_clues)
from report import build_report


def cached_demo_clues(image):
    """Previously generated Gemma output for the bundled scene, used only if its API is down."""
    specifications = [
        ("Knife", [445, 58, 692, 235], "A chef's knife and sliced plums are visible on a cutting board.",
         "Check whether the knife is relevant to the scene.",
         "It may simply have been used for ordinary food preparation.",
         "Inspect it and compare its location with other verified scene evidence."),
        ("Broken glass and red liquid", [477, 437, 800, 742],
         "Shards of clear glass and a pool of red liquid are visible on the floor.",
         "Check whether the broken glass and spill indicate a disturbance.",
         "A drinking glass may have been dropped accidentally.",
         "Identify the liquid and examine the glass pattern and nearby surfaces."),
        ("Overturned chair", [220, 645, 776, 980],
         "A wooden chair is lying on its side beside the table.",
         "Check whether the chair's position is connected to the other visible clues.",
         "The chair may have been moved or knocked over during normal activity.",
         "Document its position and check for contact marks or other physical evidence."),
    ]
    clues = []
    for index, (name, box, detail, possibility, alternative, next_check) in enumerate(specifications, 1):
        y1, x1, y2, x2 = box
        clues.append({
            "id": f"CLUE-{index:03d}", "label": name,
            "source": "Gemma 4 (cached demo result)", "confidence": None,
            "box_1000": box,
            "box": [round(x1 * image.width / 1000), round(y1 * image.height / 1000),
                    round(x2 * image.width / 1000), round(y2 * image.height / 1000)],
            "observation": detail, "context": "",
            "limitation": "Precomputed Gemma output for the bundled demo image; verify the crop.",
            "model_verdict": "unverified",
            "scan_lead": {"possibility": possibility, "alternative": alternative,
                          "next_check": next_check},
        })
    return clues


st.set_page_config(page_title="SceneSift | Visual Investigator", page_icon="🔎", layout="wide")
st.title("🔎 SceneSift")
st.caption("Visual evidence locker · investigative leads · hypothesis testing · statement cross-examination")
st.info("Research demo. A human must verify each detection and interpretation. Do not use this app to make safety, legal, or guilt decisions.")

api_key = configured_api_key()
with st.sidebar:
    st.header("Inputs")
    source = st.radio("Photo source", ["Upload", "Synthetic demo"], horizontal=True)
    upload = st.file_uploader("Incident photo", type=["png", "jpg", "jpeg", "webp"]) if source == "Upload" else None
    threshold = st.slider("Detector threshold", 0.15, 0.80, 0.30, 0.05)
    extra_terms = st.text_input("Additional objects (comma separated)",
                                placeholder="e.g. broken bottle, utility blade",
                                help="Search up to eight extra object names with Grounding DINO.")
    analyze = st.button("🔍 Run full scene scan", type="primary", use_container_width=True,
                        disabled=source == "Upload" and upload is None)
    st.caption(f"Detector: {MODEL_REPO}\n\nReasoning: {GEMMA_MODEL}")
    st.caption(f"Broad sweep: {sum(len(group) for group in DETECTION_GROUPS)} object names across sharp objects, firearms, explosives, and other visible hazards. Add specific terms above when needed.")
    st.caption("Gemma 4 key: configured locally" if api_key else "Gemma 4 key: missing from local .env")

raw = (Path(__file__).parent / "demo" / "synthetic_incident_scene.png").read_bytes() if source == "Synthetic demo" else (upload.getvalue() if upload is not None else None)
if raw is None:
    st.session_state.pop("case", None)
if raw is not None:
    digest = hashlib.sha256(raw).hexdigest()
    if "case" in st.session_state and st.session_state.case["digest"] != digest:
        del st.session_state.case
        st.session_state.pop("messages", None)
        st.session_state.pop("statement_review", None)

if analyze:
    try:
        image = open_image(raw)
        progress = st.progress(0, text="Searching the image with Grounding DINO...")
        detections = detect_objects(image, threshold, extra_terms)
        progress.progress(35, text="Grounding DINO sweep complete. Asking Gemma 4 to localize visual clues...")
        items = merge_evidence(detections, [])
        gemma_errors = []
        demo_fallback = False
        leads = None
        if api_key:
            try:
                clues = scan_visual_clues(image, api_key)
                items = merge_evidence(detections, clues)
            except Exception as exc:
                if source == "Synthetic demo":
                    items = merge_evidence(detections, cached_demo_clues(image))
                    demo_fallback = True
                else:
                    gemma_errors.append("Spatial clues: " + str(exc))
            progress.progress(70, text="Building the detective log from the visual scan...")
            leads = investigation_from_scan(items)
        progress.progress(100, text="Scene scan complete.")
        progress.empty()
        st.session_state.case = {"raw": raw, "digest": digest, "image": image,
                                 "items": items, "leads": leads,
                                 "gemma_errors": gemma_errors,
                                 "demo_fallback": demo_fallback,
                                 "source_label": "Synthetic demo" if source == "Synthetic demo" else "Uploaded photo"}
        st.session_state.messages = []
        st.session_state.pop("statement_review", None)
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")

case = st.session_state.get("case")
if case is None:
    st.write("Upload a photo or select the synthetic demo, then choose **Run full scene scan**.")
    st.stop()

image = case["image"]
items = case["items"]
st.caption(f"Original image SHA-256: `{case['digest']}`")
if case.get("demo_fallback"):
    st.info("Demo continuity mode: Google returned an API error, so this bundled scene is showing saved output from a previous successful Gemma scan.")
if case.get("gemma_errors"):
    for error in case["gemma_errors"]:
        st.warning(error)
elif not api_key:
    st.warning("Grounding DINO detection is available now. Set GEMINI_API_KEY in the local .env file for Gemma clues and reasoning.")

detector_count = sum("Grounding DINO" in item.get("source", "") for item in items)
gemma_count = sum("Gemma 4" in item.get("source", "") for item in items)
metric_evidence, metric_detector, metric_gemma = st.columns(3)
metric_evidence.metric("Evidence cards", len(items))
metric_detector.metric("Detector matches", detector_count)
metric_gemma.metric("Gemma grounded clues", gemma_count)

tab_scene, tab_questions, tab_statement, tab_export = st.tabs(
    ["Visual Evidence Locker", "Hypothesis Q&A", "Statement Cross-Examination", "Investigative Brief"]
)

with tab_scene:
    col_image, col_items = st.columns([1.5, 1], gap="large")
    with col_image:
        st.image(annotated_image(image, items), caption="Amber: Grounding DINO · Cyan: Gemma 4 spatial grounding. Boxes are proposals for human review.",
                 use_column_width=True)
        if case.get("leads"):
            st.subheader("Detective log")
            st.write(case["leads"]["direct_answer"])
            st.caption("Open Hypothesis Q&A for testable leads, alternatives, and next checks.")
    with col_items:
        st.subheader(f"Evidence cards ({len(items)})")
        if not items:
            st.write("No target object or grounded clue was found. This does not prove the scene is clear.")
        for item in items:
            with st.container(border=True):
                st.markdown(f"**{item['id']} · {item['label']}**")
                score = item.get("confidence")
                score_text = f" · detector score {score:.0%}" if score is not None else ""
                st.caption(item.get("source", "Model proposal") + score_text)
                verdict = item.get("model_verdict", "unverified").replace("_", " ")
                st.image(crop_observation(image, item), use_column_width=True)
                if item.get("observation"):
                    st.write("**Visible:**", item["observation"])
                if item.get("context"):
                    st.write("**Context:**", item["context"])
                if item.get("gemma_box_1000"):
                    st.caption("Gemma 2D box [ymin, xmin, ymax, xmax]: " + str(item["gemma_box_1000"]))
                elif item.get("source") == "Gemma 4":
                    st.caption("Gemma 2D box [ymin, xmin, ymax, xmax]: " + str(item["box_1000"]))
                st.caption("Model visual check: " + verdict)
                if item.get("limitation"):
                    st.caption("Limit: " + item["limitation"])
                item["confirmed"] = st.checkbox("I reviewed this crop", key=f"confirm_{case['digest']}_{item['id']}")
                item["note"] = st.text_input("Reviewer note (optional)", key=f"note_{case['digest']}_{item['id']}")

with tab_questions:
    st.subheader("Investigative leads")
    st.caption("Scene-supported possibilities, ordinary alternatives, and checks that could test them. This is not an exhaustive reconstruction.")
    if case.get("leads"):
        if case["leads"].get("fallback"):
            st.info("Evidence-only draft: " + case["leads"].get("fallback_reason", "No usable model explanation was returned."))
        st.markdown(format_investigation(case["leads"]))
    elif api_key:
        st.info("The automatic lead generation did not complete. You can retry it here.")
    if st.button("Regenerate investigative leads", disabled=not bool(api_key)):
        try:
            with st.spinner("Gemma is building the investigative log..."):
                case["leads"] = investigate_scene(image, items,
                    "What scene-supported investigative possibilities should be checked?", [], api_key)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    st.divider()
    st.subheader("Hypothesis Q&A")
    st.caption("Try a theory or ask what could explain a visible clue. Gemma compares it with the evidence IDs and keeps unknowns open.")
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    question = st.chat_input("What theory should the evidence test?", disabled=not bool(api_key))
    if question:
        history = st.session_state.get("messages", [])
        normalized_question = " ".join(question.lower().split())
        broad_scene_question = any(
            phrase in normalized_question
            for phrase in (
                "what does this image", "what this image", "what can this image",
                "what does the image", "what can the image", "describe this image",
                "describe the image", "tell me about this image",
            )
        )
        if broad_scene_question and case.get("leads"):
            answer = "Based on the successful Gemma visual scan:\n\n" + format_investigation(case["leads"])
            st.session_state.messages = history + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
            st.rerun()
        try:
            with st.spinner("Gemma is cross-checking the theory against visual evidence..."):
                result = investigate_scene(image, items, question, history, api_key)
                answer = format_investigation(result)
            st.session_state.messages = history + [{"role": "user", "content": question},
                                                   {"role": "assistant", "content": answer}]
            st.rerun()
        except Exception as exc:
            if case.get("leads"):
                fallback_result = question_fallback_from_scan(items, question, case.get("leads"))
                answer = (
                    "The follow-up request was unavailable, so this answer uses the successful "
                    "Gemma visual scan:\n\n"
                    + format_investigation(fallback_result)
                )
            else:
                answer = (
                    "Gemma API unavailable; evidence-only draft.\n\n"
                    + format_investigation(evidence_only_fallback(items, question))
                )
            st.session_state.messages = history + [{"role": "user", "content": question},
                                                   {"role": "assistant", "content": answer}]
            st.warning("Gemma follow-up request failed; preserving the completed visual scan. " + str(exc))
            st.rerun()
    if not api_key:
        st.info("Set GEMINI_API_KEY in the local .env file to use Gemma 4.")

with tab_statement:
    st.subheader("Statement cross-examination")
    st.caption("Gemma separates concrete claims and flags only direct visual conflicts. It cannot verify an alibi or infer actions outside the frame.")
    statement = st.text_area("Statement", placeholder="Example: There is no knife visible on the table.",
                             height=120, key=f"statement_{case['digest']}")
    if st.button("🚩 Run contradiction detector", disabled=not (statement.strip() and api_key)):
        st.session_state.pop("statement_review", None)
        try:
            with st.spinner("Gemma is checking each claim against the image..."):
                result = cross_examine(image, items, statement.strip(), api_key)
            st.session_state.statement_review = {"statement": statement.strip(), "result": result}
        except Exception as exc:
            st.error(str(exc))
    saved_review = st.session_state.get("statement_review")
    if saved_review:
        result = saved_review["result"]
        if result.get("reasoning_model"):
            st.caption("Reasoning model: " + result["reasoning_model"])
        conflict_count = sum(claim["assessment"] == "direct_visual_conflict" for claim in result["claims"])
        st.metric("Direct visual conflicts", f"{conflict_count} / {len(result['claims'])} claims")
        for claim in result["claims"]:
            with st.container(border=True):
                icon = "🔴" if claim["assessment"] == "direct_visual_conflict" else "⚪"
                st.markdown(f"{icon} **{claim['assessment'].replace('_', ' ').title()}** · {claim['claim']}")
                st.write(claim["explanation"])
                for evidence_id in claim["cited_ids"]:
                    item = next((candidate for candidate in items if candidate["id"] == evidence_id), None)
                    if item:
                        st.caption(f"{evidence_id} · {item['label']} · {item.get('source', 'model proposal')}")
                        st.image(crop_observation(image, item), width=230)

with tab_export:
    st.subheader("Download the investigative brief")
    st.write("The brief includes the annotated scene, evidence crops, investigative leads, statement comparison, and a hash of the original upload.")
    saved_review = st.session_state.get("statement_review")
    try:
        pdf = build_report(case["raw"], image, items,
                           case.get("leads", {}).get("direct_answer", "") if case.get("leads") else "",
                           saved_review["statement"] if saved_review else "",
                           saved_review["result"] if saved_review else None,
                           case["source_label"], leads=case.get("leads"),
                           chat_history=st.session_state.get("messages", []))
        st.download_button("📄 Download investigative brief", data=pdf, file_name="scenesift-investigative-brief.pdf",
                           mime="application/pdf", type="primary")
    except Exception as exc:
        st.error(f"Could not generate the report: {exc}")
