"""Compact, source-linked PDF review generated from in-memory evidence."""

from __future__ import annotations

import hashlib
import io
from datetime import datetime
from html import escape

from PIL import Image as PILImage, ImageDraw, ImageFont
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

from engine import annotated_image, crop_observation


def _image_flowable(image: PILImage.Image, max_w: float, max_h: float) -> Image:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    scale = min(max_w / image.width, max_h / image.height, 1)
    return Image(buffer, width=image.width * scale, height=image.height * scale)


def build_report(original_bytes: bytes, image: PILImage.Image, items: list[dict],
                 scene_summary: str = "", statement: str = "", review: dict | None = None,
                 source_label: str = "Uploaded photo", leads: dict | None = None,
                 chat_history: list[dict] | None = None) -> bytes:
    if not HAS_REPORTLAB:
        return _build_pillow_pdf(original_bytes, image, items, scene_summary, statement,
                                 review, source_label, leads, chat_history or [])
    output = io.BytesIO()
    doc = SimpleDocTemplate(output, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=18 * mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SceneTitle", parent=styles["Title"], fontSize=19,
                              leading=23, textColor=colors.HexColor("#112a4a"), spaceAfter=7))
    styles.add(ParagraphStyle(name="SceneSmall", parent=styles["BodyText"], fontSize=8,
                              leading=11, textColor=colors.HexColor("#516071")))
    styles.add(ParagraphStyle(name="SceneBody", parent=styles["BodyText"], fontSize=9,
                              leading=13, spaceAfter=5))
    styles.add(ParagraphStyle(name="SceneHeading", parent=styles["Heading2"], fontSize=11,
                              leading=14, textColor=colors.HexColor("#112a4a"), spaceBefore=12))
    story = [Paragraph("SceneSift - AI-Assisted Visual Review", styles["SceneTitle"]),
             Paragraph("Human review required. This document records model suggestions and reviewer choices; "
                       "it is not a forensic finding or an official police report.", styles["SceneSmall"]),
             Spacer(1, 6 * mm)]
    digest = hashlib.sha256(original_bytes).hexdigest()
    metadata = [
        ["Generated", datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")],
        ["Photo source", source_label],
        ["Original image SHA-256", digest],
        ["Detection model", "IDEA-Research/grounding-dino-base"],
        ["Reasoning model", (leads or {}).get("reasoning_model", "Not recorded")],
        ["Image size", f"{image.width} x {image.height} pixels"],
    ]
    table = Table([[Paragraph(escape(k), styles["SceneSmall"]),
                    Paragraph(escape(v), styles["SceneSmall"])] for k, v in metadata],
                  colWidths=[36 * mm, 138 * mm])
    table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.whitesmoke, colors.white]),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story += [table, Paragraph("Original uploaded scene", styles["SceneHeading"]),
              _image_flowable(image, 174 * mm, 85 * mm),
              Paragraph("Annotated review copy", styles["SceneHeading"]),
              _image_flowable(annotated_image(image, items), 174 * mm, 85 * mm)]
    if scene_summary:
        story += [Paragraph("Scene summary", styles["SceneHeading"]),
                  Paragraph(escape(scene_summary), styles["SceneBody"])]
    story.append(Paragraph("Observation log", styles["SceneHeading"]))
    if not items:
        story.append(Paragraph("No target objects were detected above the selected threshold. "
                               "This does not establish that none are present.", styles["SceneBody"]))
    for item in items:
        crop = crop_observation(image, item)
        score = item.get("confidence")
        heading = f"{item['id']} - {item['label']}" + (f" ({score:.0%} detector score)" if score is not None else "")
        parts = [Paragraph(escape(heading), styles["SceneHeading"]),
                 Paragraph("Source: " + escape(item.get("source", "Model proposal")), styles["SceneSmall"]),
                 Paragraph("Gemma visual check: " + escape(item.get("model_verdict", "unverified").replace("_", " ")),
                           styles["SceneSmall"]),
                 Paragraph("Reviewer status: " + ("confirmed for review" if item.get("confirmed") else "not confirmed"),
                           styles["SceneSmall"]),
                 _image_flowable(crop, 65 * mm, 46 * mm)]
        for title, field in (("Visible observation", "observation"), ("Context", "context"),
                             ("Limit", "limitation"), ("Reviewer note", "note")):
            value = item.get(field)
            if value:
                parts.append(Paragraph(f"<b>{title}:</b> {escape(str(value))}", styles["SceneBody"]))
        story.extend(parts)
    if leads:
        story += [Paragraph("Investigative leads (unverified)", styles["SceneHeading"]),
                  Paragraph(escape(leads.get("direct_answer", "")), styles["SceneBody"])]
        for number, lead in enumerate(leads.get("leads", []), 1):
            story.append(Paragraph(f"<b>{number}. {escape(lead.get('possibility', ''))}</b>", styles["SceneBody"]))
            for title, field in (("Visible basis", "visible_basis"), ("Alternative", "alternative"),
                                 ("Check", "next_check")):
                value = lead.get(field)
                if value:
                    story.append(Paragraph(f"<b>{title}:</b> {escape(str(value))}", styles["SceneBody"]))
            story.append(Paragraph("Evidence IDs: " + escape(", ".join(lead.get("evidence_ids", [])) or "None"),
                                   styles["SceneSmall"]))
        story.append(Paragraph("Limits: " + escape(leads.get("limits", "")), styles["SceneSmall"]))
    if chat_history:
        story.append(Paragraph("Hypothesis Q&A", styles["SceneHeading"]))
        for message in chat_history[-4:]:
            speaker = "Question" if message.get("role") == "user" else "AI review"
            story.append(Paragraph(f"<b>{speaker}:</b> {escape(str(message.get('content', ''))[:900])}",
                                   styles["SceneBody"]))
    if statement:
        story += [Paragraph("Statement cross-examination", styles["SceneHeading"]),
                  Paragraph("Statement: " + escape(statement), styles["SceneBody"])]
        if review:
            for claim in review.get("claims", []):
                story.append(Paragraph("Claim: " + escape(claim.get("claim", "")), styles["SceneBody"]))
                story.append(Paragraph("Visual assessment: " + escape(claim.get("assessment", "cannot_determine").replace("_", " ")),
                                       styles["SceneBody"]))
                story.append(Paragraph(escape(claim.get("explanation", "")), styles["SceneBody"]))
                story.append(Paragraph("Evidence IDs: " + escape(", ".join(claim.get("cited_ids", [])) or "None"),
                                       styles["SceneSmall"]))
    story += [Spacer(1, 7 * mm), Paragraph(
        "Method note: A single photograph cannot establish intent, ownership, a sequence of events, "
        "or absence of an object outside the visible frame. Detection scores are model outputs, not "
        "probabilities that a claim is true. Preserve the original photo separately.", styles["SceneSmall"])]

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#687786"))
        canvas.drawString(18 * mm, 10 * mm, "SceneSift | AI-assisted visual review")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, str(document.page))
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return output.getvalue()


def _build_pillow_pdf(original_bytes: bytes, image: PILImage.Image, items: list[dict],
                      scene_summary: str, statement: str, review: dict | None,
                      source_label: str, leads: dict | None,
                      chat_history: list[dict]) -> bytes:
    """Portable image-based PDF for existing Python installs without ReportLab."""
    page_w, page_h = 1240, 1754
    pages = []
    color = "#102a43"

    def font(size: int, bold: bool = False):
        candidates = [r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
                      "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
        for path in candidates:
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
        return ImageFont.load_default()

    body = font(20)
    small = font(16)
    heading = font(27, True)
    title = font(38, True)

    def new_page(section: str):
        page = PILImage.new("RGB", (page_w, page_h), "white")
        draw = ImageDraw.Draw(page)
        draw.text((65, 53), "SceneSift", fill=color, font=title)
        draw.text((65, 111), section, fill="#415a73", font=heading)
        draw.line((65, 150, 1175, 150), fill="#b5c5d3", width=2)
        pages.append(page)
        return page, draw

    def wrapped(draw, value: str, x: int, y: int, max_width: int,
                chosen_font=None, fill="#243b53", line_h=29):
        chosen_font = chosen_font or body
        words = str(value).split()
        line = ""
        for word in words:
            candidate = f"{line} {word}".strip()
            if draw.textlength(candidate, font=chosen_font) > max_width and line:
                draw.text((x, y), line, fill=fill, font=chosen_font)
                y += line_h
                line = word
            else:
                line = candidate
        if line:
            draw.text((x, y), line, fill=fill, font=chosen_font)
            y += line_h
        return y

    def paste_fit(page, source, x, y, max_w, max_h):
        copy = source.copy()
        copy.thumbnail((max_w, max_h), PILImage.Resampling.LANCZOS)
        page.paste(copy.convert("RGB"), (x + (max_w - copy.width) // 2, y))
        return copy.height

    page, draw = new_page("AI-Assisted Visual Review")
    y = wrapped(draw, "Human review required. Model suggestions are not forensic findings or an official police report.",
                65, 180, 1110, body, "#9b3f32") + 12
    for line in [f"Photo source: {source_label}",
                 f"Generated: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}",
                 f"Original SHA-256: {hashlib.sha256(original_bytes).hexdigest()}",
                 "Detector: IDEA-Research/grounding-dino-base | Reasoning: " +
                 (leads or {}).get("reasoning_model", "Not recorded")]:
        draw.text((65, y), line, fill="#415a73", font=small)
        y += 27
    y += 18
    draw.text((65, y), "Original uploaded scene", fill=color, font=heading)
    y += 42
    y += paste_fit(page, image, 65, y, 1110, 480) + 31
    draw.text((65, y), "Annotated review copy", fill=color, font=heading)
    y += 42
    paste_fit(page, annotated_image(image, items), 65, y, 1110, 480)

    page, draw = new_page("Observation log")
    y = 185
    if scene_summary:
        y = wrapped(draw, "Gemma scene summary: " + scene_summary, 65, y, 1110) + 20
    if not items:
        y = wrapped(draw, "No target object was detected above the selected threshold. This does not prove the scene is clear.",
                    65, y, 1110)
    for item in items:
        if y > 1320:
            page, draw = new_page("Observation log (continued)")
            y = 185
        draw.rounded_rectangle((65, y, 1175, y + 350), radius=14, fill="#f3f6f9", outline="#ccd8e3", width=2)
        score = item.get("confidence")
        heading_text = f"{item['id']}  {item['label']}" + (f"  {score:.0%}" if score is not None else "")
        draw.text((86, y + 20), heading_text, fill=color, font=heading)
        paste_fit(page, crop_observation(image, item), 86, y + 70, 345, 240)
        text_y = y + 74
        for name, value in [
            ("Source", item.get("source", "Model proposal")),
            ("Gemma check", item.get("model_verdict", "unverified").replace("_", " ")),
            ("Reviewer", "reviewed" if item.get("confirmed") else "not confirmed"),
            ("Visible", item.get("observation", "")),
            ("Context", item.get("context", "")),
            ("Limit", item.get("limitation", "")),
            ("Note", item.get("note", "")),
        ]:
            if value:
                text_y = wrapped(draw, f"{name}: {value}", 460, text_y, 680, small, line_h=23) + 5
                if text_y > y + 320:
                    break
        y += 377
    if leads:
        page, draw = new_page("Investigative leads")
        y = wrapped(draw, leads.get("direct_answer", ""), 65, 185, 1110) + 20
        for number, lead in enumerate(leads.get("leads", []), 1):
            if y > 1320:
                page, draw = new_page("Investigative leads (continued)")
                y = 185
            y = wrapped(draw, f"{number}. {lead.get('possibility', '')}", 65, y, 1110, heading, color, 38)
            for label, field in (("Visible basis", "visible_basis"), ("Alternative", "alternative"),
                                 ("Check", "next_check")):
                value = lead.get(field)
                if value:
                    y = wrapped(draw, f"{label}: {value}", 85, y + 6, 1090) + 4
            y = wrapped(draw, "Evidence IDs: " + (", ".join(lead.get("evidence_ids", [])) or "None"),
                        85, y + 6, 1090, small, line_h=23) + 22
        wrapped(draw, "Limits: " + leads.get("limits", ""), 65, min(y + 12, 1530), 1110, small, line_h=23)
    if chat_history:
        page, draw = new_page("Hypothesis Q&A")
        y = 185
        for message in chat_history[-4:]:
            if y > 1350:
                page, draw = new_page("Hypothesis Q&A (continued)")
                y = 185
            speaker = "Question" if message.get("role") == "user" else "AI review"
            y = wrapped(draw, f"{speaker}: {str(message.get('content', ''))[:700]}", 65, y, 1110) + 25
    if statement:
        page, draw = new_page("Statement cross-examination")
        y = wrapped(draw, "Statement: " + statement[:1500], 65, 185, 1110) + 24
        if review:
            for claim in review.get("claims", []):
                if y > 1350:
                    page, draw = new_page("Statement review (continued)")
                    y = 185
                y = wrapped(draw, "Claim: " + claim.get("claim", ""), 65, y, 1110, heading, color, 38)
                y = wrapped(draw, "Assessment: " + claim.get("assessment", "cannot_determine").replace("_", " "),
                            85, y + 5, 1090)
                y = wrapped(draw, claim.get("explanation", ""), 85, y + 5, 1090)
                y = wrapped(draw, "Evidence IDs: " + (", ".join(claim.get("cited_ids", [])) or "None"),
                            85, y + 5, 1090, small, line_h=23) + 25
    for number, (page, draw) in enumerate([(p, ImageDraw.Draw(p)) for p in pages], start=1):
        draw.line((65, 1685, 1175, 1685), fill="#b5c5d3", width=2)
        draw.text((65, 1700), "SceneSift | AI-assisted visual review | Human verification required", fill="#62798d", font=small)
        draw.text((1120, 1700), str(number), fill="#62798d", font=small)
    output = io.BytesIO()
    pages[0].save(output, format="PDF", save_all=True, append_images=pages[1:], resolution=150)
    return output.getvalue()
