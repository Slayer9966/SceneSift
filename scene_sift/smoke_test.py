"""Run the full local detector -> crop -> PDF path on the synthetic sample."""

from pathlib import Path

from engine import annotated_image, crop_observation, detect_objects, open_image
from report import build_report


ROOT = Path(__file__).resolve().parents[1]
sample = Path(__file__).parent / "demo" / "synthetic_kitchen_knife.png"
raw = sample.read_bytes()
image = open_image(raw)
items = detect_objects(image, threshold=0.25)
print("Detections:", [(item["label"], item["confidence"], item["box"]) for item in items])
assert any(item["label"] == "Knife" for item in items), "Synthetic sample did not produce a knife detection"
for item in items:
    crop = crop_observation(image, item)
    assert crop.width > 0 and crop.height > 0
output = ROOT / "output" / "pdf"
output.mkdir(parents=True, exist_ok=True)
annotated_image(image, items).save(output / "scenesift-sample-annotated.png")
(output / "scenesift-sample-review.pdf").write_bytes(
    build_report(raw, image, items, "Synthetic kitchen scene used to check detector behavior.",
                 statement="There is no knife in this photo.",
                 review={"claims": [{"claim": "No knife is visible",
                                     "assessment": "direct_visual_conflict",
                                     "explanation": "The detector proposed a knife; verify the crop.",
                                     "cited_ids": [items[0]["id"]]}]},
                 source_label="Synthetic demo",
                 leads={"direct_answer": "A kitchen knife is visible in a food preparation scene.",
                        "leads": [{"possibility": "Ordinary food preparation",
                                   "visible_basis": "The knife is on the work surface.",
                                   "alternative": "The photograph does not establish its use.",
                                   "next_check": "Ask for scene context and inspect the original photo.",
                                   "evidence_ids": [items[0]["id"]]}],
                        "limits": "No injury, death, or cause of death can be inferred."},
                 chat_history=[{"role": "user", "content": "What can this photo tell us?"},
                               {"role": "assistant", "content": "The knife is visible; no event is established."}]))
print("Wrote sample PNG and PDF to", output)
