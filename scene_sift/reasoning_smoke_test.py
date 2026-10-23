"""Exercise the Gemma request and response path without an external API key."""

from unittest.mock import Mock, patch

from PIL import Image

from engine import ask_question, explain_scene, review_statement


image = Image.new("RGB", (64, 64), "white")
items = [{"id": "OBS-001", "label": "Knife", "box_1000": [0, 0, 500, 500]}]


def response(text):
    return Mock(ok=True, json=lambda: {"candidates": [{"content": {"parts": [{"text": text}]}}]})


with patch("engine.requests.post", return_value=response(
    '{"scene_summary":"A knife is visible.","items":[{"id":"OBS-001",'
    '"observation":"Knife-shaped object","context":"On a surface",'
    '"limitation":"No intent can be inferred","model_verdict":"likely_match"}]}'
)) as post:
    result = explain_scene(image, items, "test-key")
    assert result["items"][0]["model_verdict"] == "likely_match"
    request = post.call_args.kwargs
    assert request["headers"]["x-goog-api-key"] == "test-key"
    assert "inline_data" in request["json"]["contents"][0]["parts"][1]

with patch("engine.requests.post", return_value=response("OBS-001 is visible in the image.")):
    assert "OBS-001" in ask_question(image, items, "What is visible?", [], "test-key")

temporary_error = Mock(ok=False, status_code=500,
                       json=lambda: {"error": {"message": "Internal error encountered."}})
with patch("engine.requests.post", side_effect=[temporary_error, response("Retried successfully.")]) as post, \
     patch("engine.time.sleep"):
    assert ask_question(image, items, "What is visible?", [], "test-key") == "Retried successfully."
    assert post.call_count == 2

with patch("engine.requests.post", return_value=response(
    '{"assessment":"direct_visual_conflict","explanation":"A knife is visible.",'
    '"cited_ids":["OBS-001","OBS-999"]}'
)):
    review = review_statement(image, items, "No knife is visible.", "test-key")
    assert review["assessment"] == "direct_visual_conflict"
    assert review["cited_ids"] == ["OBS-001"]

print("Gemma request/response smoke test passed")
