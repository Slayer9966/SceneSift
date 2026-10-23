"""Small acceptance checks for model output validation."""

import json
import unittest
from unittest.mock import patch

from PIL import Image

from investigator import (cross_examine, investigate_scene, question_fallback_from_scan,
                          scan_visual_clues)


class InvestigatorValidationTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (1000, 800), "white")
        self.items = [{"id": "OBS-001", "label": "Knife", "source": "Grounding DINO",
                       "box": [100, 100, 400, 500], "observation": "A knife is visible."}]

    def test_rejects_bad_coordinates_and_preserves_good_crop(self):
        response = {"clues": [
            {"name": "bad", "box_2d": [-2, 20, 200, 200]},
            {"name": "reversed", "box_2d": [500, 200, 100, 300]},
            {"name": "knife", "box_2d": [125, 100, 625, 400],
             "visible_detail": "A knife is visible."},
        ]}
        with patch("investigator._gemma", return_value=json.dumps(response)):
            clues = scan_visual_clues(self.image, "test")
        self.assertEqual(len(clues), 1)
        self.assertEqual(clues[0]["box"], [100, 100, 400, 500])

    def test_leads_require_known_visual_evidence(self):
        response = {"direct_answer": "A cause of death is certain.",
                    "assessment": "direct_visual_support", "evidence_ids": ["FAKE"],
                    "leads": [
                        {"possibility": "Grounded check", "visible_basis": "Knife in frame",
                         "evidence_ids": ["OBS-001", "FAKE"]},
                        {"possibility": "Unsupported", "visible_basis": "Nothing visible",
                         "evidence_ids": ["FAKE"]}], "limits": "Photo only."}
        with patch("investigator._gemma", return_value=(json.dumps(response), "gemma-4-31b-it")):
            result = investigate_scene(self.image, self.items,
                                       "What can be the cause of death?", [], "test")
        self.assertIn("cannot be determined", result["direct_answer"])
        self.assertEqual(len(result["leads"]), 1)
        self.assertEqual(result["leads"][0]["evidence_ids"], ["OBS-001"])
        self.assertEqual(result["evidence_ids"], [])

    def test_uncited_conflict_is_not_reported_as_fact(self):
        response = {"claims": [{"claim": "No knife is visible",
                                "assessment": "direct_visual_conflict",
                                "explanation": "A knife seems visible", "cited_ids": ["FAKE"]}]}
        with patch("investigator._gemma", return_value=(json.dumps(response), "gemma-4-31b-it")):
            result = cross_examine(self.image, self.items, "No knife is visible", "test")
        self.assertFalse(result["has_contradiction"])
        self.assertEqual(result["claims"][0]["assessment"], "cannot_determine")

    def test_break_in_fallback_answers_the_question_without_claiming_an_event(self):
        items = self.items + [{"id": "CLUE-001", "label": "Broken glass",
                               "observation": "Broken drinking glass is visible on the floor."}]
        result = question_fallback_from_scan(items, "Could someone have broken into the house?", None)
        self.assertIn("possible", result["direct_answer"].lower())
        self.assertIn("does not establish", result["direct_answer"].lower())
        self.assertIn("CLUE-001", result["evidence_ids"])


if __name__ == "__main__":
    unittest.main()
