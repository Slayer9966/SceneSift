# SceneSift

**AI-assisted incident photo review.** SceneSift combines the open-weight Grounding DINO detector with Gemma 4 vision reasoning. Grounding DINO searches for selected object types; Gemma proposes additional visual clues with normalized 2D boxes, writes testable investigative leads, answers follow-up hypotheses, and checks concrete claims in a statement. Every proposed clue has a crop and a human review control. The app exports a source-linked PDF brief.

This is a research and hackathon demo, not a real-world security, forensic, legal, or emergency decision system.

## Quick start on this computer (Windows PowerShell)

From the repository root:

```powershell
& 'C:\Users\faiza\anaconda3\python.exe' -m streamlit run app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`). This machine's cached checkpoint is at `C:\Detections\models\grounding-dino-base`; set `GROUNDING_DINO_PATH` to another local model folder on another computer. If that folder is absent, Transformers loads `IDEA-Research/grounding-dino-base` from Hugging Face. Put `GEMINI_API_KEY=...` in a local root `.env` file for Gemma 4 reasoning; the key field is not shown in the app. The `.env` file is Git-ignored. Detection, crops, human review, and PDF export work without a key. The existing Anaconda Python uses Pillow's PDF export fallback; ReportLab is used automatically when installed.

## Demo flow

1. Upload a JPG, PNG, or WebP incident photo, or select **Synthetic demo** in the sidebar.
   For a safe, synthetic first run, use [`demo/synthetic_kitchen_knife.png`](demo/synthetic_kitchen_knife.png). It contains a kitchen knife in ordinary food-preparation context.
2. Click **Run full scene scan**. Grounding DINO checks 27 object names across sharp implements, firearms, explosives, blunt implements, and other visible hazards. Add up to eight specific terms in **Additional objects**. Gemma proposes up to three additional localized visual clues.
   The same Gemma response supplies each clue's tentative explanation, alternative, and next check. The initial detective log does not require a second reasoning request. Follow-up questions and statement review reason over the extracted evidence records using Gemma 4; their actual model ID is included in the result.
3. Open **Visual Evidence Locker** to inspect amber detector and cyan Gemma boxes. Review each crop, tick **I reviewed this crop**, and add notes.
4. Open **Hypothesis Q&A** to read investigative possibilities and ask follow-up questions. Each accepted lead cites a known evidence ID, states its visible basis and an alternative, and suggests a check.
5. Open **Statement Cross-Examination** to compare up to six concrete claims with the photo. Direct visual conflicts cite evidence crops.
6. Download the **Investigative Brief** PDF with the original photo, annotated copy, evidence log, leads, recent Q&A, statement review, and SHA-256 hash of the uploaded bytes.

## Models and data flow

- **Detection:** [`IDEA-Research/grounding-dino-base`](https://huggingface.co/IDEA-Research/grounding-dino-base), loaded locally with Transformers from the user's existing checkpoint. The model card lists an Apache-2.0 license. Grounding DINO uses text prompts, so the app can seek specific object types without retraining.
- **Reasoning and spatial grounding:** [`gemma-4-31b-it`](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api), an open-weight multimodal model accessed through the Gemini API. Temporary API failures are retried, with `gemma-4-26b-a4b-it` as a final fallback. Gemma's normalized `[ymin, xmin, ymax, xmax]` boxes are validated and converted to pixel crops. Malformed boxes and unknown evidence IDs are discarded. The uploaded image and detection metadata are sent to that API only when a key is configured and a Gemma action is run.
- **Image crops and overlays:** Pillow. **PDF:** ReportLab when present, with a Pillow PDF fallback for the existing Anaconda environment.
- The app does not store uploaded images on disk. The browser session holds the image and results in memory. The original image bytes are hashed so a reviewer can identify the exact source file later.

## What the model can and cannot say

Detector boxes are proposals. A detector score is not the probability that an object is genuinely a threat. Many ordinary objects can cause harm in some circumstances, so no fixed prompt list can detect every potentially harmful object. The extra-term field lets a reviewer seek a specific object, but a missed object or an empty result does not mean a photo is safe.

Gemma is prompted to separate visible observations from possibilities. The app enforces the expected JSON fields, valid boxes, and known evidence citations locally, but the wording and interpretations still need human review. A single photo cannot establish intent, ownership, identity, a timeline, a medical cause of death, or whether something happened outside the frame. Statement review uses `direct_visual_conflict`, `no_direct_conflict`, or `cannot_determine`; it is not an alibi verifier. Human review is required before using or sharing the report.

## Checks

```powershell
& 'C:\Users\faiza\anaconda3\python.exe' -m unittest discover -s scene_sift -p 'test_*.py' -v
& 'C:\Users\faiza\anaconda3\python.exe' scene_sift\smoke_test.py
& 'C:\Users\faiza\anaconda3\python.exe' tmp\ui_regression.py
```

The first command verifies the local output guards. The smoke test runs the local Grounding DINO checkpoint and creates an annotated PNG and full PDF in `output/pdf`. The AppTest check exercises all four Streamlit tabs with deterministic Gemma replies. `tmp/live_investigation.py` makes a real API call when a key is configured.

## Files

- `engine.py`: model loading, detection, image crops, overlays, and Gemma API calls.
- `investigator.py`: spatial-box validation, clue merging, leads, and statement-claim checks.
- `app.py`: Streamlit upload and review workflow.
- `report.py`: PDF creation.

## License

Project source: MIT. Grounding DINO checkpoint: Apache-2.0 as listed by its model author. Gemma model use is governed by Google's Gemma terms. See the respective upstream repositories and model cards for the complete third-party terms.
