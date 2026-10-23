# 🔎 SceneSift — AI-Assisted Visual Investigator

SceneSift is an **open-source AI visual investigation system** that turns a single uploaded photograph into reviewable evidence cards, spatial annotations, investigative possibilities, grounded question answering, statement comparison, and an exportable PDF brief.

The system combines **Grounding DINO** for open-vocabulary object detection with **Gemma 4** for spatial grounding and evidence-based reasoning. Its goal is to help a human notice and organize visible clues while clearly separating observations from unverified possibilities.

> [!IMPORTANT]
> SceneSift is a research and hackathon demonstration. Its outputs are model suggestions for human review, not forensic findings, medical conclusions, safety decisions, or proof of guilt.

---

## 🔍 Key Idea

The investigation pipeline is composed of two complementary AI components:

### 1. Open-Vocabulary Object Detection

* Grounding DINO searches for sharp objects, firearms, broken glass, tools, chemical containers, and custom user-defined objects.
* Each detection receives a bounding box, confidence score, evidence ID, and image crop.
* The detector runs from an open-weight local checkpoint.

### 2. Gemma 4 Visual Grounding and Reasoning

* Gemma 4 proposes additional visible clues using normalized 2D bounding boxes.
* The same response supplies a tentative explanation, an ordinary alternative, and a suggested next check.
* Follow-up questions and statements are checked against known evidence IDs.

**All model boxes and citations are validated locally before they are shown to the reviewer.**

---

## 🧠 AI Architecture

* **Object Detector:** `IDEA-Research/grounding-dino-base`
* **Primary Vision Model:** `gemma-4-31b-it`
* **Gemma Fallback:** `gemma-4-26b-a4b-it`
* **Spatial Format:** Normalized `[ymin, xmin, ymax, xmax]` coordinates on a 0–1000 grid
* **Image Processing:** Pillow crops, annotations, and contact images
* **Interface:** Streamlit
* **Report Generation:** ReportLab with a Pillow PDF fallback
* **Output Validation:** Strict JSON parsing, coordinate checks, and evidence-ID filtering

---

## 📂 Project Structure

```text
.
├── app.py                         # Streamlit entry point
├── requirements.txt               # Root dependency file
├── .env.example                   # API-key template
├── LICENSE                        # MIT License
│
└── scene_sift/
    ├── ui.py                      # Main application workflow
    ├── engine.py                  # Grounding DINO and Gemma API integration
    ├── investigator.py            # Evidence validation and reasoning logic
    ├── report.py                  # Investigative PDF generation
    ├── smoke_test.py              # Detector, crop, and PDF smoke test
    ├── test_investigator.py       # Output-validation tests
    └── demo/
        ├── synthetic_kitchen_knife.png
        └── synthetic_incident_scene.png
```

---

## ⚙️ Requirements

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

The project requires:

* Python 3.10 or newer
* PyTorch and Transformers
* Streamlit
* Pillow and Requests
* A Google AI Studio API key for Gemma 4
* A Grounding DINO checkpoint or internet access for its first download

> [!NOTE]
> CUDA is recommended for faster detection, but the detector can run on CPU.

---

## 🔐 Configuration

Create a `.env` file in the repository root:

```env
GEMINI_API_KEY=your-google-ai-studio-key
```

The `.env` file is excluded by `.gitignore` and must never be uploaded to GitHub.

The default local Grounding DINO path is:

```text
C:\Detections\models\grounding-dino-base
```

To use another location, set `GROUNDING_DINO_PATH` before launching the app:

```powershell
$env:GROUNDING_DINO_PATH = "C:\path\to\grounding-dino-base"
```

---

## 🚀 Run the Application

From the project root, run:

```bash
python -m streamlit run app.py
```

On the original development machine:

```powershell
cd "C:\Users\faiza\OneDrive\Documents\ChatGPT\Hackathon"
& "C:\Users\faiza\anaconda3\python.exe" -m streamlit run app.py
```

Open `http://localhost:8501` in a browser. Keep the terminal open while testing. Press `Ctrl+C` to stop the server.

---

## 🧪 Demo Workflow

1. Upload a JPG, PNG, or WebP image, or select the built-in synthetic demo.
2. Optionally add custom object names and adjust the detector threshold.
3. Click **Run full scene scan**.
4. Review annotated boxes and crops in **Visual Evidence Locker**.
5. Test an explanation in **Hypothesis Q&A**.
6. Compare a statement with the visual evidence in **Statement Cross-Examination**.
7. Download the complete PDF from **Investigative Brief**.

Example statement:

```text
There is no knife visible in this photograph.
```

---

## ✨ Highlights

* **Multi-Model Pipeline:** Combines open-weight detection with Gemma 4 vision reasoning.
* **Spatial Grounding:** Converts Gemma's normalized coordinates into pixel-accurate crops.
* **Evidence Locker:** Assigns a stable ID to every proposed observation.
* **Human Review:** Includes confirmation checkboxes and reviewer notes.
* **Hypothesis Testing:** Produces tentative possibilities, alternatives, and next checks.
* **Statement Comparison:** Flags only claims that directly conflict with visible evidence.
* **Source Integrity:** Stores a SHA-256 hash of the uploaded image in the report.
* **Structured Export:** Produces an annotated, source-linked PDF investigation brief.
* **Graceful Failure:** Preserves detector results if the hosted Gemma API is temporarily unavailable.

---

## ✅ Tests

Run the schema and evidence-validation tests:

```bash
python -m unittest discover -s scene_sift -p "test_*.py" -v
```

Run the complete local detector, crop, annotation, and PDF smoke test:

```bash
python scene_sift/smoke_test.py
```

---

## ⚠️ Limitations

* A detector score is not the probability that an object caused harm.
* A missed detection does not prove an object is absent or that a scene is safe.
* A single photograph cannot establish identity, intent, ownership, a timeline, events outside the frame, guilt, or a medical cause of death.
* Gemma 4 is accessed through Google's hosted API and may occasionally return temporary server errors.
* Every bounding box, crop, explanation, and report must be reviewed by a human.

---

## 🙋‍♂️ Author

**Syed Muhammad Faizan Ali**  
📍 Islamabad, Pakistan  
📧 faizandev666@gmail.com  
🔗 [GitHub](https://github.com/Slayer9966) | [LinkedIn](https://www.linkedin.com/in/faizan-ali-7b4275297/)

---

## 📄 License

SceneSift is released under the [MIT License](LICENSE).

Grounding DINO and Gemma retain their respective upstream licenses and terms.

