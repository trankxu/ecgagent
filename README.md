# Cardiac Multimodal Agent Demo

This repository contains a **demo AI agent pipeline** for cardiac diagnosis support using multimodal input data:

- ECG
- Chest X-ray
- Cardiac MRI ("hyper resonance")
- Optional clinical context

The demo uses an LLM API (OpenAI Chat Completions compatible) for:

1. **Planner model** – proposes diagnostic hypotheses.
2. **Evaluator model** – critiques and refines planner output.
3. **Summarizer model** – converts uploaded files (text/image) into modality summaries when manual summaries are not provided.

> ⚠️ This is a technical demo and **not** a medical device.

## Quick start

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your API key:

```bash
export OPENAI_API_KEY="your_api_key"
```

## Run options

### A) CLI demo

```bash
python demo_agent.py
```

### B) Visual GUI demo (Streamlit)

```bash
streamlit run app.py
```

Then open the shown local URL in your browser.

## GUI features

- Upload modality files for ECG / Chest X-ray / Cardiac MRI.
- Optionally input manual text summaries per modality.
- If summary text is blank and file is uploaded, the summarizer model generates a summary from the file.
- Run planner + evaluator pipeline and download JSON output.

## Supported upload types (demo)

- Text-like: `.txt`, `.csv`, `.json`
- Image-like: `.png`, `.jpg`, `.jpeg`

## Output format

The final JSON includes:

- `planner_output`
- `final_recommendation.primary_diagnosis`
- `final_recommendation.differential_diagnoses`
- `final_recommendation.confidence`
- `final_recommendation.recommended_next_steps`
- `final_recommendation.safety_notes`
- `final_recommendation.evaluator_comments`
