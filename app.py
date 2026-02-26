import json
import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st

from demo_agent import CardiacDiagnosisAgent


st.set_page_config(page_title="Cardiac Multimodal Diagnosis Demo", layout="wide")
st.title("Cardiac Multimodal Diagnosis Demo")
st.caption("Prototype only — not for clinical decision-making.")


@st.cache_resource
def get_agent() -> CardiacDiagnosisAgent:
    return CardiacDiagnosisAgent()


def save_upload(uploaded_file) -> Optional[str]:
    if uploaded_file is None:
        return None

    suffix = Path(uploaded_file.name).suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


with st.sidebar:
    st.header("Model settings")
    planner_model = st.text_input("Planner model", value="gpt-4o-mini")
    evaluator_model = st.text_input("Evaluator model", value="gpt-4o-mini")
    summarizer_model = st.text_input("Summarizer model", value="gpt-4o-mini")

st.subheader("1) Upload multimodal inputs")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### ECG")
    ecg_file = st.file_uploader("Upload ECG file", type=["txt", "csv", "json", "png", "jpg", "jpeg"], key="ecg")
    ecg_text = st.text_area("ECG summary (optional)", height=160)

with col2:
    st.markdown("#### Chest X-ray")
    cxr_file = st.file_uploader("Upload Chest X-ray", type=["png", "jpg", "jpeg", "txt"], key="cxr")
    cxr_text = st.text_area("Chest X-ray summary (optional)", height=160)

with col3:
    st.markdown("#### Cardiac MRI")
    mri_file = st.file_uploader("Upload Cardiac MRI", type=["png", "jpg", "jpeg", "txt"], key="mri")
    mri_text = st.text_area("Cardiac MRI summary (optional)", height=160)

clinical_context = st.text_area("2) Clinical context (optional)", height=120)

if st.button("Run diagnosis demo", type="primary"):
    try:
        agent = get_agent()
        agent.planner_model = planner_model
        agent.evaluator_model = evaluator_model
        agent.modality_summarizer_model = summarizer_model

        ecg_path = save_upload(ecg_file)
        cxr_path = save_upload(cxr_file)
        mri_path = save_upload(mri_file)

        case = agent.build_case_from_sources(
            ecg_summary_text=ecg_text,
            chest_xray_summary_text=cxr_text,
            cardiac_mri_summary_text=mri_text,
            clinical_context=clinical_context,
            ecg_file_path=ecg_path,
            chest_xray_file_path=cxr_path,
            cardiac_mri_file_path=mri_path,
        )

        result = agent.run(case)

        st.success("Diagnosis pipeline completed.")

        with st.expander("Resolved input summaries", expanded=False):
            st.write(
                {
                    "ecg_summary": case.ecg_summary,
                    "chest_xray_summary": case.chest_xray_summary,
                    "cardiac_mri_summary": case.cardiac_mri_summary,
                    "clinical_context": case.clinical_context,
                }
            )

        st.subheader("Final output")
        st.json(result)

        st.download_button(
            label="Download JSON result",
            data=json.dumps(result, indent=2),
            file_name="cardiac_diagnosis_result.json",
            mime="application/json",
        )
    except Exception as exc:
        st.error(f"Run failed: {exc}")
