import base64
import json
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from openai import OpenAI


@dataclass
class CardiacCase:
    ecg_summary: str
    chest_xray_summary: str
    cardiac_mri_summary: str
    clinical_context: str = ""

    def as_prompt_block(self) -> str:
        return (
            f"ECG Findings:\n{self.ecg_summary}\n\n"
            f"Chest X-ray Findings:\n{self.chest_xray_summary}\n\n"
            f"Cardiac MRI Findings:\n{self.cardiac_mri_summary}\n\n"
            f"Clinical Context:\n{self.clinical_context or 'Not provided'}"
        )


class CardiacDiagnosisAgent:
    def __init__(
        self,
        planner_model: str = "gpt-4o-mini",
        evaluator_model: str = "gpt-4o-mini",
        modality_summarizer_model: str = "gpt-4o-mini",
        temperature: float = 0.2,
    ) -> None:
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.planner_model = planner_model
        self.evaluator_model = evaluator_model
        self.modality_summarizer_model = modality_summarizer_model
        self.temperature = temperature

    def _chat_json(self, model: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Model returned an empty response.")
        return json.loads(content)

    def _chat_text(self, model: str, messages: list[Dict[str, Any]]) -> str:
        response = self.client.chat.completions.create(
            model=model,
            temperature=self.temperature,
            messages=messages,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Model returned an empty text response.")
        return content

    def summarize_modality_file(self, modality_name: str, file_path: str) -> str:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        suffix = path.suffix.lower()

        system_prompt = (
            "You are a cardiac imaging/data summarizer for a prototype tool. "
            "Summarize only visible/provided findings in concise clinical language. "
            "If uncertain, state uncertainty explicitly."
        )

        text_suffixes = {".txt", ".csv", ".md", ".json"}
        if suffix in text_suffixes:
            raw = path.read_text(encoding="utf-8", errors="ignore")
            truncated = raw[:12000]
            return self._chat_text(
                self.modality_summarizer_model,
                [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Modality: {modality_name}\n"
                            "Please summarize the following modality data into 4-7 bullet points:\n\n"
                            f"{truncated}"
                        ),
                    },
                ],
            )

        if mime_type.startswith("image/"):
            file_bytes = path.read_bytes()
            data_b64 = base64.b64encode(file_bytes).decode("utf-8")
            data_url = f"data:{mime_type};base64,{data_b64}"

            return self._chat_text(
                self.modality_summarizer_model,
                [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"Modality: {modality_name}. "
                                    "Describe clinically relevant findings visible in this file."
                                ),
                            },
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
            )

        return (
            f"Unsupported file type ({suffix or mime_type}) for automatic {modality_name} summarization. "
            "Please provide a manual text summary in the GUI."
        )

    def build_case_from_sources(
        self,
        ecg_summary_text: str,
        chest_xray_summary_text: str,
        cardiac_mri_summary_text: str,
        clinical_context: str = "",
        ecg_file_path: Optional[str] = None,
        chest_xray_file_path: Optional[str] = None,
        cardiac_mri_file_path: Optional[str] = None,
    ) -> CardiacCase:
        ecg_summary = ecg_summary_text.strip()
        chest_xray_summary = chest_xray_summary_text.strip()
        cardiac_mri_summary = cardiac_mri_summary_text.strip()

        if not ecg_summary and ecg_file_path:
            ecg_summary = self.summarize_modality_file("ECG", ecg_file_path)
        if not chest_xray_summary and chest_xray_file_path:
            chest_xray_summary = self.summarize_modality_file("Chest X-ray", chest_xray_file_path)
        if not cardiac_mri_summary and cardiac_mri_file_path:
            cardiac_mri_summary = self.summarize_modality_file("Cardiac MRI", cardiac_mri_file_path)

        if not ecg_summary:
            ecg_summary = "Not provided"
        if not chest_xray_summary:
            chest_xray_summary = "Not provided"
        if not cardiac_mri_summary:
            cardiac_mri_summary = "Not provided"

        return CardiacCase(
            ecg_summary=ecg_summary,
            chest_xray_summary=chest_xray_summary,
            cardiac_mri_summary=cardiac_mri_summary,
            clinical_context=clinical_context,
        )

    def plan(self, case: CardiacCase) -> Dict[str, Any]:
        system_prompt = (
            "You are a cardiac diagnostic planning assistant. "
            "Given multimodal findings, propose likely diagnosis hypotheses in JSON. "
            "Do not claim certainty. Always include potential risks and missing data."
        )
        user_prompt = (
            "Analyze this case and return JSON with keys:\n"
            "- primary_diagnosis (string)\n"
            "- differential_diagnoses (array of strings)\n"
            "- confidence (0 to 1 float)\n"
            "- rationale (string)\n"
            "- missing_information (array of strings)\n"
            "- recommended_next_steps (array of strings)\n\n"
            f"Case:\n{case.as_prompt_block()}"
        )
        return self._chat_json(self.planner_model, system_prompt, user_prompt)

    def evaluate(self, case: CardiacCase, planner_output: Dict[str, Any]) -> Dict[str, Any]:
        system_prompt = (
            "You are a senior cardiac safety evaluator. Review the planner's draft and improve it. "
            "Return strict JSON only. Ensure cautious phrasing and add safety notes."
        )
        user_prompt = (
            "Given the case and planner output, return JSON with keys:\n"
            "- primary_diagnosis (string)\n"
            "- differential_diagnoses (array of strings)\n"
            "- confidence (0 to 1 float)\n"
            "- recommended_next_steps (array of strings)\n"
            "- safety_notes (array of strings)\n"
            "- evaluator_comments (string)\n\n"
            f"Case:\n{case.as_prompt_block()}\n\n"
            f"Planner output:\n{json.dumps(planner_output, indent=2)}"
        )
        return self._chat_json(self.evaluator_model, system_prompt, user_prompt)

    def run(self, case: CardiacCase) -> Dict[str, Any]:
        plan = self.plan(case)
        final = self.evaluate(case, plan)
        return {
            "planner_output": plan,
            "final_recommendation": final,
        }


def main() -> None:
    sample_case = CardiacCase(
        ecg_summary=(
            "Sinus tachycardia, diffuse ST-segment elevation in lateral leads, "
            "frequent premature ventricular complexes."
        ),
        chest_xray_summary=(
            "Mild cardiomegaly, bilateral interstitial opacities, no focal consolidation, "
            "small pleural effusion."
        ),
        cardiac_mri_summary=(
            "Patchy mid-wall late gadolinium enhancement in the inferolateral wall, "
            "myocardial edema on T2-weighted imaging, mildly reduced LVEF."
        ),
        clinical_context=(
            "36-year-old with chest pain and dyspnea 7 days after viral prodrome. "
            "Troponin elevated."
        ),
    )

    agent = CardiacDiagnosisAgent()
    result = agent.run(sample_case)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
