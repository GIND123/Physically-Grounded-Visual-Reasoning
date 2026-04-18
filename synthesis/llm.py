"""
GPT-4o physics reasoning agent — generates structured defect hypotheses.

For each (category, defect_type):
  1. Retrieve SOP/FMEA evidence from RAG knowledge base
  2. Send normal image + evidence to GPT-4o vision
  3. Receive structured JSON hypothesis with mechanism, bbox, severity, etc.
"""

import io
import json
import base64
import time
from datetime import datetime
from PIL import Image

from config.settings import OPENAI_API_KEY


# ── Prompt templates ──────────────────────────────────────────────────────────

HYPOTHESIS_SCHEMA = """{
  "observations": "Description of what you see in the normal image",
  "failure_mechanism": "Root cause mechanism that would produce this defect type",
  "defect_type": "exact defect type name",
  "expected_visual": "Precise description of what the defect looks like visually",
  "defect_bbox_normalized": [x1, y1, x2, y2],
  "severity": "minor|moderate|severe",
  "severity_score": 0.0-1.0,
  "confidence": 0.0-1.0,
  "supporting_evidence": ["evidence passage 1", "evidence passage 2"],
  "corrective_action": "specific process correction to prevent this defect",
  "counterfactual_prediction": "what the surface looks like after corrective action",
  "generation_prompt": "SD inpainting prompt (max 30 words)",
  "negative_prompt": "what should NOT appear in the generated image",
  "mask_guidance": "where on the object the defect should appear"
}"""

SYSTEM_PROMPT = (
    "You are a senior industrial quality engineer with 20 years of experience in "
    "failure analysis, materials science, and manufacturing process control.\n\n"
    "Given:\n"
    "1. A normal (defect-free) product image\n"
    "2. The product category and a specific defect type to analyze\n"
    "3. Retrieved evidence from SOPs and FMEAs\n\n"
    "Your task: Predict a structured defect hypothesis — what this defect would look "
    "like, where it would appear, and why.\n\n"
    "CRITICAL RULES:\n"
    "- The defect_bbox_normalized MUST be SMALL. Typical defects cover 5-15% of image area.\n"
    "  NEVER use bbox larger than [0.2, 0.2, 0.8, 0.8].\n"
    "- The generation_prompt must be specific and photorealistic (for Stable Diffusion).\n"
    "- The corrective_action must be a specific process change, not generic advice.\n"
    "- The counterfactual_prediction describes the product AFTER the corrective action.\n\n"
    "Respond with ONLY valid JSON matching this schema:\n"
    + HYPOTHESIS_SCHEMA
)


# ── Utility ───────────────────────────────────────────────────────────────────

def image_to_base64(img_path: str, max_size: int = 512) -> str:
    """Encode an image file to base64 JPEG string for GPT-4o vision."""
    img = Image.open(img_path).convert("RGB")
    img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Main function ─────────────────────────────────────────────────────────────

def generate_hypothesis(
    category: str,
    defect_type: str,
    normal_image_path: str,
    evidence_passages: list[dict],
    api_key: str = None,
) -> dict | None:
    """
    Call GPT-4o to generate a structured defect hypothesis.

    Args:
        category:           MVTec category name
        defect_type:        Defect type name
        normal_image_path:  Path to a normal (defect-free) reference image
        evidence_passages:  Retrieved KB passages (from synthesis.rag.retrieve_evidence)
        api_key:            OpenAI API key (defaults to config.settings.OPENAI_API_KEY)

    Returns:
        Hypothesis dict, or None on failure.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key or OPENAI_API_KEY)

    img_b64 = image_to_base64(normal_image_path)

    evidence_text = ""
    for p in evidence_passages:
        evidence_text += f"\n[{p['doc_type'].upper()} - {p['source']}]\n{p['text']}\n"

    user_content = [
        {
            "type": "text",
            "text": (
                f"Analyze this normal {category.replace('_', ' ')} image for "
                f"potential '{defect_type.replace('_', ' ')}' defect.\n\n"
                f"CATEGORY: {category}\n"
                f"DEFECT TYPE: {defect_type}\n\n"
                f"RETRIEVED EVIDENCE:\n{evidence_text}\n\n"
                f"Based on the image and evidence, generate a structured defect hypothesis.\n"
                f"Remember: bbox must be SMALL (5-15% of image). Be specific about the physics."
            ),
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_b64}",
                "detail": "high",
            },
        },
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=1500,
        )

        hypothesis = json.loads(response.choices[0].message.content)

        # ── Validate & clamp bbox ─────────────────────────────────────────────
        bbox = hypothesis.get("defect_bbox_normalized", [0.35, 0.35, 0.55, 0.55])
        if not isinstance(bbox, list) or len(bbox) != 4:
            bbox = [0.35, 0.35, 0.55, 0.55]
        bbox = [max(0.0, min(1.0, float(v))) for v in bbox]

        # Enforce max width/height of 35%
        for axis in [(0, 2), (1, 3)]:
            lo, hi = axis
            span = bbox[hi] - bbox[lo]
            if span > 0.35:
                cx = (bbox[lo] + bbox[hi]) / 2
                bbox[lo], bbox[hi] = cx - 0.175, cx + 0.175
            if span < 0.05:
                cx = (bbox[lo] + bbox[hi]) / 2
                bbox[lo], bbox[hi] = cx - 0.04, cx + 0.04

        bbox = [round(max(0.0, min(1.0, v)), 3) for v in bbox]
        hypothesis["defect_bbox_normalized"] = bbox

        hypothesis["_meta"] = {
            "generation_method": "gpt4o_vision_rag",
            "model":             "gpt-4o",
            "category":          category,
            "defect_type":       defect_type,
            "n_evidence_passages": len(evidence_passages),
            "normal_image":      normal_image_path,
            "timestamp":         datetime.now().isoformat(),
            "usage": {
                "prompt_tokens":     response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
        }

        return hypothesis

    except Exception as e:
        print(f"    GPT-4o error: {e}")
        return None


def generate_operator_report(
    category: str,
    defect_type: str,
    hypothesis: dict,
    verification_data: dict,
    counterfactual_data: dict,
    api_key: str = None,
) -> str | None:
    """
    Generate a human-readable operator report using GPT-4o-mini.

    Returns the report text, or None on failure.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key or OPENAI_API_KEY)

    system_prompt = (
        "You are an industrial quality control report generator.\n"
        "Given a structured defect hypothesis and verification results, produce a concise "
        "operator-facing report in this format:\n\n"
        "DEFECT ANALYSIS REPORT\n"
        "═════════════════════\n"
        "Product: [category]\n"
        "Defect Type: [type]\n"
        "Severity: [level]\n\n"
        "FINDINGS:\n[2-3 sentences]\n\n"
        "ROOT CAUSE:\n[1-2 sentences]\n\n"
        "CORRECTIVE ACTION:\n[specific process adjustment]\n\n"
        "VERIFICATION STATUS:\n[passed/failed stages and confidence]\n\n"
        "COUNTERFACTUAL:\n[effect of corrective action]\n\n"
        "Keep it under 200 words. Plain language for a factory floor operator."
    )

    user_msg = (
        f"Generate an operator report for:\n\n"
        f"Hypothesis: {json.dumps({k: v for k, v in hypothesis.items() if k != '_meta'}, indent=2)}\n\n"
        f"Verification: {verification_data.get('n_accepted', 'N/A')} of "
        f"{verification_data.get('n_total', 'N/A')} synthetic images accepted\n\n"
        f"Counterfactual: {json.dumps(counterfactual_data)}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"    Report generation error: {e}")
        return None
