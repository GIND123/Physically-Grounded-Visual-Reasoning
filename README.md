# Physically-Grounded Visual Reasoning for Industrial Anomaly Detection

> Bridging manufacturing domain knowledge and vision-language models to synthesize physically plausible defect data for industrial quality inspection.

Anomaly detection in industrial settings is bottlenecked by the scarcity of labeled defect samples. This project introduces a physically-grounded reasoning pipeline that leverages large language models (LLMs) — conditioned on materials science and manufacturing process knowledge — to generate structured defect hypotheses. These hypotheses drive diffusion-based synthetic image generation, augmenting training data in a semantically meaningful way. A counterfactual verification mechanism assesses hypothesis quality by predicting visual outcomes under corrective process interventions. The full system is evaluated on the MVTec Anomaly Detection benchmark across 15 product categories.

---

## Key Contributions

- **Physics-grounded hypothesis generation** — LLMs reason over material properties, process parameters, and failure mechanisms rather than generating defects arbitrarily.
- **Retrieval-Augmented Generation (RAG)** — Domain evidence from standard operating procedures and failure databases is retrieved at inference time to ground each hypothesis.
- **Counterfactual reasoning** — Each defect hypothesis includes a counterfactual prediction: what the product would look like if the root-cause corrective action were applied, enabling causal validation.
- **Operator-facing reports** — The pipeline generates structured inspection reports suitable for production floor use.
- **Full MVTec AD coverage** — 15 product categories spanning metals, polymers, textiles, electronics, and organic materials.

---

## Pipeline

```
┌─────────────────────────────────┐
│  Defect Taxonomy + Process      │
│  Context (material, mechanism,  │
│  severity, process parameters)  │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  RAG Retrieval                  │
│  (SOPs · Failure Databases)     │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  LLM Hypothesis Generation      │
│  · Failure mechanism            │
│  · Defect location + bbox       │
│  · Severity classification      │
│  · Corrective action            │
│  · Counterfactual prediction    │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Diffusion-based Synthetic      │
│  Image Generation               │
│  (hypothesis-conditioned)       │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Anomaly Detection Training     │
│  Real images + Synthetic        │
│  augmentation (EfficientAD)     │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Counterfactual Verification    │
│  + Operator Inspection Report   │
└─────────────────────────────────┘
```

---

## Supported Product Categories

| # | Category | Material Class | Defect Mechanisms |
|---|---|---|---|
| 1 | Bottle | Glass | Thermal shock, mechanical impact, contamination |
| 2 | Cable | Copper / PVC | Tool damage, assembly error, puncture |
| 3 | Capsule | Gelatin | Mechanical stress, imprint failure, puncture |
| 4 | Carpet | Woven textile | Contamination, cutting, threading error |
| 5 | Grid | Metal mesh | Bending, adhesive contamination, breakage |
| 6 | Hazelnut | Organic | Impact cracking, surface cutting |
| 7 | Leather | Processed hide | Staining, folding, adhesive residue |
| 8 | Metal Nut | Steel | Stamping error, oxidation, orientation flip |
| 9 | Pill | Pharmaceutical | Surface cracking, imprint failure, contamination |
| 10 | Screw | Steel | Thread damage, surface scratching |
| 11 | Tile | Ceramic | Fracture, oil contamination, surface roughness |
| 12 | Toothbrush | Nylon / Plastic | Bristle deformation |
| 13 | Transistor | Epoxy / Copper | Lead bending, case damage, misplacement |
| 14 | Wood | Hardwood | Liquid damage, scratch, color variation |
| 15 | Zipper | Nylon | Tooth damage, fabric misalignment |

---

## Hypothesis Format

Each hypothesis is a structured JSON object encoding physical reasoning about the defect:

```json
{
  "observed_anomaly_cues": ["..."],
  "failure_mechanism": "One-sentence physical root cause",
  "mechanism_explanation": "Why this mechanism produces this defect on this material",
  "defect_region": "Natural language location on the object",
  "defect_bbox_normalized": [x_min, y_min, x_max, y_max],
  "severity": "minor | moderate | critical",
  "severity_visual_description": "How severity visually manifests",
  "defect_class_token": "snake_case_token_for_diffusion_conditioning",
  "confidence": 0.0,
  "corrective_action": "Specific process intervention",
  "counterfactual_prediction": "Expected visual outcome after correction"
}
```

---

## Experimental Results

### Image-Level AUROC — Real vs. Real + Synthetic (EfficientAD)

| Category | Baseline (Real Only) | Augmented (+ Synthetic) |
|---|---|---|
| Bottle | 0.9516 | 0.9524 |
| Metal Nut | 0.9848 | 0.9800 |
| Transistor | 0.9396 | 0.9417 |

### Ablation Study — Image AUROC

| Category | No Augmentation | Full Pipeline | w/o Critic | Random Placement |
|---|---|---|---|---|
| Bottle | 0.9024 | **0.9175** | 0.9175 | 0.9302 |
| Metal Nut | 1.0000 | **1.0000** | 1.0000 | 1.0000 |
| Transistor | 0.9429 | **0.9417** | 0.9417 | 0.9417 |

### Counterfactual Defect Suppression

| Category | Defect Type | Corrective Action Applied | Suppressed |
|---|---|---|---|
| Metal Nut | color | Reduce process temperature | Yes |
| Bottle | contamination | Improve cleanroom filtration | Yes |
| Metal Nut | scratch | Replace worn tooling | Yes |
| Bottle | broken_large | Reduce cooling rate | No |
| Transistor | cut_lead | Recalibrate cutting tools | No |

---

## Repository Structure

```
├── stage_1.ipynb          # End-to-end pipeline notebook
├── configs/
│   ├── defect_taxonomy.json      # Material + mechanism definitions (15 categories)
│   ├── hypothesis/               # Generated hypotheses per defect type
│   ├── process_context/          # Per-category manufacturing parameters
│   └── prompt_templates/         # LLM prompt templates (hypothesis, counterfactual, RAG, report)
├── datasets/                     # Dataset metadata and statistics
└── results/                      # AUROC evaluations, ablations, verification scores
```

---

## Getting Started

### Requirements

- Python 3.9+
- CUDA-capable GPU
- [MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

### Setup

```bash 
git clone https://github.com/GIND123/Physically-Grounded-Visual-Reasoning.git
cd Physically-Grounded-Visual-Reasoning
pip install -r requirements.txt
```

### Usage

Open `stage_1.ipynb` and configure the path to your local MVTec AD directory. The notebook runs the full pipeline end-to-end:

1. Load defect taxonomy and manufacturing process context
2. Build RAG index from domain knowledge sources
3. Generate per-defect LLM hypotheses
4. Synthesize defect images via diffusion conditioning
5. Train and evaluate anomaly detection model
6. Run counterfactual verification
7. Generate operator inspection reports

---
