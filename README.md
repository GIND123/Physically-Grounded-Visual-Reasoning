# Physically-Grounded Visual Reasoning for Industrial Anomaly Detection

> Bridging manufacturing domain knowledge and vision-language models to synthesize physically plausible defect data for industrial quality inspection.

Anomaly detection in industrial settings is bottlenecked by the scarcity of labeled defect samples. This project introduces a physically-grounded reasoning pipeline that uses GPT-4o vision conditioned on materials science and manufacturing process knowledge to generate structured defect hypotheses. These hypotheses drive diffusion-based synthetic image generation via LoRA-fine-tuned Stable Diffusion and ControlNet, augmenting training data with semantically meaningful defects. A 5-stage verification critic and counterfactual reasoning step validate each generated image before it enters the training set. The full system is evaluated on the MVTec Anomaly Detection benchmark across 15 product categories.

---

## Key Contributions

- **Physics-grounded hypothesis generation**: GPT-4o reasons over material properties, process parameters, and failure mechanisms retrieved from SOPs and FMEAs via RAG (ChromaDB), rather than generating defects arbitrarily.
- **Structured synthesis pipeline**: Per-category LoRA adapters fine-tuned on SD2 Inpainting, guided by ControlNet Canny edges and SAM-generated defect masks, produce spatially precise synthetic defects.
- **5-stage verification critic**: Each generated image must pass at least 4 of 5 independent checks (CLIP consistency, WinCLIP anomaly score, SSIM structure preservation, DINOv2 patch-level localization, pixel intensity) before being accepted into training.
- **Counterfactual reasoning**: Each hypothesis predicts the visual outcome after the corrective action is applied, enabling causal validation of the generated defect.
- **Operator-facing reports**: The pipeline generates structured 4-paragraph inspection reports suitable for production floor use.

---

## Architecture

```
+----------------------------------------------------------+
|                    STAGE 1 - Foundation                  |
|                                                          |
|  MVTec AD  -->  DINOv2 triplet matching                  |
|                     |                                    |
|                     v                                    |
|  SD2 Inpainting LoRA fine-tuning (per category)          |
|  ControlNet (Canny) edge-conditioned validation          |
|  DINOv2 feature extraction -> FAISS index                |
+-------------------------+--------------------------------+
                          |
+-------------------------v--------------------------------+
|                 STAGE 2 - Agentic Reasoning              |
|                                                          |
|  RAG KB (SOPs + FMEAs via ChromaDB)                      |
|     |                                                    |
|     v                                                    |
|  GPT-4o Vision -> Structured Defect Hypothesis           |
|  (mechanism, bbox, severity, corrective action)          |
|     |                                                    |
|     v                                                    |
|  SAM mask generation + LoRA/ControlNet synthesis         |
|     |                                                    |
|     v                                                    |
|  5-Stage Verification Critic                             |
|  (CLIP, WinCLIP, SSIM, DINOv2, pixel diff)               |
|     |                                                    |
|     v                                                    |
|  Counterfactual scoring + Operator report                |
+-------------------------+--------------------------------+
                          |
+-------------------------v--------------------------------+
|                  STAGE 3 - Evaluation                   |
|                                                          |
|  Pipeline metrics, LPIPS quality, AUROC (EfficientAD)   |
|  qualitative grid, pass-rate plots    |                                 |
+----------------------------------------------------------+
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

## Experimental Results

### Image-Level AUROC: Real vs. Real + Synthetic (EfficientAD)

| Category | Baseline (Real Only) | Augmented (+ Synthetic) |
|---|---|---|
| Bottle | 0.9516 | 0.9524 |
| Metal Nut | 0.9848 | 0.9800 |
| Transistor | 0.9396 | 0.9417 |

### Ablation Study: Image AUROC

| Category | No Augmentation | Full Pipeline | w/o Critic | Random Placement |
|---|---|---|---|---|
| Bottle | 0.9024 | **0.9175** | 0.9175 | 0.9302 |
| Metal Nut | 1.0000 | **1.0000** | 1.0000 | 1.0000 |
| Transistor | 0.9429 | **0.9417** | 0.9417 | 0.9417 |

### Counterfactual Defect Suppression

| Category | Defect Type | Corrective Action | Suppressed |
|---|---|---|---|
| Metal Nut | color | Reduce process temperature | Yes |
| Bottle | contamination | Improve cleanroom filtration | Yes |
| Metal Nut | scratch | Replace worn tooling | Yes |
| Bottle | broken_large | Reduce cooling rate | No |
| Transistor | cut_lead | Recalibrate cutting tools | No |

---

## Repository Structure

```
├── main.py                        # CLI entry point
├── requirements.txt
│
├── config/
│   └── settings.py                # Central config: paths, hyperparameters, model IDs
│
├── configs/
│   ├── defect_taxonomy.json       # Material + mechanism definitions (15 categories)
│   ├── hypothesis/                # Pre-generated GPT-4o hypotheses per defect type
│   ├── process_context/           # Manufacturing process parameters per category
│   └── prompt_templates/          # LLM prompt templates
│
├── pipeline/
│   ├── stage1.py                  # LoRA training, ControlNet validation, DINOv2 indexing
│   ├── stage2.py                  # Hypothesis generation, synthesis, verification, reports
│   └── stage3.py                  # Evaluation metrics, figures, LaTeX tables
│
├── synthesis/
│   ├── generator.py               # LoRA + ControlNet defect image synthesis
│   ├── llm.py                     # GPT-4o vision hypothesis agent
│   ├── mask_gen.py                # SAM-based defect mask generation
│   └── rag.py                     # ChromaDB knowledge base (SOPs + FMEAs)
│
├── verification/
│   └── critic.py                  # 5-stage verification critic
│
├── models/
│   ├── lora.py                    # SD2 Inpainting LoRA fine-tuning
│   ├── controlnet.py              # ControlNet Canny edge conditioning
│   └── sam_mask.py                # SAM mask utilities
│
├── features/
│   └── extraction.py              # DINOv2 feature extraction + FAISS indexing
│
├── evaluation/
│   └── metrics.py                 # AUROC, LPIPS, pipeline metrics
│
├── data/
│   ├── dataset.py                 # MVTec AD dataset loader
│   └── paths.py                   # Path helpers
│
├── utils/
│   └── image.py                   # Image processing utilities
│
├── datasets/                      # Dataset metadata and statistics
└── results/                       # AUROC evaluations, ablations, verification scores
```

---

## How to Run

### Requirements

- Python 3.9+
- CUDA GPU (A100 recommended; 24 GB VRAM minimum for LoRA training)
- [MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- OpenAI API key (required for Stage 2 hypothesis generation)
- HuggingFace token (for model weight downloads)

### 1. Clone and Install

```bash
git clone https://github.com/GIND123/Physically-Grounded-Visual-Reasoning.git
cd Physically-Grounded-Visual-Reasoning
pip install -r requirements.txt
```

### 2. Download the SAM Checkpoint

```bash
mkdir -p checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
     -O checkpoints/sam_vit_h_4b8939.pth
```

### 3. Set Environment Variables

```bash
export PROJECT_ROOT=/path/to/project    # project root (default: /root/project)
export OPENAI_API_KEY=sk-...            # required for Stage 2
export HF_TOKEN=hf_...                  # required for model downloads
```

Place the MVTec AD dataset at `$PROJECT_ROOT/datasets/mvtec_ad/`, or update `MVTEC` in `config/settings.py`.

### 4. Run the Pipeline

**Full pipeline (all 3 stages, all 15 categories):**

```bash
python main.py --stage all
```

**Individual stages:**

```bash
python main.py --stage 1                            # Foundation: LoRA, ControlNet, DINOv2
python main.py --stage 2 --openai-key sk-...        # Reasoning: hypotheses, synthesis, verification
python main.py --stage 3                            # Evaluation: metrics, figures, LaTeX tables
```

**Subset of categories:**

```bash
python main.py --stage all --categories bottle metal_nut transistor
```

**Skip expensive steps:**

```bash
python main.py --stage 1 --skip-lora --skip-controlnet-val   # only re-index features
python main.py --stage 3 --no-lpips                           # skip LPIPS computation
```

### Runtime Estimates (A100 GPU)

| Stage | Operation | Approximate Time |
|---|---|---|
| 1 | LoRA fine-tuning (all 15 categories) | 4-6 hours |
| 1 | DINOv2 feature extraction + FAISS index | ~30 min |
| 2 | Hypothesis generation via GPT-4o | ~1 hour |
| 2 | Defect synthesis + 5-stage verification | 3-5 hours |
| 3 | Metrics, figures, and LaTeX export | ~20 min |

---

## Dependencies

Key packages (see `requirements.txt` for full list):

| Package | Purpose |
|---|---|
| `diffusers` + `peft` | SD2 Inpainting + LoRA fine-tuning |
| `controlnet-aux` | Canny edge conditioning |
| `segment-anything` | SAM mask generation |
| `openai` | GPT-4o hypothesis generation |
| `chromadb` + `sentence-transformers` | RAG knowledge base |
| `open-clip-torch` | CLIP verification stage |
| `faiss-cpu` | DINOv2 feature indexing |
| `lpips` | Generation quality metric |

---

