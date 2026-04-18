"""
Stage 3 pipeline — Evaluation & Paper Results.

Steps:
  1. Aggregate pipeline metrics (accept rate, CF suppression) per category
  2. Compute generation quality (LPIPS vs real defects)
  3. Generate paper figures (qualitative grid, stage pass rates, acceptance rates)
  4. Export LaTeX tables
"""

import os
import json

from config.settings import (
    ALL_CATEGORIES, S2_DIR, S3_DIR, FIGURES_DIR, TABLES_DIR,
)
from evaluation.metrics import (
    compute_pipeline_metrics,
    compute_generation_quality,
    make_qualitative_grid,
    plot_stage_pass_rates,
    plot_acceptance_rates,
)


# ── Default showcase pairs for Figure 1 ──────────────────────────────────────

DEFAULT_SHOWCASE = [
    ("bottle",      "broken_large"),
    ("hazelnut",    "crack"),
    ("metal_nut",   "scratch"),
    ("pill",        "crack"),
    ("transistor",  "bent_lead"),
    ("zipper",      "broken_teeth"),
    ("capsule",     "squeeze"),
    ("toothbrush",  "defective"),
]


# ── LaTeX table helpers ───────────────────────────────────────────────────────

def _latex_pipeline_table(pipeline_metrics: dict) -> str:
    """Return a LaTeX table string for pipeline metrics."""
    rows = []
    for cat in ALL_CATEGORIES:
        m = pipeline_metrics.get(cat, {})
        rows.append(
            f"  {cat.replace('_', '\\_'):<20} & "
            f"{m.get('n_defect_types',  0):>5} & "
            f"{m.get('n_generated',     0):>5} & "
            f"{m.get('accept_rate',   0.0):>5.0%} & "
            f"{m.get('cf_rate',       0.0):>5.0%} \\\\"
        )
    body = "\n".join(rows)
    return (
        "\\begin{tabular}{lrrrrr}\n"
        "\\toprule\n"
        "Category & Types & Gen & Accept Rate & CF Rate \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}"
    )


def _latex_quality_table(gen_quality: dict) -> str:
    """Return a LaTeX table string for LPIPS generation quality."""
    rows = []
    for cat in ALL_CATEGORIES:
        q = gen_quality.get(cat, {})
        rows.append(
            f"  {cat.replace('_', '\\_'):<20} & "
            f"{q.get('lpips', float('nan')):.4f} & "
            f"{q.get('n_syn', 0):>5} \\\\"
        )
    body = "\n".join(rows)
    return (
        "\\begin{tabular}{lrr}\n"
        "\\toprule\n"
        "Category & LPIPS↓ & N Synthetic \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}"
    )


# ── Main entry point ──────────────────────────────────────────────────────────

def run_stage3(
    showcase: list[tuple] = None,
    compute_lpips: bool = True,
) -> None:
    """Run the complete Stage 3 evaluation pipeline."""
    for d in [S3_DIR, FIGURES_DIR, TABLES_DIR]:
        os.makedirs(d, exist_ok=True)

    # Load Stage 2 results
    ver_path = os.path.join(S2_DIR, "verification_results.json")
    cf_path  = os.path.join(S2_DIR, "counterfactual_results.json")

    if not os.path.exists(ver_path):
        raise FileNotFoundError(f"Stage 2 results missing: {ver_path}")

    with open(ver_path) as f:
        verification_results = json.load(f)
    with open(cf_path) as f:
        counterfactual_results = json.load(f)

    print("=" * 70)
    print("STAGE 3: Evaluation & Paper Results")
    print("=" * 70)

    # 1. Pipeline metrics
    print("\n[1] Computing pipeline metrics ...")
    pipeline_metrics = compute_pipeline_metrics(verification_results, counterfactual_results)

    with open(os.path.join(S3_DIR, "pipeline_metrics.json"), "w") as f:
        json.dump(pipeline_metrics, f, indent=2)

    with open(os.path.join(TABLES_DIR, "pipeline_table.tex"), "w") as f:
        f.write(_latex_pipeline_table(pipeline_metrics))
    print("  Saved pipeline_metrics.json + pipeline_table.tex")

    # 2. Generation quality
    gen_quality = {}
    if compute_lpips:
        print("\n[2] Computing LPIPS generation quality ...")
        gen_quality = compute_generation_quality()
        with open(os.path.join(S3_DIR, "generation_quality.json"), "w") as f:
            json.dump(gen_quality, f, indent=2)
        with open(os.path.join(TABLES_DIR, "quality_table.tex"), "w") as f:
            f.write(_latex_quality_table(gen_quality))
        print("  Saved generation_quality.json + quality_table.tex")

    # 3. Paper figures
    print("\n[3] Generating paper figures ...")

    make_qualitative_grid(
        showcase or DEFAULT_SHOWCASE,
        output_path=os.path.join(FIGURES_DIR, "fig_pipeline_grid.png"),
    )

    plot_stage_pass_rates(
        pipeline_metrics,
        output_path=os.path.join(FIGURES_DIR, "fig_stage_pass_rates.png"),
    )

    plot_acceptance_rates(
        pipeline_metrics,
        output_path=os.path.join(FIGURES_DIR, "fig_acceptance_rates.png"),
    )

    print("\n" + "=" * 70)
    print("STAGE 3 COMPLETE")
    print(f"  Metrics:   {S3_DIR}/pipeline_metrics.json")
    print(f"  Figures:   {FIGURES_DIR}/")
    print(f"  Tables:    {TABLES_DIR}/")
    print("=" * 70)
