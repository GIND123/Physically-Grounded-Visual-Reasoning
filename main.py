"""
PGVR-IDA Main Entry Point

Usage:
    # Full pipeline
    python main.py --stage all

    # Individual stages
    python main.py --stage 1
    python main.py --stage 2 --openai-key sk-...
    python main.py --stage 3

    # Specific categories only
    python main.py --stage 1 --categories bottle metal_nut transistor

Environment variables:
    PGVR_ROOT      — Project root directory (default: /root/project)
    OPENAI_API_KEY — OpenAI API key (required for Stage 2)
    HF_TOKEN       — HuggingFace token (for model downloads)
"""

import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="PGVR-IDA Industrial Defect Augmentation Pipeline"
    )
    parser.add_argument(
        "--stage",
        choices=["1", "2", "3", "all"],
        default="all",
        help="Which pipeline stage to run (default: all)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Subset of MVTec categories to process (default: all 15)",
    )
    parser.add_argument(
        "--openai-key",
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Project root directory (or set PGVR_ROOT env var)",
    )
    parser.add_argument(
        "--skip-lora",
        action="store_true",
        help="Stage 1: skip LoRA training",
    )
    parser.add_argument(
        "--skip-controlnet-val",
        action="store_true",
        help="Stage 1: skip ControlNet validation",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Stage 1: skip DINOv2 feature extraction",
    )
    parser.add_argument(
        "--no-lpips",
        action="store_true",
        help="Stage 3: skip LPIPS computation (faster)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Set root before importing settings
    if args.root:
        os.environ["PGVR_ROOT"] = args.root
    if args.openai_key:
        os.environ["OPENAI_API_KEY"] = args.openai_key

    from pipeline.stage1 import run_stage1
    from pipeline.stage2 import run_stage2
    from pipeline.stage3 import run_stage3

    run_s1 = args.stage in ("1", "all")
    run_s2 = args.stage in ("2", "all")
    run_s3 = args.stage in ("3", "all")

    if run_s1:
        run_stage1(
            categories=args.categories,
            train_lora=not args.skip_lora,
            validate_cnet=not args.skip_controlnet_val,
            extract_features=not args.skip_features,
        )

    if run_s2:
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            print("WARNING: OPENAI_API_KEY not set — hypothesis generation will fail.")
        run_stage2(
            categories=args.categories,
            api_key=key or None,
        )

    if run_s3:
        run_stage3(compute_lpips=not args.no_lpips)


if __name__ == "__main__":
    main()
