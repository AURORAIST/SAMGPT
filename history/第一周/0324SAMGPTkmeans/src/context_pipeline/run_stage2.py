from __future__ import annotations

import argparse
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from stage2_learnability import Stage2Config, run_stage2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage-2 learnability scoring and representative pool")
    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Path to stage1 structural_encodings.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output folder; default is <csv_dir>/stage2",
    )
    parser.add_argument("--top-m-per-cluster", type=int, default=5, help="Representative samples per cluster")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for novelty")
    parser.add_argument("--beta", type=float, default=1.0, help="Weight for coverage")
    parser.add_argument("--gamma", type=float, default=1.0, help="Weight for redundancy")
    parser.add_argument(
        "--center-path",
        type=str,
        default="",
        help="Optional kmeans centers path (.npy). Default: sibling kmeans_centers.npy",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_csv = Path(args.input_csv).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (input_csv.parent / "stage2").resolve()

    print("=" * 80)
    print("Stage-2: learnability scoring and representative pool")
    print("input_csv=", str(input_csv))
    print("output_dir=", str(output_dir))
    print("top_m_per_cluster=", args.top_m_per_cluster)
    print("alpha,beta,gamma=", args.alpha, args.beta, args.gamma)
    print("=" * 80)

    config = Stage2Config(
        input_csv=str(input_csv),
        output_dir=str(output_dir),
        top_m_per_cluster=args.top_m_per_cluster,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        center_path=args.center_path,
    )

    result = run_stage2(config)
    print("[DONE] num_samples=", result["num_samples"])
    print("[DONE] representative_pool_size=", result["representative_pool_size"])
    print("[DONE] summary=", result["summary_json"])
    print("[DONE] scores_csv=", result["scores_csv"])


if __name__ == "__main__":
    main()
