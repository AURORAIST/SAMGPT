from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore")

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from stage_pipeline import Stage1Config, Stage2Config, run_stage1, run_stage2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Unified Stage-1/Stage-2 pipeline runner")
    parser.add_argument(
        "--stage",
        type=str,
        default="both",
        choices=["1", "2", "both"],
        help="Which stage to run: 1, 2, or both",
    )

    # Stage-1 args
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=str,
        default=["Cora", "Citeseer", "Pubmed", "Photo", "Computers"],
        help="Datasets for Stage-1 or for Stage-2 auto path when --stage=both",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str((Path(__file__).resolve().parents[2] / "data").resolve()),
        help="Dataset root folder",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str((Path(__file__).resolve().parents[2] / "result" / "stage_outputs" / "stage1").resolve()),
        help="Stage-1 output root",
    )
    parser.add_argument(
        "--stage2-root",
        type=str,
        default=str((Path(__file__).resolve().parents[2] / "result" / "stage_outputs" / "stage2").resolve()),
        help="Stage-2 output root (used by --stage=both and as default for --stage=2)",
    )
    parser.add_argument("--max-subgraph-nodes", type=int, default=64, help="Max nodes per subgraph")
    parser.add_argument("--subgraph-hop-n", type=int, default=2, help="n-hop expansion depth")
    parser.add_argument("--num-clusters", type=int, default=8, help="Number of GMM clusters")
    parser.add_argument(
        "--encoding-mode",
        type=str,
        default="raw9",
        choices=["raw9", "decorrelated7"],
        help="Stage-1 clustering feature encoding mode",
    )
    parser.add_argument("--seed", type=int, default=39, help="Random seed")
    parser.add_argument("--use-log1p", type=int, default=1, help="1: enable log1p; 0: disable")
    parser.add_argument("--remove-selfloops", type=int, default=0, help="1: remove self-loops; 0: keep")

    # Stage-2 args
    parser.add_argument(
        "--input-csv",
        type=str,
        default="",
        help="Stage-2 input csv path. Required when --stage=2. Optional otherwise.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Stage-2 output dir. Default: <stage2-root>/<dataset_name> for --stage=2; inferred for --stage=both.",
    )
    parser.add_argument("--top-m-per-cluster", type=int, default=5, help="Stage-2 representative pool size per cluster")
    parser.add_argument("--alpha", type=float, default=1.0, help="Stage-2 novelty weight")
    parser.add_argument("--beta", type=float, default=1.0, help="Stage-2 coverage weight")
    parser.add_argument("--gamma", type=float, default=1.0, help="Stage-2 redundancy weight")
    parser.add_argument(
        "--center-path",
        type=str,
        default="",
        help="Optional cluster centers .npy path for Stage-2",
    )

    return parser


def run_stage1_for_datasets(args: argparse.Namespace) -> None:
    print("=" * 80)
    print("Stage-1: dataset definition and structural encoding")
    print("data_root=", args.data_root)
    print("output_root=", args.output_root)
    print("datasets=", args.datasets)
    print("subgraph_hop_n=", int(args.subgraph_hop_n))
    print("remove_selfloops=", int(args.remove_selfloops))
    print("cluster_method=", "gmm")
    print("encoding_mode=", args.encoding_mode)
    print("use_log1p=", int(args.use_log1p))
    print("=" * 80)

    for name in args.datasets:
        config = Stage1Config(
            dataset=name,
            data_root=args.data_root,
            output_root=args.output_root,
            max_subgraph_nodes=args.max_subgraph_nodes,
            subgraph_hop_n=args.subgraph_hop_n,
            num_clusters=args.num_clusters,
            encoding_mode=args.encoding_mode,
            seed=args.seed,
            remove_selfloops=bool(args.remove_selfloops),
            use_log1p=bool(args.use_log1p),
        )
        result = run_stage1(config)
        print(
            f"[DONE] stage1 dataset={result['dataset']} samples={result['num_samples']} "
            f"effective_k={result['effective_k']} output={result['output_dir']}"
        )
        print(f"summary={result['summary_json']}")


def run_stage2_once(args: argparse.Namespace, input_csv: Path, output_dir: Path) -> None:
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
    print("[DONE] stage2 num_samples=", result["num_samples"])
    print("[DONE] representative_pool_size=", result["representative_pool_size"])
    print("[DONE] summary=", result["summary_json"])
    print("[DONE] scores_csv=", result["scores_csv"])


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.stage in ("1", "both"):
        run_stage1_for_datasets(args)

    if args.stage == "2":
        if not args.input_csv:
            parser.error("--input-csv is required when --stage=2")
        input_csv = Path(args.input_csv).resolve()
        if args.output_dir:
            output_dir = Path(args.output_dir).resolve()
        else:
            ds_name = input_csv.parent.name
            output_dir = (Path(args.stage2_root).resolve() / ds_name).resolve()
        run_stage2_once(args, input_csv, output_dir)

    if args.stage == "both":
        output_root = Path(args.output_root).resolve()
        stage2_root = Path(args.stage2_root).resolve()
        for name in args.datasets:
            input_csv = (output_root / name.lower() / "structural_encodings.csv").resolve()
            if not input_csv.exists():
                raise FileNotFoundError(f"Stage-1 output missing for dataset={name}: {input_csv}")
            output_dir = (stage2_root / name.lower()).resolve()
            run_stage2_once(args, input_csv, output_dir)


if __name__ == "__main__":
    main()
