from __future__ import annotations

import argparse
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from stage1 import Stage1Config, run_stage1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Stage-1 dataset definition and structural encoding")
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=str,
        default=["Cora"],
        help="One or more dataset names supported by src/utils/dataset.py",
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
        default=str((Path(__file__).resolve().parent / "outputs").resolve()),
        help="Output root folder",
    )
    parser.add_argument("--max-subgraph-nodes", type=int, default=64, help="Maximum number of nodes per subgraph")
    parser.add_argument("--kmeans-k", type=int, default=8, help="Number of clusters")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--cluster-method",
        type=str,
        default="gmm",
        choices=["gmm", "kmeans"],
        help="Clustering method",
    )
    parser.add_argument(
        "--use-log1p",
        type=int,
        default=1,
        help="1: apply log1p on skewed count features before scaling; 0: disable",
    )
    parser.add_argument(
        "--remove-selfloops",
        type=int,
        default=1,
        help="1: remove self-loops before encoding; 0: keep self-loops",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("Stage-1: dataset definition and structural encoding")
    print("data_root=", args.data_root)
    print("output_root=", args.output_root)
    print("datasets=", args.datasets)
    print("remove_selfloops=", int(args.remove_selfloops))
    print("cluster_method=", args.cluster_method)
    print("use_log1p=", int(args.use_log1p))
    print("=" * 80)

    for name in args.datasets:
        config = Stage1Config(
            dataset=name,
            data_root=args.data_root,
            output_root=args.output_root,
            max_subgraph_nodes=args.max_subgraph_nodes,
            num_clusters=args.kmeans_k,
            seed=args.seed,
            remove_selfloops=bool(args.remove_selfloops),
            cluster_method=args.cluster_method,
            use_log1p=bool(args.use_log1p),
        )

        result = run_stage1(config)
        print(
            f"[DONE] dataset={result['dataset']} samples={result['num_samples']} "
            f"effective_k={result['effective_k']} output={result['output_dir']}"
        )
        print(f"summary={result['summary_json']}")


if __name__ == "__main__":
    main()
