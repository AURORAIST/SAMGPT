from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="dgl.backend.pytorch.sparse")
# Support both module execution and direct script execution.
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# Unified Stage-1 implementation with a self-loop switch.
from stage1 import Stage1Config, run_stage1


def parse_args() -> argparse.Namespace:
    # 仅暴露阶段一需要的参数，避免与主训练入口参数耦合。
    parser = argparse.ArgumentParser("Stage-1 dataset definition and structural encoding")
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=str,
        default=["Cora","Citeseer", "Pubmed", "Photo", "Computers"],
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
    parser.add_argument(
        "--max-subgraph-nodes",
        type=int,
        default=64,
        help="Maximum number of nodes per subgraph",
    )
    parser.add_argument(
        "--subgraph-hop-n",
        type=int,
        default=2,
        help="Neighborhood expansion hops for subgraph construction (default: 2)",
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=8,
        help="Number of GMM clusters",
    )
    parser.add_argument(
        "--encoding-mode",
        type=str,
        default="raw9",
        choices=["raw9", "decorrelated7"],
        help="Feature encoding mode used for clustering",
    )
    parser.add_argument("--seed", type=int, default=123456, help="Random seed for GMM")
    parser.add_argument(
        "--use-log1p",
        type=int,
        default=1,
        help="1: apply log1p on skewed count features before scaling; 0: disable",
    )
    parser.add_argument(
        "--remove-selfloops",
        type=int,
        default=0,
        help="1: remove self-loops before encoding; 0: keep self-loops",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 打印一次运行配置，便于记录实验日志。
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

    # 支持批量数据集串行处理，保证每个数据集输出到独立目录。
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

        # 单数据集执行：子图构建 -> 结构编码 -> log1p/标准化 -> 聚类 -> 落盘。
        result = run_stage1(config)
        print(
            f"[DONE] dataset={result['dataset']} samples={result['num_samples']} "
            f"effective_k={result['effective_k']} output={result['output_dir']}"
        )
        print(f"summary={result['summary_json']}")


if __name__ == "__main__":
    main()
