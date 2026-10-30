from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stage_pipeline import Stage1Config, Stage2Config, run_stage1, run_stage2

RAW_KEYS = [
    "num_nodes",
    "num_edges",
    "avg_degree",
    "max_degree",
    "var_degree",
    "density",
    "avg_clustering",
    "triangles",
    "diameter",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Grid-evaluate clustering settings and verify stage_pipeline")
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer", "pubmed", "photo", "computers"])
    parser.add_argument(
        "--outputs-root",
        type=str,
        default=str((PROJECT_ROOT / "result" / "stage_outputs" / "stage1").resolve()),
    )
    parser.add_argument(
        "--stage2-outputs-root",
        type=str,
        default=str((PROJECT_ROOT / "result" / "stage_outputs" / "stage2").resolve()),
    )
    parser.add_argument("--data-root", type=str, default=str((PROJECT_ROOT / "data").resolve()))
    parser.add_argument("--k-list", nargs="+", type=int, default=[8, 10, 12])
    parser.add_argument("--covariance-types", nargs="+", type=str, default=["full", "diag"])
    parser.add_argument("--encoding-mode", type=str, default="raw9", choices=["raw9", "decorrelated7"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-csv", type=str, default="")
    parser.add_argument("--out-json", type=str, default="")

    # Stage pipeline verification switches.
    parser.add_argument("--verify-stage1", type=int, default=1, help="1: run stage1 before grid evaluation")
    parser.add_argument("--verify-stage2", type=int, default=0, help="1: run stage2 and verify outputs")
    parser.add_argument("--stage1-max-subgraph-nodes", type=int, default=64)
    parser.add_argument("--stage1-subgraph-hop-n", type=int, default=2)
    parser.add_argument("--stage1-num-clusters", type=int, default=8)
    parser.add_argument("--stage1-use-log1p", type=int, default=1)
    parser.add_argument("--stage1-remove-selfloops", type=int, default=0)

    parser.add_argument("--stage2-top-m-per-cluster", type=int, default=5)
    parser.add_argument("--stage2-alpha", type=float, default=1.0)
    parser.add_argument("--stage2-beta", type=float, default=1.0)
    parser.add_argument("--stage2-gamma", type=float, default=1.0)
    return parser.parse_args()


def build_features(df: pd.DataFrame, mode: str) -> np.ndarray:
    x = df[RAW_KEYS].astype(float).values
    if mode == "raw9":
        idx = [0, 1, 3, 4, 7, 8]
        out = x.copy()
        out[:, idx] = np.log1p(np.clip(out[:, idx], 0.0, None))
        return out

    num_nodes = np.maximum(x[:, 0], 0.0)
    num_edges = np.maximum(x[:, 1], 0.0)
    avg_degree = np.maximum(x[:, 2], 0.0)
    max_degree = np.maximum(x[:, 3], 0.0)
    var_degree = np.maximum(x[:, 4], 0.0)
    density = np.clip(x[:, 5], 0.0, 1.0)
    avg_clustering = np.clip(x[:, 6], 0.0, 1.0)
    triangles = np.maximum(x[:, 7], 0.0)
    diameter = np.maximum(x[:, 8], 0.0)

    tri_denom = np.maximum(
        num_nodes * np.maximum(num_nodes - 1.0, 0.0) * np.maximum(num_nodes - 2.0, 0.0) / 6.0,
        1.0,
    )

    return np.column_stack(
        [
            np.log1p(num_edges),
            max_degree / np.maximum(num_nodes - 1.0, 1.0),
            np.sqrt(var_degree) / np.maximum(avg_degree, 1e-6),
            density,
            avg_clustering,
            triangles / tri_denom,
            diameter / np.maximum(num_nodes - 1.0, 1.0),
        ]
    ).astype(np.float64)


def evaluate_one(Xs: np.ndarray, k: int, cov: str, seed: int) -> dict[str, float]:
    model = GaussianMixture(n_components=k, covariance_type=cov, random_state=seed, reg_covar=1e-6)
    labels = model.fit_predict(Xs)
    probs = model.predict_proba(Xs)
    max_prob = probs.max(axis=1)

    sil = float(silhouette_score(Xs, labels))
    dbi = float(davies_bouldin_score(Xs, labels))
    counts = np.bincount(labels, minlength=k)
    largest_cluster_ratio = float(counts.max() / counts.sum())
    mean_max_prob = float(max_prob.mean())
    low_conf_ratio = float((max_prob < 0.6).mean())

    score = sil - 0.2 * dbi - 0.3 * largest_cluster_ratio + 0.05 * mean_max_prob - 0.1 * low_conf_ratio
    return {
        "silhouette": sil,
        "davies_bouldin": dbi,
        "largest_cluster_ratio": largest_cluster_ratio,
        "mean_max_prob": mean_max_prob,
        "low_conf_ratio": low_conf_ratio,
        "score": float(score),
    }


def maybe_run_stage_pipeline(args: argparse.Namespace, ds: str, outputs_root: Path) -> tuple[Path, str]:
    ds_name = ds[0].upper() + ds[1:] if ds and ds[0].islower() else ds

    if args.verify_stage1 == 1:
        run_stage1(
            Stage1Config(
                dataset=ds_name,
                data_root=str(Path(args.data_root).resolve()),
                output_root=str(outputs_root),
                max_subgraph_nodes=int(args.stage1_max_subgraph_nodes),
                num_clusters=int(args.stage1_num_clusters),
                seed=int(args.seed),
                remove_selfloops=bool(args.stage1_remove_selfloops),
                use_log1p=bool(args.stage1_use_log1p),
                subgraph_hop_n=int(args.stage1_subgraph_hop_n),
                encoding_mode=args.encoding_mode,
            )
        )

    csv_path = outputs_root / ds.lower() / "structural_encodings.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"missing structural encodings: {csv_path}")

    if args.verify_stage2 == 1:
        stage2_root = Path(args.stage2_outputs_root).resolve()
        run_stage2(
            Stage2Config(
                input_csv=str(csv_path),
                output_dir=str((stage2_root / ds.lower()).resolve()),
                top_m_per_cluster=int(args.stage2_top_m_per_cluster),
                alpha=float(args.stage2_alpha),
                beta=float(args.stage2_beta),
                gamma=float(args.stage2_gamma),
            )
        )

    return csv_path, ds_name


def main() -> None:
    args = parse_args()
    outputs_root = Path(args.outputs_root).resolve()
    out_csv = Path(args.out_csv).resolve() if args.out_csv else (outputs_root / "cluster_grid_results.csv")
    out_json = Path(args.out_json).resolve() if args.out_json else (outputs_root / "cluster_grid_recommendations.json")

    rows: list[dict[str, float | int | str]] = []
    rec: dict[str, dict[str, float | int | str]] = {}

    for ds in args.datasets:
        try:
            csv_path, ds_name = maybe_run_stage_pipeline(args, ds, outputs_root)
        except Exception as e:
            print(f"[skip] {ds}: stage-pipeline verify failed: {e}")
            continue

        df = pd.read_csv(csv_path)
        if len(df) < 10:
            print(f"[skip] {ds}: too few samples ({len(df)})")
            continue

        X = build_features(df=df, mode=args.encoding_mode)
        Xs = StandardScaler().fit_transform(X)

        best_row: dict[str, float | int | str] | None = None
        for k in args.k_list:
            k_eff = min(max(2, int(k)), len(df) - 1)
            for cov in args.covariance_types:
                met = evaluate_one(Xs=Xs, k=k_eff, cov=cov, seed=args.seed)
                row = {
                    "dataset": ds.lower(),
                    "dataset_name": ds_name,
                    "n_samples": int(len(df)),
                    "encoding_mode": args.encoding_mode,
                    "k": int(k_eff),
                    "covariance_type": cov,
                    "verify_stage1": int(args.verify_stage1),
                    "verify_stage2": int(args.verify_stage2),
                    **met,
                }
                rows.append(row)
                if best_row is None or float(row["score"]) > float(best_row["score"]):
                    best_row = row

        if best_row is not None:
            rec[ds.lower()] = best_row
            print(
                f"[best] {ds.lower()} k={best_row['k']} cov={best_row['covariance_type']} "
                f"sil={best_row['silhouette']:.4f} dbi={best_row['davies_bouldin']:.4f} "
                f"lcr={best_row['largest_cluster_ratio']:.4f}"
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "dataset_name",
                "n_samples",
                "encoding_mode",
                "k",
                "covariance_type",
                "verify_stage1",
                "verify_stage2",
                "silhouette",
                "davies_bouldin",
                "largest_cluster_ratio",
                "mean_max_prob",
                "low_conf_ratio",
                "score",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    out_json.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"[done] wrote {out_csv}")
    print(f"[done] wrote {out_json}")


if __name__ == "__main__":
    main()
