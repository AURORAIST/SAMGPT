from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def configure_chinese_font(font_path: str) -> str:
    if font_path:
        p = Path(font_path).expanduser().resolve()
        if p.exists():
            font_manager.fontManager.addfont(str(p))
            font_name = font_manager.FontProperties(fname=str(p)).get_name()
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["axes.unicode_minus"] = False
            return f"{font_name} ({p})"

    available = {f.name for f in font_manager.fontManager.ttflist}
    candidates = [
        "Noto Sans CJK SC",
        "Source Han Sans CN",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
        "Arial Unicode MS",
    ]
    chosen = [name for name in candidates if name in available]
    plt.rcParams["font.sans-serif"] = chosen if chosen else candidates
    plt.rcParams["axes.unicode_minus"] = False
    return chosen[0] if chosen else "<none-found>"


def _scatter_by_labels(ax, emb2d: np.ndarray, labels: np.ndarray, title: str) -> None:
    valid_mask = labels >= 0
    if np.any(valid_mask):
        sc = ax.scatter(
            emb2d[valid_mask, 0],
            emb2d[valid_mask, 1],
            c=labels[valid_mask],
            cmap="tab10",
            s=18,
            alpha=0.85,
            edgecolors="none",
        )
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    if np.any(~valid_mask):
        ax.scatter(
            emb2d[~valid_mask, 0],
            emb2d[~valid_mask, 1],
            c="#bbbbbb",
            s=14,
            alpha=0.6,
            edgecolors="none",
            label="未知标签",
        )
        ax.legend(loc="best")

    ax.set_title(title)
    ax.set_xlabel("维度1")
    ax.set_ylabel("维度2")


def save_embedding_comparison(
    embedding_name: str,
    emb2d: np.ndarray,
    cluster_labels: np.ndarray,
    true_labels: np.ndarray | None,
    out_png: Path,
    title_prefix: str,
) -> None:
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    _scatter_by_labels(ax1, emb2d, cluster_labels.astype(np.int64), f"{embedding_name} 按聚类标签着色")

    ax2 = fig.add_subplot(1, 2, 2)
    if true_labels is None:
        empty = np.full(cluster_labels.shape[0], -1, dtype=np.int64)
        _scatter_by_labels(ax2, emb2d, empty, f"{embedding_name} 按真实标签着色（不可用）")
    else:
        _scatter_by_labels(ax2, emb2d, true_labels.astype(np.int64), f"{embedding_name} 按真实标签着色")

    fig.suptitle(title_prefix, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_sweep_curves(rows: list[dict], out_png: Path) -> None:
    ks = np.array([int(r["num_components"]) for r in rows], dtype=np.int64)
    sil = np.array([float(r["silhouette"]) for r in rows], dtype=np.float64)
    dbi = np.array([float(r["davies_bouldin"]) for r in rows], dtype=np.float64)
    ch = np.array([float(r["calinski_harabasz"]) for r in rows], dtype=np.float64)
    aic = np.array([float(r["aic"]) for r in rows], dtype=np.float64)
    bic = np.array([float(r["bic"]) for r in rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(ks, sil, marker="o", label="Silhouette")
    ax.plot(ks, dbi, marker="s", label="DBI")
    ax.set_title("内部指标（越高越好/DBI越低越好）")
    ax.set_xlabel("num_components")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ax.plot(ks, ch, marker="o", color="#d62728", label="CH")
    ax.set_title("Calinski-Harabasz（越高越好）")
    ax.set_xlabel("num_components")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.plot(ks, aic, marker="o", color="#2ca02c", label="AIC")
    ax.plot(ks, bic, marker="s", color="#1f77b4", label="BIC")
    ax.set_title("信息准则（越低越好）")
    ax.set_xlabel("num_components")
    ax.legend()
    ax.grid(alpha=0.25)

    has_supervised = any(np.isfinite(float(r.get("nmi", np.nan))) for r in rows)
    ax = axes[1, 1]
    if has_supervised:
        nmi = np.array([float(r.get("nmi", np.nan)) for r in rows], dtype=np.float64)
        ari = np.array([float(r.get("ari", np.nan)) for r in rows], dtype=np.float64)
        acc = np.array([float(r.get("acc", np.nan)) for r in rows], dtype=np.float64)
        ax.plot(ks, nmi, marker="o", label="NMI")
        ax.plot(ks, ari, marker="s", label="ARI")
        ax.plot(ks, acc, marker="^", label="ACC")
        ax.set_title("监督指标（越高越好）")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "无可用真实标签，跳过监督指标", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("监督指标")
    ax.set_xlabel("num_components")
    ax.grid(alpha=0.25)

    fig.suptitle("GMM 组件数扫描结果", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
