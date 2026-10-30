from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np


def save_similarity_curve_plot(curve_payload: dict, output_path: str, title: str) -> None:
    """Save similarity density curve with q25/q50/q75 vertical lines."""
    if not isinstance(curve_payload, dict):
        return

    xs = np.asarray(curve_payload.get('bin_centers', []), dtype=np.float64)
    ys = np.asarray(curve_payload.get('density', []), dtype=np.float64)
    if xs.size == 0 or ys.size == 0 or xs.size != ys.size:
        return

    q25 = float(curve_payload.get('q25', np.nan))
    q50 = float(curve_payload.get('q50', np.nan))
    q75 = float(curve_payload.get('q75', np.nan))

    fig = plt.figure(figsize=(9, 5))
    plt.plot(xs, ys, color='#1f77b4', linewidth=1.8, label='similarity density')

    if np.isfinite(q25):
        plt.axvline(q25, color='#d62728', linestyle='--', linewidth=1.2, label=f'q25={q25:.3f}')
    if np.isfinite(q50):
        plt.axvline(q50, color='#2ca02c', linestyle='--', linewidth=1.2, label=f'q50={q50:.3f}')
    if np.isfinite(q75):
        plt.axvline(q75, color='#ff7f0e', linestyle='--', linewidth=1.2, label=f'q75={q75:.3f}')

    plt.title(title)
    plt.xlabel('cosine similarity')
    plt.ylabel('density')
    plt.grid(alpha=0.25)
    plt.legend(loc='best')
    plt.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def dump_similarity_curves(
    curve_stats: dict,
    curve_dir: str,
    epoch: int,
    save_plot: bool = False,
) -> dict[str, str]:
    """Dump lightweight similarity summaries, with optional plot export."""
    outputs: dict[str, str] = {}
    if not isinstance(curve_stats, dict):
        return outputs

    os.makedirs(curve_dir, exist_ok=True)
    for branch in ['single', 'feature', 'structure']:
        payload = curve_stats.get(branch, None)
        if not isinstance(payload, dict):
            continue

        summary_payload = {
            'q25': float(payload.get('q25', np.nan)),
            'q50': float(payload.get('q50', np.nan)),
            'q75': float(payload.get('q75', np.nan)),
            'num_pairs': int(payload.get('num_pairs', 0)),
        }
        summary_path = os.path.join(curve_dir, f'similarity_curve_{branch}_epoch_{int(epoch):04d}.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_payload, f, indent=2)

        outputs[f'{branch}_json'] = summary_path
        if save_plot:
            png_path = os.path.join(curve_dir, f'similarity_curve_{branch}_epoch_{int(epoch):04d}.png')
            save_similarity_curve_plot(
                curve_payload=payload,
                output_path=png_path,
                title=f'EGRAPHCL {branch} similarity curve (epoch={int(epoch)})',
            )
            outputs[f'{branch}_png'] = png_path

    return outputs
