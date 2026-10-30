import csv
import json
import logging
import os
import time

os.environ.setdefault('MPLCONFIGDIR', os.path.join('/tmp', 'matplotlib'))

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from preprompt import pca_compression
from torch_geometric.loader import DataLoader

from . import process
from .dataset import load_dataset
from .execute_helpers import resolve_device
from usp_sam import ego_subgraph_mask, ego_subgraph_mask_block, _mask_to_csr


__all__ = [
    '_append_csv_row',
    '_build_ego_exclusion_rows_from_cache',
    'build_or_load_ego_cache',
    '_class_count_json',
    '_csr_mask_to_row_lists',
    '_cuda_mem_mb',
    '_delta_norm_from_snapshot',
    '_ego_block_cache_dir',
    '_ego_full_cache_path',
    '_identity_like_adj',
    '_offset_ego_exclusion_rows',
    '_permute_target_edges',
    '_snapshot_named_params',
    '_split_integrity_stats',
    '_sync_if_cuda',
    '_usp_device',
    '_usp_edge_dropout',
    '_usp_feature_dropout',
    '_usp_fewshot_path',
    '_usp_load_graph',
    '_usp_save_loss_curves_plot',
    '_usp_read_fewshot',
    '_usp_safe_torch_load',
    '_usp_sparse_to_device',
    '_usp_train_downstream_classifier',
    '_validate_neg_exclude_scope',
    '_write_formal_batch1_summary',
    '_write_query_objective_summary',
    '_write_rows_csv',
    '_write_sanity_row',
    '_write_sanity_summary',
]


def _usp_safe_torch_load(path, map_location=None):
    return torch.load(path, map_location=map_location, weights_only=False)


def _usp_device(args):
    return resolve_device(args.device, args.gpu)


def _usp_sparse_to_device(adj, device, sparse=True):
    if sparse:
        return process.sparse_mx_to_torch_sparse_tensor(adj).coalesce().to(device)
    return torch.FloatTensor(adj.todense()).to(device)


def _usp_feature_dropout(features, drop_prob):
    if drop_prob <= 0:
        return features
    keep = torch.rand_like(features) > drop_prob
    return features * keep.float()


def _usp_edge_dropout(adj, drop_prob):
    if drop_prob <= 0:
        return adj
    if not adj.is_sparse:
        mask = (torch.rand_like(adj) > drop_prob).float()
        eye = torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
        return adj * mask + eye
    adj = adj.coalesce()
    indices = adj.indices()
    values = adj.values()
    keep = torch.rand(values.size(0), device=values.device) > drop_prob
    row, col = indices[0], indices[1]
    keep = keep | (row == col)
    new_adj = torch.sparse_coo_tensor(indices[:, keep], values[keep], adj.size(), device=adj.device)
    return new_adj.coalesce()


def _permute_target_edges(adj, seed):
    """Randomly rewire target edges while preserving node count and edge count."""
    if adj.is_sparse:
        dense = adj.to_dense()
    else:
        dense = adj
    device = dense.device
    n = dense.size(0)
    undirected = torch.triu(dense.bool(), diagonal=1)
    edge_count = int(undirected.sum().item())
    generator = torch.Generator()
    generator.manual_seed(seed)
    possible = n * (n - 1) // 2
    edge_count = min(edge_count, possible)
    chosen = torch.randperm(possible, generator=generator)[:edge_count]
    rows = []
    cols = []
    # Invert upper-triangular pair index without materializing all pairs.
    for idx in chosen.tolist():
        remaining = idx
        u = 0
        width = n - 1
        while remaining >= width:
            remaining -= width
            u += 1
            width -= 1
        v = u + 1 + remaining
        rows.append(u)
        cols.append(v)
    rewired = torch.eye(n, dtype=dense.dtype, device=device)
    if rows:
        row = torch.tensor(rows, dtype=torch.long, device=device)
        col = torch.tensor(cols, dtype=torch.long, device=device)
        rewired[row, col] = 1.0
        rewired[col, row] = 1.0
    if adj.is_sparse:
        return rewired.to_sparse().coalesce()
    return rewired


def _identity_like_adj(adj):
    n = adj.size(0)
    device = adj.device
    dtype = adj.dtype
    idx = torch.arange(n, device=device)
    if adj.is_sparse:
        indices = torch.stack([idx, idx], dim=0)
        values = torch.ones(n, dtype=dtype, device=device)
        return torch.sparse_coo_tensor(indices, values, adj.size(), device=device).coalesce()
    return torch.eye(n, dtype=dtype, device=device)


def _class_count_json(labels, idx, num_classes):
    selected = labels[idx].detach().cpu()
    counts = torch.bincount(selected.long(), minlength=num_classes).tolist()
    return json.dumps({str(i): int(v) for i, v in enumerate(counts)}, ensure_ascii=False)


def _split_integrity_stats(labels, support_idx, support_labels, query_idx, original_labels=None):
    original_labels = labels if original_labels is None else original_labels
    support_set = set(support_idx.detach().cpu().tolist())
    query_set = set(query_idx.detach().cpu().tolist())
    overlap = len(support_set.intersection(query_set))
    if overlap > 0:
        raise ValueError(f"Support/query split overlap detected: {overlap} nodes.")
    if support_labels is not None and not torch.equal(original_labels[support_idx].long(), support_labels.long()):
        raise ValueError("Few-shot support label file does not match labels[support_idx].")
    if not torch.equal(labels[query_idx].long(), original_labels[query_idx].long()):
        raise ValueError("Label shuffle modified query labels; only support labels may be shuffled.")
    num_classes = int(original_labels.max().item() + 1)
    return {
        'num_support': int(support_idx.numel()),
        'num_query': int(query_idx.numel()),
        'num_classes': num_classes,
        'support_class_count': _class_count_json(labels, support_idx, num_classes),
        'query_class_count': _class_count_json(original_labels, query_idx, num_classes),
        'support_query_overlap': overlap,
        'split_integrity_passed': True,
    }


def _write_rows_csv(path, rows):
    if not rows:
        return
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)


def _write_sanity_summary(path, rows):
    groups = {}
    for row in rows:
        key = (
            row.get('method'),
            row.get('epoch'),
            row.get('label_shuffle'),
            row.get('feature_permuted'),
            row.get('edge_permuted'),
        )
        groups.setdefault(key, []).append(row)
    summary_rows = []
    for (method, epoch, label_shuffle, feature_permuted, edge_permuted), group in groups.items():
        acc = np.array([float(r['acc']) for r in group], dtype=float)
        macro = np.array([float(r['macro_f1']) for r in group], dtype=float)
        summary_rows.append({
            'method': method,
            'epoch': epoch,
            'label_shuffle': label_shuffle,
            'feature_permuted': feature_permuted,
            'edge_permuted': edge_permuted,
            'n': len(group),
            'mean_acc': float(acc.mean()),
            'std_acc': float(acc.std(ddof=0)),
            'max_acc': float(acc.max()),
            'p95_acc': float(np.percentile(acc, 95)),
            'mean_macro_f1': float(macro.mean()),
            'std_macro_f1': float(macro.std(ddof=0)),
        })
    _write_rows_csv(path, summary_rows)


def _write_formal_batch1_summary(path, rows):
    groups = {}
    for row in rows:
        key = (row.get('method'), row.get('shot'))
        groups.setdefault(key, []).append(row)
    summary_rows = []
    metrics = ['acc', 'balanced_acc', 'macro_f1', 'micro_f1', 'margin', 'compactness', 'separation']
    for (method, shot), group in groups.items():
        out = {
            'method': method,
            'shot': shot,
            'n': len(group),
        }
        for metric in metrics:
            values = np.array([float(r[metric]) for r in group], dtype=float)
            out[f'mean_{metric}'] = float(values.mean())
            out[f'std_{metric}'] = float(values.std(ddof=0))
        summary_rows.append(out)
    _write_rows_csv(path, summary_rows)


def _write_query_objective_summary(path, rows):
    groups = {}
    for row in rows:
        key = (
            row.get('method'),
            row.get('objective_active'),
            row.get('query_mode'),
            row.get('shot'),
        )
        groups.setdefault(key, []).append(row)
    summary_rows = []
    metrics = [
        'acc',
        'balanced_acc',
        'macro_f1',
        'micro_f1',
        'margin',
        'compactness',
        'separation',
        'encoder_param_delta_norm',
        'readout_param_delta_norm',
        'prompt_param_delta_norm',
    ]
    for (method, objective_active, query_mode, shot), group in groups.items():
        out = {
            'method': method,
            'objective_active': objective_active,
            'query_mode': query_mode,
            'shot': shot,
            'n': len(group),
        }
        for metric in metrics:
            values = np.array([float(r[metric]) for r in group], dtype=float)
            out[f'mean_{metric}'] = float(values.mean())
            out[f'std_{metric}'] = float(values.std(ddof=0))
        summary_rows.append(out)
    _write_rows_csv(path, summary_rows)


def _snapshot_named_params(module, name_filter=None):
    snap = {}
    for name, param in module.named_parameters():
        if name_filter is None or name_filter(name, param):
            snap[name] = param.detach().clone()
    return snap


def _delta_norm_from_snapshot(module, snapshot, name_filter=None):
    total = 0.0
    for name, param in module.named_parameters():
        if name not in snapshot:
            continue
        if name_filter is not None and not name_filter(name, param):
            continue
        diff = param.detach() - snapshot[name]
        total += float(diff.pow(2).sum().item())
    return total ** 0.5


def _usp_load_graph(dataset_name, unify_dim, cache_dir, device, sparse=True, data_root=None):
    cache_prefix = os.path.join(cache_dir, dataset_name)
    feature_path = f'{cache_prefix}_feature.pt'
    adj_path = f'{cache_prefix}_adj.pt'

    rebuild_cache = not (os.path.exists(feature_path) and os.path.exists(adj_path))
    if not rebuild_cache:
        cached_feature = _usp_safe_torch_load(feature_path, map_location='cpu')
        rebuild_cache = cached_feature.size(1) != unify_dim

    if rebuild_cache:
        data = load_dataset(dataset_name, path=data_root or './data')[0]
        feature, adj = process.process_tu(data, data.x.shape[1])
        feature = torch.FloatTensor(pca_compression(feature, k=unify_dim))
        torch.save(feature, feature_path)
        torch.save(adj, adj_path)

    feature = _usp_safe_torch_load(feature_path).to(device)
    raw_adj = _usp_safe_torch_load(adj_path)
    norm_adj = process.normalize_adj(raw_adj + sp.eye(raw_adj.shape[0]))
    torch_adj = _usp_sparse_to_device(norm_adj, device, sparse=sparse)
    return feature, raw_adj, norm_adj, torch_adj


def _usp_fewshot_path(parent_directory, dataset_name, shot, split):
    base = os.path.join(parent_directory, 'data', f'fewshot_{dataset_name.lower()}', f'{shot}-shot_{dataset_name.lower()}', str(split))
    return os.path.join(base, 'idx.pt'), os.path.join(base, 'labels.pt')


def _usp_read_fewshot(parent_directory, dataset_name, shot, split, device):
    idx_path, label_path = _usp_fewshot_path(parent_directory, dataset_name, shot, split)
    if not (os.path.exists(idx_path) and os.path.exists(label_path)):
        raise FileNotFoundError(f'Missing few-shot split files: {idx_path}, {label_path}')
    idx = _usp_safe_torch_load(idx_path).type(torch.long).to(device)
    labels = _usp_safe_torch_load(label_path).type(torch.long).squeeze().to(device)
    return idx, labels


def _usp_save_loss_curves_plot(curves, output_path, title='Loss curves', ylabel='Loss'):
    valid_curves = []
    for curve in curves:
        values = [float(v) for v in curve.get('values', []) if v is not None and np.isfinite(float(v))]
        if values:
            valid_curves.append({
                'label': str(curve.get('label', f'curve_{len(valid_curves)}')),
                'values': values,
            })
    if not valid_curves:
        msg = f'[CURVE] no finite values, skip plot: {output_path}'
        print(msg)
        logging.info(msg)
        return None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    max_len = max(len(curve['values']) for curve in valid_curves)
    aligned = np.full((len(valid_curves), max_len), np.nan, dtype=float)
    for row, curve in enumerate(valid_curves):
        values = np.asarray(curve['values'], dtype=float)
        aligned[row, : values.shape[0]] = values

    try:
        mpl_config_dir = os.path.join('/tmp', 'matplotlib')
        os.makedirs(mpl_config_dir, exist_ok=True)
        os.environ.setdefault('MPLCONFIGDIR', mpl_config_dir)
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:
        svg_path = os.path.splitext(output_path)[0] + '.svg'
        _usp_save_loss_curves_svg(valid_curves, aligned, svg_path, title, ylabel)
        summary_values = [value for curve in valid_curves for value in curve['values']]
        msg = (
            f'[CURVE] matplotlib unavailable ({exc}); saved SVG loss plot: {svg_path} '
            f'curves={len(valid_curves)} n={len(summary_values)} '
            f'first_mean={np.nanmean(aligned[:, 0]):.6f} '
            f'last_mean={np.nanmean(aligned[:, -1]):.6f} '
            f'min={min(summary_values):.6f} max={max(summary_values):.6f}'
        )
        print(msg)
        logging.info(msg)
        return svg_path

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    show_each_label = len(valid_curves) <= 12
    for row, curve in enumerate(valid_curves):
        values = aligned[row, : len(curve['values'])]
        epochs = np.arange(1, values.shape[0] + 1)
        if show_each_label:
            ax.plot(epochs, values, linewidth=1.6, alpha=0.9, label=curve['label'])
        else:
            ax.plot(epochs, values, color='0.65', linewidth=0.9, alpha=0.35)

    if not show_each_label:
        mean_curve = np.nanmean(aligned, axis=0)
        mean_epochs = np.arange(1, mean_curve.shape[0] + 1)
        ax.plot(mean_epochs, mean_curve, color='#1f77b4', linewidth=2.4, label='mean')
    ax.set_title(title)
    ax.set_xlabel('Downstream epoch')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    summary_values = [value for curve in valid_curves for value in curve['values']]
    msg = (
        f'[CURVE] saved loss plot: {output_path} '
        f'curves={len(valid_curves)} n={len(summary_values)} '
        f'first_mean={np.nanmean(aligned[:, 0]):.6f} '
        f'last_mean={np.nanmean(aligned[:, -1]):.6f} '
        f'min={min(summary_values):.6f} max={max(summary_values):.6f}'
    )
    print(msg)
    logging.info(msg)
    return output_path


def _usp_save_loss_curves_svg(valid_curves, aligned, output_path, title, ylabel):
    import html

    width = 1000
    height = 620
    left = 82
    right = 28
    top = 58
    bottom = 82
    plot_w = width - left - right
    plot_h = height - top - bottom
    finite_values = aligned[np.isfinite(aligned)]
    y_min = float(np.min(finite_values))
    y_max = float(np.max(finite_values))
    if y_max <= y_min:
        y_pad = max(abs(y_min) * 0.05, 1e-6)
        y_min -= y_pad
        y_max += y_pad
    else:
        y_pad = (y_max - y_min) * 0.06
        y_min -= y_pad
        y_max += y_pad

    def map_x(epoch):
        denom = max(aligned.shape[1] - 1, 1)
        return left + ((epoch - 1) / denom) * plot_w

    def map_y(value):
        return top + (1.0 - ((value - y_min) / (y_max - y_min))) * plot_h

    def polyline(values):
        points = []
        for idx, value in enumerate(values, start=1):
            if np.isfinite(value):
                points.append(f'{map_x(idx):.2f},{map_y(float(value)):.2f}')
        return ' '.join(points)

    show_each_label = len(valid_curves) <= 12
    palette = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#17becf', '#8c564b', '#e377c2']
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" font-family="Arial" font-size="20" fill="#222">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222" stroke-width="1"/>',
    ]
    for tick in range(6):
        ratio = tick / 5
        y = top + ratio * plot_h
        value = y_max - ratio * (y_max - y_min)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#ddd" stroke-width="1"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="12" fill="#555">{value:.4f}</text>')
    for tick in range(6):
        ratio = tick / 5
        x = left + ratio * plot_w
        epoch = 1 + ratio * max(aligned.shape[1] - 1, 0)
        lines.append(f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 5}" stroke="#222" stroke-width="1"/>')
        lines.append(f'<text x="{x:.2f}" y="{top + plot_h + 22}" text-anchor="middle" font-family="Arial" font-size="12" fill="#555">{epoch:.0f}</text>')

    if show_each_label:
        for row, curve in enumerate(valid_curves):
            color = palette[row % len(palette)]
            lines.append(
                f'<polyline points="{polyline(aligned[row, : len(curve["values"])])}" '
                f'fill="none" stroke="{color}" stroke-width="2" stroke-opacity="0.9"/>'
            )
            legend_y = top + 20 + row * 18
            lines.append(f'<line x1="{left + plot_w - 155}" y1="{legend_y}" x2="{left + plot_w - 130}" y2="{legend_y}" stroke="{color}" stroke-width="2"/>')
            lines.append(f'<text x="{left + plot_w - 124}" y="{legend_y + 4}" font-family="Arial" font-size="12" fill="#333">{html.escape(curve["label"])}</text>')
    else:
        for row, curve in enumerate(valid_curves):
            lines.append(
                f'<polyline points="{polyline(aligned[row, : len(curve["values"])])}" '
                'fill="none" stroke="#999" stroke-width="1" stroke-opacity="0.32"/>'
            )
        mean_curve = np.nanmean(aligned, axis=0)
        lines.append(f'<polyline points="{polyline(mean_curve)}" fill="none" stroke="#1f77b4" stroke-width="3"/>')
        lines.append(f'<line x1="{left + plot_w - 105}" y1="{top + 22}" x2="{left + plot_w - 72}" y2="{top + 22}" stroke="#1f77b4" stroke-width="3"/>')
        lines.append(f'<text x="{left + plot_w - 64}" y="{top + 26}" font-family="Arial" font-size="13" fill="#333">mean</text>')

    lines.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 24}" text-anchor="middle" font-family="Arial" font-size="14" fill="#333">Downstream epoch</text>')
    lines.append(f'<text x="20" y="{top + plot_h / 2:.1f}" text-anchor="middle" transform="rotate(-90 20 {top + plot_h / 2:.1f})" font-family="Arial" font-size="14" fill="#333">{html.escape(ylabel)}</text>')
    lines.append('</svg>')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return output_path


def _usp_train_downstream_classifier(
    classifier,
    node_embeds,
    subgraph_embeds,
    support_idx,
    support_labels,
    epochs,
    lr,
    log_prefix='[EVAL][downstream]',
    log_interval=10,
    eval_idx=None,
    eval_labels=None,
):
    class_counts = torch.bincount(support_labels.long(), minlength=classifier.nb_classes)
    attention_degenerate = classifier.prototype == 'attention' and bool((class_counts <= 1).all().item())
    if classifier.query != 'hybrid' and attention_degenerate:
        classifier(node_embeds, subgraph_embeds, support_idx, support_idx, support_labels, update_prototypes=True)
        msg = (
            f'{log_prefix} epoch 1/1 loss=0.000000 support_acc=100.00 '
            f'note=no_effective_downstream_params_for_1shot_attention_{classifier.query}'
        )
        if eval_idx is not None and eval_labels is not None:
            with torch.no_grad():
                logits = classifier(node_embeds, subgraph_embeds, eval_idx)
                preds = torch.argmax(logits, dim=1)
                eval_acc = (preds == eval_labels).float().mean().item() * 100.0
            msg += f' eval_acc={eval_acc:.2f}'
        print(msg)
        logging.info(msg)
        return {
            'loss_history': [0.0],
            'best_eval_acc': eval_acc if eval_idx is not None and eval_labels is not None else None,
            'best_eval_epoch': 1,
        }
    if classifier.prototype == 'mean' and classifier.query != 'hybrid':
        classifier(node_embeds, subgraph_embeds, support_idx, support_idx, support_labels, update_prototypes=True)
        msg = f'{log_prefix} epoch 1/1 loss=0.000000 support_acc=100.00'
        if eval_idx is not None and eval_labels is not None:
            with torch.no_grad():
                logits = classifier(node_embeds, subgraph_embeds, eval_idx)
                preds = torch.argmax(logits, dim=1)
                eval_acc = (preds == eval_labels).float().mean().item() * 100.0
            msg += f' eval_acc={eval_acc:.2f}'
        print(msg)
        logging.info(msg)
        return {
            'loss_history': [0.0],
            'best_eval_acc': eval_acc if eval_idx is not None and eval_labels is not None else None,
            'best_eval_epoch': 1,
        }
    opt = torch.optim.Adam(classifier.parameters(), lr=lr)
    best_eval_acc = None
    best_eval_epoch = None
    loss_history = []
    for epoch in range(epochs):
        opt.zero_grad()
        logits = classifier(
            node_embeds,
            subgraph_embeds,
            support_idx,
            support_idx=support_idx,
            support_labels=support_labels,
            update_prototypes=True,
        )
        loss = F.cross_entropy(logits, support_labels)
        loss_history.append(float(loss.detach()))
        loss.backward()
        opt.step()
        should_log = ((epoch + 1) % max(int(log_interval), 1) == 0) or (epoch + 1 == epochs)
        if should_log:
            with torch.no_grad():
                support_pred = torch.argmax(logits, dim=1)
                support_acc = (support_pred == support_labels).float().mean().item() * 100.0
                msg = f'{log_prefix} epoch {epoch + 1}/{epochs} loss={loss.item():.6f} support_acc={support_acc:.2f}'
                if eval_idx is not None and eval_labels is not None:
                    eval_logits = classifier(node_embeds, subgraph_embeds, eval_idx)
                    eval_pred = torch.argmax(eval_logits, dim=1)
                    eval_acc = (eval_pred == eval_labels).float().mean().item() * 100.0
                    if best_eval_acc is None or eval_acc > best_eval_acc:
                        best_eval_acc = eval_acc
                        best_eval_epoch = epoch + 1
                    msg += f' eval_acc={eval_acc:.2f} best_eval_acc={best_eval_acc:.2f}@{best_eval_epoch}'
            print(msg)
            logging.info(msg)
    return {
        'loss_history': loss_history,
        'best_eval_acc': best_eval_acc,
        'best_eval_epoch': best_eval_epoch,
    }



def _ego_full_cache_path(cache_dir, name, k, max_neighbors, num_nodes):
    mn = "none" if max_neighbors is None else str(max_neighbors)
    return os.path.join(cache_dir, f"{name}_ego_full_v2_csr_k{k}_max{mn}_N{num_nodes}.pt")


def _ego_block_cache_dir(cache_dir, name, k, max_neighbors, num_nodes, block_size):
    mn = "none" if max_neighbors is None else str(max_neighbors)
    return os.path.join(cache_dir, f"{name}_ego_blocks_v2_csr_k{k}_max{mn}_N{num_nodes}_B{block_size}")


def _cuda_mem_mb(device):
    if isinstance(device, torch.device) and device.type == "cuda":
        return {
            "gpu_allocated_mb": float(torch.cuda.memory_allocated(device) / 1024 / 1024),
            "gpu_reserved_mb": float(torch.cuda.memory_reserved(device) / 1024 / 1024),
        }
    return {
        "gpu_allocated_mb": None,
        "gpu_reserved_mb": None,
    }


def _sync_if_cuda(device):
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.synchronize(device)


def _append_csv_row(path, row):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _write_sanity_row(path, row):
    """Append single row to sanity calibration CSV file (with header if needed)."""
    _append_csv_row(path, row)


def _validate_neg_exclude_scope(args, source_graphs, cache_threshold=50_000_000):
    if args.neg_exclude_scope != 'ego':
        return
    print(
        "[NEG-EGO] ego exclusion enabled with CSR ego lists; dense N×N exclude masks are disabled.",
        flush=True,
    )


def _csr_mask_to_row_lists(mask):
    if isinstance(mask, dict):
        indices = mask["ego_indices"]
        indptr = mask["ego_indptr"]
        rows = []
        for r in range(indptr.numel() - 1):
            s = int(indptr[r].item())
            e = int(indptr[r + 1].item())
            rows.append(indices[s:e].clone())
        return rows
    if torch.is_tensor(mask):
        if mask.dim() != 2:
            raise ValueError(f"Expected a 2D mask, got shape {tuple(mask.shape)}")
        rows = []
        for r in range(mask.size(0)):
            rows.append(torch.nonzero(mask[r], as_tuple=False).squeeze(-1).to(torch.long))
        return rows
    raise TypeError(f"Unsupported CSR mask type: {type(mask)}")


def _build_ego_exclusion_rows_from_cache(cache_obj):
    cache_type = cache_obj.get('type')
    if cache_type == 'full':
        return _csr_mask_to_row_lists(cache_obj['mask'])
    if cache_type == 'block':
        rows = []
        for block_path in cache_obj['block_paths']:
            mask_block = _usp_safe_torch_load(block_path, map_location='cpu')
            rows.extend(_csr_mask_to_row_lists(mask_block))
        return rows
    raise ValueError(f"Unsupported ego cache type: {cache_type}")


def _offset_ego_exclusion_rows(rows, offset, device):
    return [row.to(device=device, dtype=torch.long) + offset for row in rows]


def build_or_load_ego_cache(
    name,
    adj,
    cache_dir,
    result_dir,
    k,
    max_neighbors,
    cache_threshold=50_000_000,
    block_size=512,
):
    profile_path = os.path.join(result_dir, "usp_ego_cache_profile.csv")
    t0 = time.time()

    num_nodes = adj.size(0)
    if adj.is_sparse:
        num_edges_or_nnz = int(adj._nnz())
    else:
        num_edges_or_nnz = int(adj.bool().sum().item())

    base_row = {
        "dataset": name,
        "num_nodes": num_nodes,
        "num_edges_or_nnz": num_edges_or_nnz,
        "subgraph_hop": k,
        "max_neighbors": max_neighbors,
        "cache_threshold": cache_threshold,
        "cache_type": None,
        "cache_hit": None,
        "num_blocks": None,
        "avg_ego_size": None,
        "max_ego_size": None,
        "time_sec": None,
        "path": None,
    }

    adj_cpu = adj.to_dense().cpu() if adj.is_sparse else adj.detach().cpu()

    if num_nodes * num_nodes <= cache_threshold:
        mask_path = _ego_full_cache_path(cache_dir, name, k, max_neighbors, num_nodes)
        if os.path.exists(mask_path):
            mask = _usp_safe_torch_load(mask_path, map_location="cpu")
            cache_hit = True
            status = "load_full"
        else:
            dense_mask = ego_subgraph_mask(adj_cpu, k=k, max_neighbors=max_neighbors).cpu()
            mask = _mask_to_csr(dense_mask)
            torch.save(mask, mask_path)
            cache_hit = False
            status = "build_full"

        # compute ego sizes whether mask is CSR dict or dense
        if isinstance(mask, dict):
            indptr = mask["ego_indptr"]
            ego_sizes = (indptr[1:] - indptr[:-1]).float()
        else:
            ego_sizes = mask.sum(dim=1).float()
        row = dict(base_row)
        row.update(
            {
                "cache_type": "full",
                "cache_hit": cache_hit,
                "num_blocks": 1,
                "avg_ego_size": float(ego_sizes.mean().item()),
                "max_ego_size": int(ego_sizes.max().item()),
                "time_sec": time.time() - t0,
                "path": mask_path,
            }
        )
        _append_csv_row(profile_path, row)
        print(
            f"[EGO-CACHE] {name}: {status}, N={num_nodes}, nnz={num_edges_or_nnz}, "
            f"avg_ego={row['avg_ego_size']:.2f}, max_ego={row['max_ego_size']}, "
            f"time={row['time_sec']:.2f}s",
            flush=True,
        )
        return {"type": "full", "mask": mask, "path": mask_path, "cache_format_version": 2}

    block_dir = _ego_block_cache_dir(cache_dir, name, k, max_neighbors, num_nodes, block_size)
    os.makedirs(block_dir, exist_ok=True)
    meta_path = os.path.join(block_dir, "meta.json")
    num_blocks = (num_nodes + block_size - 1) // block_size

    block_paths = []
    ego_sum = 0.0
    ego_max = 0
    cache_hits = 0

    for start in range(0, num_nodes, block_size):
        end = min(start + block_size, num_nodes)
        block_path = os.path.join(block_dir, f"block_{start:08d}_{end:08d}.pt")
        block_paths.append(block_path)
        if os.path.exists(block_path):
            mask_block = _usp_safe_torch_load(block_path, map_location="cpu")
            cache_hits += 1
        else:
            node_idx = torch.arange(start, end, device=adj_cpu.device)
            dense_block = ego_subgraph_mask_block(
                adj_cpu,
                node_idx,
                k=k,
                max_neighbors=max_neighbors,
            ).cpu()
            mask_block = _mask_to_csr(dense_block)
            torch.save(mask_block, block_path)

        # compute ego sizes for this block whether CSR or dense
        if isinstance(mask_block, dict):
            indptr = mask_block["ego_indptr"]
            ego_sizes = (indptr[1:] - indptr[:-1]).float()
        else:
            ego_sizes = mask_block.sum(dim=1).float()
        ego_sum += float(ego_sizes.sum().item())
        ego_max = max(ego_max, int(ego_sizes.max().item()))

    avg_ego = ego_sum / max(1, num_nodes)
    meta = {
        "dataset": name,
        "num_nodes": num_nodes,
        "num_edges_or_nnz": num_edges_or_nnz,
        "k": k,
        "max_neighbors": max_neighbors,
        "block_size": block_size,
        "num_blocks": num_blocks,
        "block_paths": block_paths,
        "avg_ego_size": avg_ego,
        "max_ego_size": ego_max,
        "cache_format_version": 2,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    row = dict(base_row)
    row.update(
        {
            "cache_type": "block",
            "cache_hit": cache_hits == num_blocks,
            "num_blocks": num_blocks,
            "avg_ego_size": avg_ego,
            "max_ego_size": ego_max,
            "time_sec": time.time() - t0,
            "path": block_dir,
        }
    )
    _append_csv_row(profile_path, row)
    print(
        f"[EGO-CACHE] {name}: block_cache, N={num_nodes}, nnz={num_edges_or_nnz}, "
        f"blocks={num_blocks}, hit={cache_hits}/{num_blocks}, avg_ego={avg_ego:.2f}, "
        f"max_ego={ego_max}, time={row['time_sec']:.2f}s",
        flush=True,
    )
    return {
        "type": "block",
        "block_dir": block_dir,
        "block_paths": block_paths,
        "block_size": block_size,
        "num_nodes": num_nodes,
        "meta_path": meta_path,
    }
