import os

import numpy as np
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


def compute_cluster_quality_metrics(x, labels, max_points, seed):
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n = int(x.shape[0])
    if n < 3:
        return {'valid': False, 'reason': 'too_few_samples', 'n_samples': n}

    uniq, _ = np.unique(labels, return_counts=True)
    k = int(len(uniq))
    if k < 2:
        return {'valid': False, 'reason': 'single_cluster', 'n_samples': n, 'n_clusters': k}

    max_points = max(512, int(max_points))
    if n > max_points:
        rng = np.random.default_rng(int(seed))
        keep = np.sort(rng.choice(np.arange(n), size=max_points, replace=False))
        x_eval = x[keep]
        labels_eval = labels[keep]
    else:
        x_eval = x
        labels_eval = labels

    uniq_eval, counts_eval = np.unique(labels_eval, return_counts=True)
    if int(len(uniq_eval)) < 2 or int(np.min(counts_eval)) < 2:
        return {
            'valid': False,
            'reason': 'insufficient_cluster_support_after_sampling',
            'n_samples': int(x_eval.shape[0]),
            'n_clusters': int(len(uniq_eval)),
        }

    x_center = x_eval - x_eval.mean(axis=0, keepdims=True)
    cov = np.matmul(x_center.T, x_center) / max(1, int(x_eval.shape[0] - 1))
    evals, _ = np.linalg.eigh(cov)
    evals = np.sort(np.maximum(evals, 0.0))[::-1]
    var_sum = float(np.sum(evals)) if evals.size > 0 else 0.0
    pc1 = float(evals[0] / var_sum) if var_sum > 1e-12 and evals.size > 0 else 0.0
    pc2 = float(evals[1] / var_sum) if var_sum > 1e-12 and evals.size > 1 else 0.0
    anisotropy = float(evals[0] / max(evals[1], 1e-12)) if evals.size > 1 else float('inf')

    sil = float(silhouette_score(x_eval, labels_eval, metric='euclidean'))
    dbi = float(davies_bouldin_score(x_eval, labels_eval))
    chi = float(calinski_harabasz_score(x_eval, labels_eval))

    return {
        'valid': True,
        'n_samples': int(x_eval.shape[0]),
        'n_clusters': int(len(uniq_eval)),
        'silhouette': sil,
        'davies_bouldin': dbi,
        'calinski_harabasz': chi,
        'pc1_explained_ratio': pc1,
        'pc2_explained_ratio': pc2,
        'pc1_over_pc2_ratio': anisotropy,
    }


def save_encoder_filter_visualization(output_dir, x, labels, stage1_keep, stage2_keep, enabled=True):
    if not bool(enabled):
        return {'saved': False, 'reason': 'disabled'}

    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if int(x.shape[0]) < 2:
        return {'saved': False, 'reason': 'too_few_samples'}

    try:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        stage1_arr = np.asarray(stage1_keep, dtype=np.int64)
        stage2_arr = np.asarray(stage2_keep, dtype=np.int64)
        selected_idx = np.sort(np.unique(np.concatenate([stage1_arr, stage2_arr], axis=0)))
        if selected_idx.size == 0:
            selected_idx = np.arange(x.shape[0], dtype=np.int64)

        x_center = x - x.mean(axis=0, keepdims=True)
        cov = np.matmul(x_center.T, x_center)
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1][:2]
        comp = evecs[:, order]
        x2_all = np.matmul(x_center, comp)
        x2_sel = x2_all[selected_idx]
        labels_sel = labels[selected_idx]

        unique_labels = np.unique(labels)
        centers_hd = []
        center_ids = []
        for cid in unique_labels:
            idx = np.where(labels == cid)[0]
            if idx.size == 0:
                continue
            centers_hd.append(x[idx].mean(axis=0))
            center_ids.append(int(cid))
        centers_hd = np.asarray(centers_hd, dtype=np.float64)
        if centers_hd.size > 0:
            centers_2d = np.matmul(centers_hd - x.mean(axis=0, keepdims=True), comp)
        else:
            centers_2d = np.empty((0, 2), dtype=np.float64)

        plt.figure(figsize=(8, 6), dpi=150)
        plt.scatter(x2_all[:, 0], x2_all[:, 1], c=labels, s=6, alpha=0.55, cmap='tab20')
        if centers_2d.shape[0] > 0:
            plt.scatter(
                centers_2d[:, 0],
                centers_2d[:, 1],
                c='red',
                s=120,
                marker='X',
                edgecolors='black',
                linewidths=0.8,
                zorder=5,
                label='Cluster Centers',
            )
            for i, cid in enumerate(center_ids):
                plt.annotate(
                    f'C{cid}',
                    (centers_2d[i, 0], centers_2d[i, 1]),
                    textcoords='offset points',
                    xytext=(4, 4),
                    fontsize=7,
                    color='black',
                    zorder=6,
                )
            plt.legend(loc='best', fontsize=8, frameon=True)
        plt.title('Encoder Clusters Before Filtering (2D SVD projection)')
        plt.xlabel('comp-1')
        plt.ylabel('comp-2')
        plt.tight_layout()
        out_before_png = os.path.join(output_dir, 'encoder_filter_clusters_before.png')
        plt.savefig(out_before_png)
        plt.close()

        plt.figure(figsize=(8, 6), dpi=150)
        plt.scatter(x2_sel[:, 0], x2_sel[:, 1], c=labels_sel, s=7, alpha=0.65, cmap='tab20')
        if centers_2d.shape[0] > 0:
            plt.scatter(
                centers_2d[:, 0],
                centers_2d[:, 1],
                c='red',
                s=120,
                marker='X',
                edgecolors='black',
                linewidths=0.8,
                zorder=5,
                label='Cluster Centers',
            )
            for i, cid in enumerate(center_ids):
                plt.annotate(
                    f'C{cid}',
                    (centers_2d[i, 0], centers_2d[i, 1]),
                    textcoords='offset points',
                    xytext=(4, 4),
                    fontsize=7,
                    color='black',
                    zorder=6,
                )
            plt.legend(loc='best', fontsize=8, frameon=True)
        plt.title('Encoder Clusters After Filtering (2D SVD projection)')
        plt.xlabel('comp-1')
        plt.ylabel('comp-2')
        plt.tight_layout()
        out_png = os.path.join(output_dir, 'encoder_filter_clusters.png')
        plt.savefig(out_png)
        plt.close()

        return {
            'saved': True,
            'file': out_png,
            'before_file': out_before_png,
            'total_points': int(x2_all.shape[0]),
            'selected_points': int(x2_sel.shape[0]),
        }
    except Exception as e:
        return {'saved': False, 'reason': str(e)}


def assess_cluster_quality(quality_all, quality_selected):
    def _merge_status(cur, nxt):
        order = {'pass': 0, 'warn': 1, 'fail': 2}
        return nxt if order[nxt] > order[cur] else cur

    result = {
        'status': 'pass',
        'summary': 'cluster quality looks healthy',
        'checks': [],
    }

    if not bool(quality_all.get('valid', False)):
        return {
            'status': 'fail',
            'summary': f"all metrics invalid: {quality_all.get('reason', 'unknown')}",
            'checks': [{'name': 'all_valid', 'status': 'fail'}],
        }
    if not bool(quality_selected.get('valid', False)):
        return {
            'status': 'fail',
            'summary': f"selected metrics invalid: {quality_selected.get('reason', 'unknown')}",
            'checks': [{'name': 'selected_valid', 'status': 'fail'}],
        }

    def _check_threshold(name, value, pass_cond, warn_cond, pass_msg, warn_msg, fail_msg):
        if pass_cond(value):
            status = 'pass'
            msg = pass_msg
        elif warn_cond(value):
            status = 'warn'
            msg = warn_msg
        else:
            status = 'fail'
            msg = fail_msg
        result['status'] = _merge_status(result['status'], status)
        result['checks'].append({'name': name, 'status': status, 'value': float(value), 'message': msg})

    sil_all = float(quality_all['silhouette'])
    sil_sel = float(quality_selected['silhouette'])
    dbi_all = float(quality_all['davies_bouldin'])
    dbi_sel = float(quality_selected['davies_bouldin'])
    aniso_all = float(quality_all['pc1_over_pc2_ratio'])
    aniso_sel = float(quality_selected['pc1_over_pc2_ratio'])

    _check_threshold(
        'silhouette_all',
        sil_all,
        pass_cond=lambda v: v >= 0.15,
        warn_cond=lambda v: v >= 0.05,
        pass_msg='good global cluster separation',
        warn_msg='global separation is weak',
        fail_msg='global cluster separation is poor',
    )
    _check_threshold(
        'silhouette_selected',
        sil_sel,
        pass_cond=lambda v: v >= 0.12,
        warn_cond=lambda v: v >= 0.03,
        pass_msg='selected set remains separable',
        warn_msg='selected set separation is weak',
        fail_msg='selected set is heavily mixed',
    )
    _check_threshold(
        'dbi_all',
        dbi_all,
        pass_cond=lambda v: v <= 2.5,
        warn_cond=lambda v: v <= 4.0,
        pass_msg='global compactness and separation are acceptable',
        warn_msg='global DBI is high',
        fail_msg='global DBI indicates poor clustering',
    )
    _check_threshold(
        'dbi_selected',
        dbi_sel,
        pass_cond=lambda v: v <= 2.8,
        warn_cond=lambda v: v <= 4.5,
        pass_msg='selected compactness is acceptable',
        warn_msg='selected DBI is high',
        fail_msg='selected DBI indicates severe overlap',
    )

    sil_drop = float(sil_sel - sil_all)
    dbi_jump = float(dbi_sel - dbi_all)
    _check_threshold(
        'selection_silhouette_delta',
        sil_drop,
        pass_cond=lambda v: v >= -0.05,
        warn_cond=lambda v: v >= -0.12,
        pass_msg='selection keeps separation quality',
        warn_msg='selection causes moderate separation drop',
        fail_msg='selection causes large separation drop',
    )
    _check_threshold(
        'selection_dbi_delta',
        dbi_jump,
        pass_cond=lambda v: v <= 0.6,
        warn_cond=lambda v: v <= 1.5,
        pass_msg='selection keeps compactness quality',
        warn_msg='selection increases overlap moderately',
        fail_msg='selection increases overlap severely',
    )

    aniso_msg = 'projection anisotropy is normal'
    aniso_status = 'pass'
    if aniso_all > 25.0 or aniso_sel > 25.0:
        aniso_status = 'warn'
        aniso_msg = '2D projection is highly elongated; verify in high-dimensional metrics'
    result['status'] = _merge_status(result['status'], aniso_status)
    result['checks'].append(
        {
            'name': 'projection_anisotropy',
            'status': aniso_status,
            'value_all': aniso_all,
            'value_selected': aniso_sel,
            'message': aniso_msg,
        }
    )

    if result['status'] == 'pass':
        result['summary'] = 'cluster quality looks healthy'
    elif result['status'] == 'warn':
        result['summary'] = 'cluster quality is usable but has risks'
    else:
        result['summary'] = 'cluster quality is problematic'

    return result
