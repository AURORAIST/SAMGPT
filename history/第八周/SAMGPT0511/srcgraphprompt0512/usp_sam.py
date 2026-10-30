import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_node_matrix(x):
    if x.dim() == 3:
        if x.size(0) != 1:
            raise ValueError("USP-SAM expects a single graph tensor with shape [1, N, F] or [N, F].")
        return x.squeeze(0)
    if x.dim() == 2:
        return x
    raise ValueError(f"Expected node embeddings with 2 or 3 dims, got shape {tuple(x.shape)}.")


def cosine_logits(left, right, temperature=0.2):
    left = F.normalize(left, p=2, dim=-1)
    right = F.normalize(right, p=2, dim=-1)
    return torch.matmul(left, right.t()) / temperature


def info_nce(anchor, target, temperature=0.2):
    logits = cosine_logits(anchor, target, temperature)
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels), logits


def sampled_info_nce(anchor, positive, negative, temperature=0.2):
    """InfoNCE with explicit negative samples.

    Shapes:
        anchor: [N, F]
        positive: [N, F]
        negative: [N, K, F]
    """
    anchor = F.normalize(anchor, p=2, dim=-1)
    positive = F.normalize(positive, p=2, dim=-1)
    negative = F.normalize(negative, p=2, dim=-1)
    pos_logits = torch.sum(anchor * positive, dim=-1, keepdim=True)
    neg_logits = torch.sum(anchor.unsqueeze(1) * negative, dim=-1)
    logits = torch.cat([pos_logits, neg_logits], dim=1) / temperature
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels), logits


def ego_subgraph_mask(adj, k=2, include_self=True):
    """Return a boolean [N, N] mask whose row v marks nodes in the k-hop ego graph of v.

    The implementation accepts torch dense or torch sparse adjacency. It deliberately
    returns a mask rather than materialized node lists so readout can be implemented as
    masked matrix operations.
    """
    if k < 0:
        raise ValueError("k must be non-negative.")

    if adj.is_sparse:
        dense_adj = adj.to_dense()
    else:
        dense_adj = adj

    reach = dense_adj.bool()
    n = reach.size(0)
    eye = torch.eye(n, dtype=torch.bool, device=reach.device)
    if include_self:
        reach = reach | eye

    frontier = reach.clone()
    total = reach.clone()
    dense_float = reach.float()
    for _ in range(1, k):
        frontier = torch.matmul(frontier.float(), dense_float).bool()
        total = total | frontier
    return total | eye if include_self else total


def structural_role_features(adj, ego_mask=None):
    """Temporary structural feature placeholder, not original SAMGPT structure token."""
    if adj.is_sparse:
        dense_adj = adj.to_dense()
    else:
        dense_adj = adj
    binary_adj = dense_adj.bool().float()
    n = binary_adj.size(0)
    eye = torch.eye(n, device=binary_adj.device, dtype=binary_adj.dtype)
    no_self = binary_adj * (1.0 - eye)
    degree = no_self.sum(dim=1)
    triangles = (torch.matmul(no_self, no_self) * no_self).sum(dim=1) / 2.0
    denom = degree * (degree - 1.0) / 2.0
    clustering = torch.where(denom > 0, triangles / denom.clamp_min(1.0), torch.zeros_like(degree))
    if ego_mask is None:
        ego_size = binary_adj.bool().sum(dim=1).float()
    else:
        ego_size = ego_mask.float().sum(dim=1)
    feats = torch.stack([degree, clustering, ego_size], dim=1)
    mean = feats.mean(dim=0, keepdim=True)
    std = feats.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (feats - mean) / std


def structural_role_buckets(adj, ego_mask=None, bucket="mixed", num_bins=4):
    feats = structural_role_features(adj, ego_mask)
    names = {"degree": 0, "clustering": 1, "ego_size": 2}
    if bucket in names:
        values = feats[:, names[bucket]]
    elif bucket == "mixed":
        values = feats[:, 0] + feats[:, 1] + feats[:, 2]
    else:
        raise ValueError(f"Unknown structure bucket: {bucket}")
    quantiles = torch.quantile(values.detach(), torch.linspace(0, 1, num_bins + 1, device=values.device)[1:-1])
    return torch.bucketize(values, quantiles)


def _sample_with_replacement(candidates, count, device):
    if candidates.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    draw = torch.randint(0, candidates.numel(), (count,), device=device)
    return candidates[draw]


def random_negative_indices(num_nodes, num_negatives, device):
    rows = []
    all_idx = torch.arange(num_nodes, device=device)
    for i in range(num_nodes):
        candidates = all_idx[all_idx != i]
        rows.append(_sample_with_replacement(candidates, num_negatives, device))
    return torch.stack(rows, dim=0), {
        "num_intra_neg": None,
        "num_inter_neg": None,
        "num_negatives_effective": num_negatives,
        "intra_neg_ratio_effective": None,
        "negative_fallback_reason": None,
    }


def domain_balanced_negative_indices(domain_ids, num_negatives=128, intra_neg_ratio=0.5, exclude_mask=None):
    device = domain_ids.device
    num_nodes = domain_ids.numel()
    all_idx = torch.arange(num_nodes, device=device)
    domains = torch.unique(domain_ids)
    num_intra = int(round(num_negatives * intra_neg_ratio))
    num_inter = max(0, num_negatives - num_intra)
    rows = []
    fallback_reasons = set()

    for i in range(num_nodes):
        same_domain = domain_ids == domain_ids[i]
        if exclude_mask is not None:
            excluded = exclude_mask[i].bool()
        else:
            excluded = all_idx == i
        intra_candidates = all_idx[same_domain & ~excluded]
        if intra_candidates.numel() == 0:
            intra_candidates = all_idx[same_domain & (all_idx != i)]
            fallback_reasons.add("insufficient_intra_domain_nodes")
        elif intra_candidates.numel() < num_intra:
            fallback_reasons.add("insufficient_intra_domain_nodes")
        intra = _sample_with_replacement(intra_candidates, num_intra, device)

        other_domains = domains[domains != domain_ids[i]]
        inter_parts = []
        if other_domains.numel() > 0 and num_inter > 0:
            base = num_inter // other_domains.numel()
            rem = num_inter % other_domains.numel()
            for pos, domain in enumerate(other_domains):
                take = base + (1 if pos < rem else 0)
                candidates = all_idx[domain_ids == domain]
                if candidates.numel() < take:
                    fallback_reasons.add("insufficient_inter_domain_nodes")
                inter_parts.append(_sample_with_replacement(candidates, take, device))
        elif num_inter > 0:
            fallback_reasons.add("insufficient_inter_domain_nodes")
        inter = torch.cat(inter_parts, dim=0) if inter_parts else torch.empty(0, dtype=torch.long, device=device)

        neg = torch.cat([intra, inter], dim=0)
        if neg.numel() < num_negatives:
            fallback = all_idx[all_idx != i]
            neg = torch.cat([neg, _sample_with_replacement(fallback, num_negatives - neg.numel(), device)], dim=0)
            fallback_reasons.add("fallback_to_available_nodes")
        rows.append(neg[:num_negatives])

    if num_nodes <= num_negatives:
        fallback_reasons.add("batch_too_small")
    return torch.stack(rows, dim=0), {
        "num_intra_neg": num_intra,
        "num_inter_neg": num_inter,
        "num_negatives_effective": num_negatives,
        "intra_neg_ratio_effective": num_intra / max(1, num_negatives),
        "negative_fallback_reason": "|".join(sorted(fallback_reasons)) if fallback_reasons else None,
    }


def structural_alignment_loss(subgraph_embeds, domain_ids, role_buckets, temperature=0.2):
    logits = cosine_logits(subgraph_embeds, subgraph_embeds, temperature)
    num_nodes = logits.size(0)
    eye = torch.eye(num_nodes, dtype=torch.bool, device=logits.device)
    positive = (role_buckets.unsqueeze(0) == role_buckets.unsqueeze(1)) & (
        domain_ids.unsqueeze(0) != domain_ids.unsqueeze(1)
    )
    positive = positive & ~eye
    denominator_mask = ~eye
    log_prob = logits - logits.masked_fill(~denominator_mask, torch.finfo(logits.dtype).min).logsumexp(dim=1, keepdim=True)
    valid = positive.sum(dim=1) > 0
    if not valid.any():
        return subgraph_embeds.new_tensor(0.0)
    return -(log_prob * positive.float()).sum(dim=1)[valid].div(positive.sum(dim=1)[valid].float()).mean()


class MeanSubgraphReadout(nn.Module):
    def forward(self, node_embeds, subgraph_mask):
        h = _as_node_matrix(node_embeds)
        weights = subgraph_mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return torch.matmul(weights, h)


class AttentionSubgraphReadout(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, node_embeds, subgraph_mask):
        h = _as_node_matrix(node_embeds)
        scores = self.score(torch.tanh(self.proj(h))).squeeze(-1)
        scores = scores.unsqueeze(0).expand(subgraph_mask.size(0), -1)
        scores = scores.masked_fill(~subgraph_mask.bool(), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        return torch.matmul(weights, h)


class PromptWeightedSubgraphReadout(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.prompt = nn.Parameter(torch.empty(hidden_dim))
        self.weight = nn.Parameter(torch.empty(hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.prompt, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.weight.unsqueeze(0))

    def forward(self, node_embeds, subgraph_mask):
        h = _as_node_matrix(node_embeds)
        scores = (h * self.prompt * self.weight).sum(dim=-1)
        scores = scores.unsqueeze(0).expand(subgraph_mask.size(0), -1)
        scores = scores.masked_fill(~subgraph_mask.bool(), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        return torch.matmul(weights, h)

    def regularization(self):
        return self.prompt.pow(2).mean() + self.weight.pow(2).mean()


class StructuralRoleReadout(nn.Module):
    """Temporary structural feature placeholder, not original SAMGPT structure token."""

    def __init__(self, hidden_dim, struct_dim=3):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim + struct_dim + hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, node_embeds, subgraph_mask, structure_features):
        h = _as_node_matrix(node_embeds)
        if structure_features is None:
            raise ValueError("structure_features is required for StructuralRoleReadout.")
        role = structure_features.to(device=h.device, dtype=h.dtype)
        token = role
        if token.size(1) != h.size(1):
            repeat = (h.size(1) + token.size(1) - 1) // token.size(1)
            token = token.repeat(1, repeat)[:, : h.size(1)]
        score_input = torch.cat([h, role, h * token], dim=-1)
        scores = self.score(score_input).squeeze(-1)
        scores = scores.unsqueeze(0).expand(subgraph_mask.size(0), -1)
        scores = scores.masked_fill(~subgraph_mask.bool(), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        return torch.matmul(weights, h)


def build_subgraph_readout(kind, hidden_dim, align_type="none"):
    if align_type == "placeholder_structural_role":
        return StructuralRoleReadout(hidden_dim)
    kind = kind.lower()
    if kind == "mean":
        return MeanSubgraphReadout()
    if kind == "attention":
        return AttentionSubgraphReadout(hidden_dim)
    if kind in {"prompt", "prompt_weighted", "prompt-weighted"}:
        return PromptWeightedSubgraphReadout(hidden_dim)
    raise ValueError(f"Unknown readout kind: {kind}")


class USPPretrainingHead(nn.Module):
    """Unified node-subgraph and subgraph-subgraph contrastive pretraining head."""

    def __init__(
        self,
        hidden_dim,
        readout="prompt_weighted",
        k=2,
        temperature=0.2,
        lambda_ss=1.0,
        lambda_align=0.0,
        lambda_reg=0.0,
        align_type="none",
        structure_bucket="mixed",
    ):
        super().__init__()
        self.k = k
        self.temperature = temperature
        self.lambda_ss = lambda_ss
        self.lambda_align = lambda_align
        self.lambda_reg = lambda_reg
        self.align_type = align_type
        self.structure_bucket = structure_bucket
        self.readout = build_subgraph_readout(readout, hidden_dim, align_type=align_type)

    def encode(self, gcn, features, adj, sparse=True, prompt_layers=None):
        return _as_node_matrix(gcn(features, adj, sparse, False, prompt_layers))

    def subgraph_embeddings(self, node_embeds, adj, k=None):
        mask = ego_subgraph_mask(adj, self.k if k is None else k)
        if self.align_type == "placeholder_structural_role":
            structure_features = structural_role_features(adj, mask)
            return self.readout(node_embeds, mask, structure_features), mask
        return self.readout(node_embeds, mask), mask

    def forward(
        self,
        gcn,
        features,
        adj,
        sparse=True,
        aug_features_1=None,
        aug_adj_1=None,
        aug_features_2=None,
        aug_adj_2=None,
        prompt_layers=None,
        align_loss=None,
    ):
        h = self.encode(gcn, features, adj, sparse, prompt_layers)
        g, mask = self.subgraph_embeddings(h, adj)
        loss_ns, logits_ns = info_nce(h, g, self.temperature)

        if aug_features_1 is not None and aug_adj_1 is not None:
            h1 = self.encode(gcn, aug_features_1, aug_adj_1, sparse, prompt_layers)
            g1, _ = self.subgraph_embeddings(h1, aug_adj_1)
        else:
            g1 = g

        if aug_features_2 is not None and aug_adj_2 is not None:
            h2 = self.encode(gcn, aug_features_2, aug_adj_2, sparse, prompt_layers)
            g2, _ = self.subgraph_embeddings(h2, aug_adj_2)
        else:
            g2 = g

        loss_ss, logits_ss = info_nce(g1, g2, self.temperature)
        if align_loss is None:
            align_loss = h.new_tensor(0.0)

        reg_loss = h.new_tensor(0.0)
        if hasattr(self.readout, "regularization"):
            reg_loss = self.readout.regularization()

        total = loss_ns + self.lambda_ss * loss_ss + self.lambda_align * align_loss + self.lambda_reg * reg_loss
        return {
            "loss": total,
            "loss_ns": loss_ns,
            "loss_ss": loss_ss,
            "loss_align": align_loss,
            "loss_reg": reg_loss,
            "node_embeds": h,
            "subgraph_embeds": g,
            "subgraph_mask": mask,
            "logits_ns": logits_ns.detach(),
            "logits_ss": logits_ss.detach(),
        }


class ClassPrototypeSubgraphClassifier(nn.Module):
    """Few-shot classifier using class prototype subgraphs and query-prototype similarity."""

    def __init__(
        self,
        hidden_dim,
        nb_classes,
        prototype="mean",
        query="hybrid",
        temperature=0.2,
        learnable_eta=True,
        eta=0.5,
    ):
        super().__init__()
        self.nb_classes = nb_classes
        self.prototype = prototype
        self.query = query
        self.temperature = temperature
        if learnable_eta:
            eta = torch.logit(torch.tensor(float(eta)).clamp(1e-4, 1 - 1e-4))
            self.eta_logit = nn.Parameter(eta)
        else:
            self.register_buffer("eta_logit", torch.logit(torch.tensor(float(eta)).clamp(1e-4, 1 - 1e-4)))
        self.prototype_att = nn.Linear(hidden_dim, 1, bias=False)
        self.register_buffer("class_prototypes", torch.empty(nb_classes, hidden_dim))

    @property
    def eta(self):
        return torch.sigmoid(self.eta_logit)

    def make_query(self, node_embeds, subgraph_embeds):
        if self.query == "node":
            return node_embeds
        if self.query == "subgraph":
            return subgraph_embeds
        if self.query == "hybrid":
            return self.eta * node_embeds + (1.0 - self.eta) * subgraph_embeds
        raise ValueError(f"Unknown query mode: {self.query}")

    def build_prototypes(self, support_subgraph_embeds, support_labels):
        protos = []
        for cls in range(self.nb_classes):
            cls_embeds = support_subgraph_embeds[support_labels == cls]
            if cls_embeds.numel() == 0:
                raise ValueError(f"No support examples for class {cls}.")
            if self.prototype == "mean":
                protos.append(cls_embeds.mean(dim=0))
            elif self.prototype == "attention":
                weights = torch.softmax(self.prototype_att(cls_embeds).squeeze(-1), dim=0)
                protos.append(torch.sum(weights.unsqueeze(-1) * cls_embeds, dim=0))
            else:
                raise ValueError(f"Unknown prototype mode: {self.prototype}")
        self.class_prototypes = torch.stack(protos, dim=0)
        return self.class_prototypes

    def forward(
        self,
        node_embeds,
        subgraph_embeds,
        query_idx,
        support_idx=None,
        support_labels=None,
        update_prototypes=False,
    ):
        h = _as_node_matrix(node_embeds)
        g = _as_node_matrix(subgraph_embeds)
        if update_prototypes:
            if support_idx is None or support_labels is None:
                raise ValueError("support_idx and support_labels are required when update_prototypes=True.")
            self.build_prototypes(g[support_idx], support_labels)
        query = self.make_query(h[query_idx], g[query_idx])
        return cosine_logits(query, self.class_prototypes, self.temperature)


def similarity_margin(logits, labels):
    true_scores = logits.gather(1, labels.view(-1, 1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels.view(-1, 1), torch.finfo(logits.dtype).min)
    return true_scores - masked.max(dim=1).values


def prototype_compactness(embeds, labels, prototypes):
    values = []
    for cls in range(prototypes.size(0)):
        cls_embeds = embeds[labels == cls]
        if cls_embeds.numel() == 0:
            continue
        dist = 1.0 - F.cosine_similarity(cls_embeds, prototypes[cls].unsqueeze(0), dim=-1)
        values.append(dist.mean())
    if not values:
        return embeds.new_tensor(float("nan"))
    return torch.stack(values).mean()


def prototype_separation(prototypes):
    sim = F.cosine_similarity(prototypes.unsqueeze(1), prototypes.unsqueeze(0), dim=-1)
    eye = torch.eye(prototypes.size(0), dtype=torch.bool, device=prototypes.device)
    return (1.0 - sim.masked_select(~eye)).mean()


def _build_prototypes(embeds, labels, num_classes, prototype_type="mean"):
    protos = []
    for cls in range(num_classes):
        cls_embeds = embeds[labels == cls]
        if cls_embeds.numel() == 0:
            raise ValueError(f"No support examples for class {cls}.")
        if prototype_type == "mean":
            protos.append(cls_embeds.mean(dim=0))
        elif prototype_type == "attention":
            scores = torch.matmul(cls_embeds, cls_embeds.mean(dim=0, keepdim=True).t()).squeeze(-1)
            weights = torch.softmax(scores, dim=0)
            protos.append(torch.sum(weights.unsqueeze(-1) * cls_embeds, dim=0))
        else:
            raise ValueError(f"Unknown prototype_type: {prototype_type}")
    return torch.stack(protos, dim=0)


def evaluate_fewshot_prototype(
    embeddings,
    subgraph_embeddings,
    labels,
    support_idx,
    query_idx,
    query_mode="node",
    prototype_type="mean",
    metric="cosine",
):
    """Shared few-shot prototype evaluator for SAMGPT/USP sanity checks.

    SAMGPT can pass only node embeddings and use query_mode='node'. USP can pass
    both node and subgraph embeddings and use node/subgraph/hybrid query modes.
    """
    from sklearn.metrics import balanced_accuracy_score, f1_score

    h = _as_node_matrix(embeddings)
    g = _as_node_matrix(subgraph_embeddings) if subgraph_embeddings is not None else h
    labels = labels.long()
    support_labels = labels[support_idx].long()
    num_classes = int(labels.max().item() + 1)
    prototypes = _build_prototypes(g[support_idx], support_labels, num_classes, prototype_type)

    if query_mode == "node":
        query = h[query_idx]
    elif query_mode == "subgraph":
        query = g[query_idx]
    elif query_mode == "hybrid":
        query = 0.5 * h[query_idx] + 0.5 * g[query_idx]
    else:
        raise ValueError(f"Unknown query_mode: {query_mode}")

    if metric != "cosine":
        raise ValueError("Only cosine metric is implemented for the shared evaluator.")
    logits = cosine_logits(query, prototypes, temperature=1.0)
    preds = logits.argmax(dim=1)
    query_labels = labels[query_idx].long()
    acc = (preds == query_labels).float().mean().item() * 100.0
    pred_np = preds.detach().cpu().numpy()
    label_np = query_labels.detach().cpu().numpy()
    return {
        "logits": logits,
        "preds": preds,
        "prototypes": prototypes,
        "acc": acc,
        "balanced_acc": balanced_accuracy_score(label_np, pred_np) * 100.0,
        "macro_f1": f1_score(label_np, pred_np, average="macro") * 100.0,
        "micro_f1": f1_score(label_np, pred_np, average="micro") * 100.0,
        "margin": similarity_margin(logits, query_labels).mean().item(),
        "compactness": prototype_compactness(g[support_idx], support_labels, prototypes).item(),
        "separation": prototype_separation(prototypes).item(),
    }
