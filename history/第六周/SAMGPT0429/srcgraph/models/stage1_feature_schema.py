from __future__ import annotations

from collections.abc import Iterable

RAW9_FEATURE_KEYS = [
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

DEFAULT_RAW9_FEATURE_MASK = "123456789"
_ALLOWED_MASK_SEPARATORS = {",", " ", "_", "-", "[", "]", "(", ")"}

RAW9_DIGIT_TO_FEATURE = {
    str(i + 1): name for i, name in enumerate(RAW9_FEATURE_KEYS)
}
RAW9_FEATURE_TO_DIGIT = {
    name: digit for digit, name in RAW9_DIGIT_TO_FEATURE.items()
}

# 统一策略：raw9 全量特征均进行 log1p 预处理。
RAW9_LOG1P_FEATURE_KEYS = list(RAW9_FEATURE_KEYS)


def raw9_feature_indices(feature_names: list[str]) -> list[int]:
    """Map raw9 feature names to indices defined by RAW9_FEATURE_KEYS."""
    name_to_idx = {name: i for i, name in enumerate(RAW9_FEATURE_KEYS)}
    idx: list[int] = []
    for name in feature_names:
        if name not in name_to_idx:
            raise ValueError(
                f"Unknown raw9 feature name '{name}'. "
                f"Available={RAW9_FEATURE_KEYS}"
            )
        idx.append(int(name_to_idx[name]))
    return idx


def normalize_raw9_feature_mask(feature_mask: str | Iterable[object], *, allow_empty: bool = False) -> str:
    """Normalize mask like '1,3,5'/'135' to ordered unique digits preserving first appearance."""
    if isinstance(feature_mask, Iterable) and not isinstance(feature_mask, (str, bytes)):
        digits: list[str] = []
        for item in feature_mask:
            token = str(item).strip()
            if token in RAW9_DIGIT_TO_FEATURE:
                if token not in digits:
                    digits.append(token)
            else:
                raise ValueError(
                    f"Invalid token '{item}' in stage1_feature_mask={feature_mask}. "
                    "Expected values in [1..9]."
                )

        normalized_iter = "".join(digits)
        if normalized_iter == "" and not allow_empty:
            raise ValueError("stage1_feature_mask must include at least one digit in [1..9]")
        return normalized_iter

    raw = str(feature_mask).strip()
    if raw == "":
        if allow_empty:
            return ""
        raise ValueError("stage1_feature_mask cannot be empty")

    digits: list[str] = []
    for ch in raw:
        if ch in RAW9_DIGIT_TO_FEATURE:
            if ch not in digits:
                digits.append(ch)
        elif ch in _ALLOWED_MASK_SEPARATORS:
            continue
        else:
            raise ValueError(
                f"Invalid character '{ch}' in stage1_feature_mask={feature_mask}. "
                "Only digits 1-9 and separators ',',' ','_','-' are allowed."
            )

    normalized = "".join(digits)
    if normalized == "" and not allow_empty:
        raise ValueError("stage1_feature_mask must include at least one digit in [1..9]")
    return normalized


def parse_raw9_feature_mask(feature_mask: str) -> tuple[list[int], list[str], str]:
    """Return indices, feature names and normalized digits for raw9 mask."""
    normalized = normalize_raw9_feature_mask(feature_mask, allow_empty=False)
    idx = [int(d) - 1 for d in normalized]
    names = [RAW9_DIGIT_TO_FEATURE[d] for d in normalized]
    return idx, names, normalized
