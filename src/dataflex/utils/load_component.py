import yaml
from typing import Dict, Any, Optional

from dataflex.utils.logging import logger

#: Bucket names that were renamed, mapped to the older spellings still accepted.
#: The reorder family used to be called "reorderer", so a config written before
#: the rename says `reorderers:`. Reading it keeps working; only a warning marks
#: it as deprecated.
_BUCKET_ALIASES: Dict[str, tuple] = {
    "reorders": ("reorderers",),
}

_warned_aliases = set()


def _resolve_bucket(root: Dict[str, Any], type: str):
    """Return (bucket, actual_key), falling back to a deprecated spelling."""
    if root.get(type):
        return root[type], type

    for legacy in _BUCKET_ALIASES.get(type, ()):
        if root.get(legacy):
            if legacy not in _warned_aliases:
                _warned_aliases.add(legacy)
                logger.warning(
                    f"[Dataflex] config section '{legacy}:' is deprecated, rename it to '{type}:'. "
                    f"It still works for now."
                )
            return root[legacy], legacy

    return {}, type


def load_component(type: str, cfg_file: str, name: str, runtime_vars: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    with open(cfg_file, "r", encoding="utf-8") as f:
        root = yaml.safe_load(f) or {}
    bucket, key = _resolve_bucket(root, type)
    if name not in bucket:
        available = ", ".join(sorted(bucket.keys()))
        raise ValueError(f"{key} '{name}' not found. Available: {available}")
    params = dict(bucket[name].get("params") or {})

    # Simple placeholder substitution (e.g. ${output_dir})
    if runtime_vars:
        def subst(v):
            if isinstance(v, str):
                for k, val in runtime_vars.items(): v = v.replace(k, val)
                return v
            if isinstance(v, dict):  return {kk: subst(vv) for kk, vv in v.items()}
            if isinstance(v, list):  return [subst(x) for x in v]
            return v
        params = subst(params)

    return params
