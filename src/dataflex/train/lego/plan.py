"""The artifact that index-space stages pass to each other."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class DataPlan:
    """What the model will train on for the upcoming interval.

    `indices` holds positions into the trainer's ``train_dataset``. Its *order*
    is meaningful: the lego trainer feeds it through a ``Subset`` +
    ``SequentialSampler``, so position here is position in the training stream.

    It is a numpy array rather than a Python list because the first plan of each
    boundary spans the whole dataset. At the corpus sizes the mixers already
    target, tens of millions of rows, a list of Python ints costs on the order of
    a gigabyte while the equivalent int64 array costs a few hundred megabytes and
    slices without copying object pointers. ``Subset`` accepts an array directly,
    so nothing downstream has to change.

    Stages hand a plan to the next stage, each narrowing or rearranging it:

        pool -> select (filter) -> mix (domain quota) -> reorder (arrange)

    `weights` is an optional per-index multiplier. Index-space stages normally
    leave it alone; it exists so a stage can express "keep this sample but
    de-emphasise it" and so provenance survives into the loss stack.
    """

    indices: np.ndarray
    weights: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.indices, np.ndarray):
            self.indices = np.asarray(list(self.indices), dtype=np.int64)
        elif self.indices.dtype != np.int64:
            self.indices = self.indices.astype(np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def as_list(self) -> List[int]:
        """Plain Python list, for the component signatures that require one."""
        return self.indices.tolist()

    @classmethod
    def over(cls, indices: Sequence[int], **meta: Any) -> "DataPlan":
        if isinstance(indices, range):
            arr = np.arange(indices.start, indices.stop, indices.step, dtype=np.int64)
        else:
            arr = np.asarray(indices, dtype=np.int64)
        return cls(indices=arr, meta=dict(meta))

    def replace_indices(self, indices: Sequence[int], stage: str = "", **meta: Any) -> "DataPlan":
        """Return a new plan over `indices`, carrying weights for survivors.

        Weights are keyed by dataset index rather than by position, because
        stages reorder and drop entries. Anything not previously weighted stays
        unweighted.
        """
        indices = np.asarray(indices, dtype=np.int64) if not isinstance(indices, np.ndarray) else indices.astype(np.int64)
        new_weights = None
        if self.weights is not None:
            by_index = dict(zip(self.indices.tolist(), self.weights))
            if any(int(i) in by_index for i in indices):
                new_weights = np.array([by_index.get(int(i), 1.0) for i in indices], dtype=np.float64)

        merged = dict(self.meta)
        if stage:
            # Copy rather than append in place: `dict(self.meta)` is shallow, so
            # mutating the list would also rewrite the upstream plan's history.
            merged["stages"] = list(merged.get("stages", [])) + [stage]
        merged.update(meta)
        return DataPlan(indices=indices, weights=new_weights, meta=merged)

    def scale_weights(self, index_to_factor: Dict[int, float]) -> "DataPlan":
        """Multiply the per-index weights by `index_to_factor`."""
        base = self.weights if self.weights is not None else np.ones(len(self.indices), dtype=np.float64)
        factors = np.array([index_to_factor.get(int(i), 1.0) for i in self.indices], dtype=np.float64)
        return DataPlan(indices=self.indices.copy(), weights=base * factors, meta=dict(self.meta))

    def domain_histogram(self, domain_ids: Optional[Sequence[int]]) -> Optional[Dict[int, int]]:
        """Count how many of the planned samples come from each domain."""
        if domain_ids is None or len(self.indices) == 0:
            return None
        labels = np.asarray(domain_ids)
        picked = labels[self.indices]
        values, counts = np.unique(picked, return_counts=True)
        return {int(v): int(c) for v, c in zip(values, counts)}

    def describe(self, domain_ids: Optional[Sequence[int]] = None) -> str:
        parts = [f"{len(self.indices)} samples"]
        if self.weights is not None:
            parts.append(f"weights[min={self.weights.min():.3f}, max={self.weights.max():.3f}]")
        hist = self.domain_histogram(domain_ids)
        if hist:
            parts.append("domains=" + ",".join(f"{k}:{v}" for k, v in sorted(hist.items())))
        if self.meta.get("stages"):
            parts.append("via " + " -> ".join(self.meta["stages"]))
        return " | ".join(parts)
