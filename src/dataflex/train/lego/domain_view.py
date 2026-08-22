"""A `mixture_manager` stand-in backed by one dataset plus domain labels.

Every mixer takes a `mixture_manager` and reads a small, stable set of things off
it: `names`, sometimes `initial_proportions`, and sometimes `sources[name]` for
per-domain evaluation. `MixedProportionManager` satisfies that by *being* a
collection of per-source datasets, which is exactly the representation that makes
composition impossible (it leaves `train_dataset = None`).

This view offers the same attributes over the composition representation — one
concatenated dataset plus a `domain_ids` label per row — so existing mixers run
unmodified while selectors and reorder stages still have a real dataset to index.

What it deliberately does *not* do is materialise a mixed dataset.
`MixStage` turns proportions into an index quota, so `set_proportions` here only
records the latest vector for logging.
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from dataflex.utils.logging import logger


class DomainView:
    def __init__(
        self,
        names: Sequence[str],
        domain_ids,
        dataset=None,
        initial_proportions: Optional[Sequence[float]] = None,
        mixer_eval_datasets: Optional[Dict[str, Any]] = None,
    ):
        self.names: List[str] = list(names)
        self.domain_ids = np.asarray(domain_ids, dtype=np.int64)
        self.dataset = dataset
        self.initial_proportions = list(initial_proportions) if initial_proportions is not None else None
        self.mixer_eval_datasets = mixer_eval_datasets or {}

        self.k = len(self.names)
        self.sizes = {
            name: int((self.domain_ids == i).sum()) for i, name in enumerate(self.names)
        }
        self.probs = (
            np.asarray(self.initial_proportions, dtype=np.float64)
            if self.initial_proportions and len(self.initial_proportions) == self.k
            else np.ones(max(1, self.k), dtype=np.float64) / max(1, self.k)
        )

        self._sources: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # What mixers read
    # ------------------------------------------------------------------

    @property
    def sources(self) -> Dict[str, Any]:
        """Per-domain dataset views, built on demand.

        `dynamic_moe` both reads and re-assigns entries here, so this is a plain
        mutable dict rather than a computed mapping.
        """
        if self._sources is None:
            self._sources = {}
            if self.dataset is not None and hasattr(self.dataset, "select"):
                for i, name in enumerate(self.names):
                    rows = np.flatnonzero(self.domain_ids == i).tolist()
                    self._sources[name] = self.dataset.select(rows)
            else:
                logger.warning(
                    "[Dataflex][Lego] the composed dataset does not support .select(); "
                    "per-domain source views are unavailable to mixers that need them"
                )
        return self._sources

    @property
    def per_source(self) -> Dict[str, Any]:
        # Some mixer variants look for this name instead.
        return self.sources

    def set_proportions(self, proportions: Optional[Sequence[float]]) -> None:
        """Record the latest vector. Allocation itself belongs to `MixStage`."""
        if proportions is None:
            return
        probs = np.asarray(proportions, dtype=np.float64).reshape(-1)
        if probs.size and np.all(np.isfinite(probs)) and probs.sum() > 0:
            self.probs = probs / probs.sum()

    def domain_indices(self, domain: int) -> List[int]:
        return np.flatnonzero(self.domain_ids == domain).tolist()

    def describe(self) -> str:
        return ", ".join(f"{n}={self.sizes.get(n, 0)}" for n in self.names)
