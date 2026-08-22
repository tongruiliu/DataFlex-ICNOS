"""One place where per-sample scores are computed and shared.

A loss-based selector, a dynamic reorder and a loss weighter all want the same
per-sample signal. Run independently each does its own full forward pass over the
pool, so composing three of them triples the scoring bill for no extra
information. The ScoreBoard computes a named metric at most once per planning
boundary and hands the same array to every stage that asks.

Scores are stored densely over dataset positions so a stage can index them
directly, with NaN for "not scored". Staleness is tracked per metric so a stage
can opt into reusing a slightly old signal (`max_age`) rather than forcing a
recompute.
"""

import json
import os
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataflex.utils.logging import logger
from dataflex.utils.loss_utils import per_sample_loss_from_outputs


class _IndexedDataset(Dataset):
    """Attach the originating index to each item.

    Scoring is sharded across ranks and gathered back in an order nobody
    controls, so every sample has to carry its own identity.
    """

    def __init__(self, dataset, indices: Optional[Sequence[int]] = None):
        self.dataset = dataset
        self.indices = list(indices) if indices is not None else None

    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.dataset)

    def __getitem__(self, i):
        idx = self.indices[i] if self.indices is not None else i
        item = self.dataset[idx]
        if isinstance(item, dict):
            return {"idx": idx, **item}
        return {"idx": idx, "item": item}


class ScoreBoard:
    """Dense per-sample score cache with staleness tracking."""

    def __init__(
        self,
        dataset=None,
        accelerator=None,
        data_collator=None,
        dataset_size: int = 0,
        batch_size: int = 8,
        num_workers: int = 2,
        cache_dir: Optional[str] = None,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.accelerator = accelerator
        self.data_collator = data_collator
        self.dataset_size = int(dataset_size or (len(dataset) if dataset is not None else 0))
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.cache_dir = cache_dir
        self.seed = int(seed)

        self._values: Dict[str, np.ndarray] = {}
        #: When each position was last scored, -1 for never. Per position rather
        #: than per metric because a partial fill would otherwise re-date the
        #: whole array: one stage with a permissive `max_age` triggering a fill
        #: for a handful of positions would make arbitrarily old values look
        #: fresh to the next stage that asked for age 0.
        self._scored_at: Dict[str, np.ndarray] = {}
        self._step: Dict[str, int] = {}
        #: How many times a metric was actually computed, versus served from
        #: cache. Exposed so a run can show that sharing is working.
        self.compute_calls: Dict[str, int] = {}
        self.cache_hits: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Cache protocol
    # ------------------------------------------------------------------

    def age(self, metric: str, step_id: int) -> Optional[int]:
        if metric not in self._step:
            return None
        return step_id - self._step[metric]

    def has_fresh(self, metric: str, step_id: int, max_age: Optional[int] = 0) -> bool:
        age = self.age(metric, step_id)
        if age is None:
            return False
        if max_age is None:
            return True
        return age <= max_age

    def put(self, metric: str, indices: Sequence[int], values: Sequence[float], step_id: int) -> None:
        arr = self._values.get(metric)
        if arr is None:
            arr = np.full(self.dataset_size, np.nan, dtype=np.float64)
        stamps = self._scored_at.get(metric)
        if stamps is None:
            stamps = np.full(self.dataset_size, -1, dtype=np.int64)
        idx = np.asarray(list(indices), dtype=np.int64)
        arr[idx] = np.asarray(list(values), dtype=np.float64)
        stamps[idx] = int(step_id)
        self._values[metric] = arr
        self._scored_at[metric] = stamps
        self._step[metric] = int(step_id)

    def get(
        self,
        metric: str,
        indices: Sequence[int],
        step_id: int = 0,
        compute_fn: Optional[Callable[[Sequence[int]], Sequence[float]]] = None,
        max_age: Optional[int] = 0,
        fill: str = "median",
    ) -> np.ndarray:
        """Return scores for `indices`, computing them only if necessary.

        Args:
            metric: cache key, e.g. "loss".
            indices: positions to score.
            step_id: current global step, for staleness bookkeeping.
            compute_fn: called with the positions that still need a value. When
                omitted and nothing is cached, raises.
            max_age: reuse a cached value if it is at most this many steps old.
                `0` means only this boundary's value counts; `None` means any
                cached value is acceptable.
            fill: what to do about positions that remain unscored —
                "median" (neutral: keeps them mid-curriculum) or "nan".
        """
        indices = list(indices)
        if not indices:
            return np.zeros(0, dtype=np.float64)

        idx = np.asarray(indices, dtype=np.int64)
        cached = self._values.get(metric)
        stamps = self._scored_at.get(metric)

        if cached is None or stamps is None:
            to_compute = indices
        else:
            # A position is usable when it holds a real number that is young
            # enough. Judging each position on its own stamp is what keeps a
            # partial fill from passing stale values off as fresh.
            usable = (stamps[idx] >= 0) & ~np.isnan(cached[idx])
            if max_age is not None:
                usable &= (step_id - stamps[idx]) <= max_age
            stale = idx[~usable]
            if stale.size == 0:
                self.cache_hits[metric] = self.cache_hits.get(metric, 0) + 1
                logger.info(
                    f"[Dataflex][Lego] '{metric}' served from cache "
                    f"(oldest={int((step_id - stamps[idx]).max())} steps, {len(indices)} positions)"
                )
                return cached[idx].copy()
            to_compute = stale.tolist()

        if compute_fn is None:
            if cached is None:
                raise KeyError(f"no scores for metric '{metric}' and no compute_fn given")
            to_compute = []

        if to_compute:
            self.compute_calls[metric] = self.compute_calls.get(metric, 0) + 1
            logger.info(
                f"[Dataflex][Lego] computing '{metric}' for {len(to_compute)} positions "
                f"(compute #{self.compute_calls[metric]} this run)"
            )
            values = compute_fn(to_compute)
            self.put(metric, to_compute, values, step_id)

        arr = self._values[metric]
        out = arr[idx].copy()
        if np.isnan(out).any():
            if fill == "median":
                finite = out[np.isfinite(out)]
                out[np.isnan(out)] = float(np.median(finite)) if finite.size else 0.0
            # "nan" leaves them for the caller to deal with
        return out

    # ------------------------------------------------------------------
    # The built-in metric: per-sample loss under the current model
    # ------------------------------------------------------------------

    def per_sample_loss_fn(self, model, step_id: int, max_samples: Optional[int] = None):
        """Build a `compute_fn` for `get("loss", ...)`."""

        def compute(positions: Sequence[int]) -> List[float]:
            target = list(positions)
            if max_samples is not None and len(target) > max_samples:
                rng = np.random.default_rng(self.seed + int(step_id))
                picked = sorted(rng.choice(len(target), size=max_samples, replace=False).tolist())
                subset = [target[i] for i in picked]
                logger.info(
                    f"[Dataflex][Lego] scoring a {len(subset)}/{len(target)} subsample "
                    f"(max_samples={max_samples}); the rest stay unscored and take the median"
                )
                scored = self._compute_losses(model, subset, step_id)
                return [scored.get(p, np.nan) for p in target]

            scored = self._compute_losses(model, target, step_id)
            return [scored.get(p, np.nan) for p in target]

        return compute

    def _compute_losses(self, model, positions: Sequence[int], step_id: int) -> Dict[int, float]:
        if model is None:
            raise ValueError("per-sample loss needs a model")
        if not positions:
            return {}

        was_training = model.training
        model.eval()

        loader = DataLoader(
            _IndexedDataset(self.dataset, positions),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
        )
        loader = self.accelerator.prepare(loader)

        local_losses, local_indices = [], []
        for batch in tqdm(
            loader,
            desc=f"[Lego scoring @ step {step_id}]",
            disable=not (self.accelerator is None or self.accelerator.is_main_process),
            dynamic_ncols=True,
        ):
            idx = batch["idx"]
            if not torch.is_tensor(idx):
                idx = torch.tensor(idx, dtype=torch.long, device=self.accelerator.device)
            idx = idx.view(-1).to(dtype=torch.long)

            inputs = {k: v for k, v in batch.items() if k not in ("idx", "domain_id")}
            with torch.no_grad():
                outputs = model(**inputs)
                per_sample = per_sample_loss_from_outputs(outputs, inputs)

            local_losses.append(per_sample)
            local_indices.append(idx)

        if local_losses:
            local_losses = torch.cat(local_losses, dim=0).float()
            local_indices = torch.cat(local_indices, dim=0)
        else:
            device = self.accelerator.device if self.accelerator is not None else "cpu"
            local_losses = torch.zeros(0, device=device)
            local_indices = torch.zeros(0, dtype=torch.long, device=device)

        if self.accelerator is not None:
            all_losses = self.accelerator.gather(local_losses).detach().cpu().tolist()
            all_indices = self.accelerator.gather(local_indices).detach().cpu().tolist()
        else:
            all_losses = local_losses.detach().cpu().tolist()
            all_indices = local_indices.detach().cpu().tolist()

        # Gathering pads the final batch, so an index can come back twice.
        # First occurrence wins, which keeps this deterministic.
        out: Dict[int, float] = {}
        for value, i in zip(all_losses, all_indices):
            if i not in out and np.isfinite(value):
                out[int(i)] = float(value)

        if was_training:
            model.train()
        return out

    # ------------------------------------------------------------------
    # Persistence, reusing the selector cache layout
    # ------------------------------------------------------------------

    def save(self, step_id: int, indices: Optional[Sequence[int]] = None) -> Optional[str]:
        """Dump the board in the `{"indices", "metric": {...}}` selector format.

        Same shape `selector_io.save_selection` writes, so a selector run and a
        lego run produce interchangeable artifacts and either can seed the
        other.
        """
        if not self.cache_dir:
            return None
        if self.accelerator is not None and not self.accelerator.is_main_process:
            return None
        if not self._values:
            return None

        idx = list(indices) if indices is not None else list(range(self.dataset_size))
        payload = {
            "indices": [int(i) for i in idx],
            "metric": {
                name: [None if np.isnan(v) else float(v) for v in arr[np.asarray(idx, dtype=np.int64)]]
                for name, arr in self._values.items()
            },
            "computed_at_step": {k: int(v) for k, v in self._step.items()},
        }
        os.makedirs(self.cache_dir, exist_ok=True)
        path = os.path.join(self.cache_dir, f"scoreboard_step_{step_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        logger.info(f"[Dataflex][Lego] scoreboard saved to {path}")
        return path

    def summary(self) -> str:
        parts = []
        for metric in sorted(set(self.compute_calls) | set(self.cache_hits)):
            parts.append(
                f"{metric}: computed {self.compute_calls.get(metric, 0)}x, "
                f"reused {self.cache_hits.get(metric, 0)}x"
            )
        return "; ".join(parts) if parts else "no metrics requested"
