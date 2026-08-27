# Copyright 2025 the DataFlex team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The seams DataFlex needs from the stock transformers training loop.

DataFlex used to vendor `Trainer._inner_training_loop` three times over to reach
three things the loop does not expose: which samples the next optimizer step gets,
how many optimizer steps the run has, and the `(inputs, loss)` pair inside a step.
The copies restated roughly three quarters of it upstream code,
code, so every transformers release had to be re-diffed against all three.

All three are reachable without a copy:

    which samples   ->  get_batch_samples override  (once per optimizer step)
    how many steps  ->  args.max_steps, set before training starts
    loss / backward ->  compute_loss / training_step overrides

Subclasses say *what* they want by implementing the three `dataflex_*` methods
below; this mixin handles *when*, and never touches the loop. Nothing here is
version-specific, so the same code runs on transformers v4 and v5.

One thing to know about `get_batch_samples`: transformers hands it an
`epoch_iterator` built from `self.train_dataset`, and we deliberately ignore it,
pulling from a stage dataloader of our own instead. That is the whole mechanism by
which the training data can change mid-run. It is a public method with a two-in,
two-out contract, but changing data is not its advertised purpose, so it is the
first thing to re-check on a transformers upgrade.
"""

from typing import Any, List, Optional

from dataflex.utils.logging import logger


#: Return this from `dataflex_next_indices` when the stage changed but no index
#: restriction applies -- the mixture path swaps `self.train_dataset` wholesale
#: instead of narrowing it, so there are no indices to hand back.
WHOLE_DATASET = object()


def group_by_length_requested(args) -> bool:
    """Whether length-grouped batching was asked for.

    transformers v5 replaced the `group_by_length` flag with
    `train_sampling_strategy="group_by_length"`. Reading both keeps this check
    from becoming the thing that breaks on the next rename.
    """
    if getattr(args, "train_sampling_strategy", None) == "group_by_length":
        return True
    return bool(getattr(args, "group_by_length", False))


class _StageIterator:
    """Endless iterator over one stage's dataloader.

    A stage holds one interval's worth of data, not one epoch's, so running dry
    mid-interval is normal -- drop_last and per-rank sharding both shorten it.
    Re-reading keeps the step budget, not the stage length, in charge of when
    training ends. Truly empty stages still raise `StopIteration` so the caller
    is not spun forever.
    """

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iterator = iter(dataloader)
        self._epoch = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._iterator)
        except StopIteration:
            pass

        self._epoch += 1
        if hasattr(self.dataloader, "set_epoch"):
            self.dataloader.set_epoch(self._epoch)
        self._iterator = iter(self.dataloader)
        return next(self._iterator)


class DataflexTrainerMixin:
    """Dynamic training data, expressed as overrides of stock Trainer methods.

    Handles the step budget and the batch source. Trainers that also need to reach
    into the loss do so by overriding `compute_loss` / `training_step` themselves;
    those need no help from here.
    """

    # ------------------------------------------------------------------
    # What a subclass implements
    # ------------------------------------------------------------------

    def dataflex_step_budget(self) -> Optional[int]:
        """Total optimizer steps for the run, or None to let upstream decide.

        Declaring this before training starts, rather than patching
        `state.max_steps` once the loop is running, is what keeps the LR schedule
        honest: the scheduler horizon and the fractional `logging_steps` /
        `save_steps` are all derived from it.
        """
        return None

    def dataflex_initial_indices(self) -> Optional[List[int]]:
        """Indices for the first stage, or None to train on the whole dataset."""
        return None

    def dataflex_next_indices(self, model, step_id: int) -> Optional[Any]:
        """Indices for the next stage, `WHOLE_DATASET`, or None to keep the current stage.

        Called once per optimizer step, before any batch of that step is pulled,
        so a component sees a model whose gradients are zeroed and whose
        parameters are synchronized.
        """
        return None

    # ------------------------------------------------------------------
    # Seam 1: how many optimizer steps
    # ------------------------------------------------------------------

    def train(self, *args, **kwargs):
        """Publish the step budget as `args.max_steps`, then hand over to upstream.

        The obvious alternative is to override `set_initial_training_values` and
        return the budget directly. That works on v5 but not on v4: v4's loop takes
        its per-epoch iteration count from `len(epoch_dataloader)` rather than from
        that method's `num_update_steps_per_epoch`, so the budget would be ignored.

        Setting `max_steps` instead lets each version derive its own consistent set
        of values -- LR schedule horizon, epoch count, stopping condition -- and
        keeps this class free of any version branch.
        """
        budget = self.dataflex_step_budget()
        if budget is not None and budget != self.args.max_steps:
            logger.info(f"[Dataflex] step budget = {budget} optimizer steps (max_steps)")
            self.args.max_steps = budget
        return super().train(*args, **kwargs)

    # ------------------------------------------------------------------
    # Seam 2: which samples
    # ------------------------------------------------------------------

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        """Serve one optimizer step's batches from the current stage.

        `epoch_iterator` is upstream's iterator over the full dataset; it is
        unused on purpose. See the module docstring.

        Only the *source* of the batches is ours. Pulling them and counting
        `num_items_in_batch` stays with `super()`, because that count decides how
        the loss is scaled across an accumulation window and is gathered across
        ranks -- a second implementation of it would be a silent correctness bug
        the first time upstream refined the counting rules.
        """
        indices = self.dataflex_next_indices(self.model_wrapped, self.state.global_step)
        if indices is WHOLE_DATASET:
            self._dataflex_open_stage(None)
        elif indices is not None:
            self._dataflex_open_stage(indices)
        elif getattr(self, "_dataflex_iterator", None) is None:
            self._dataflex_open_stage(self.dataflex_initial_indices())

        return super().get_batch_samples(self._dataflex_iterator, num_batches, device)

    def _dataflex_open_stage(self, indices: Optional[List[int]]) -> None:
        """Point the batch source at a new set of indices."""
        dataloader = self.get_train_dataloader(indices)
        self._dataflex_iterator = _StageIterator(dataloader)
        # The vendored loops never did this, so callbacks kept reporting the
        # first dataloader for the whole run.
        self.callback_handler.train_dataloader = dataloader
