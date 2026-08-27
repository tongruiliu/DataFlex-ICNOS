# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
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

"""Dynamic mixture: a component re-decides the per-domain sampling proportions.

Unlike selection, which narrows the dataset to a set of indices, mixing rebuilds
the dataset from its per-domain sources -- hence `WHOLE_DATASET` rather than an
index list. Two components need more than the proportions: DoReMi weights the loss
per domain (`compute_loss`), and ODM attributes a bandit reward per batch
(`training_step`).
"""

import functools
from typing import Any, List, Optional

import torch
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from torch.utils.data import DataLoader, RandomSampler
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers.trainer_utils import has_length, seed_worker
from transformers.utils import is_datasets_available
from typing_extensions import override

# Registers the built-in mixers into REGISTRY as a side effect.
import dataflex.train.mixer  # noqa: F401
from dataflex.core.registry import REGISTRY
from dataflex.train.hooks import WHOLE_DATASET, DataflexTrainerMixin, group_by_length_requested
from dataflex.utils.load_component import load_component
from dataflex.utils.logging import logger


if is_datasets_available():
    import datasets


class MixTrainer(DataflexTrainerMixin, CustomSeq2SeqTrainer):
    def __init__(self, finetuning_args, processor=None, gen_kwargs=None, model_args=None, **kwargs):
        self.mixture_manager = kwargs.pop("mixture_manager", None)
        self.mixer = None
        self._last_batch = None

        # The pt stage passes `model_args`, the sft stage passes `gen_kwargs`.
        if gen_kwargs is None and model_args is not None:
            super().__init__(finetuning_args=finetuning_args, processor=processor, model_args=model_args, **kwargs)
        else:
            super().__init__(finetuning_args=finetuning_args, processor=processor, gen_kwargs=gen_kwargs, **kwargs)

        if not self.finetuning_args.static_mix and self.mixture_manager is not None:
            name = finetuning_args.component_name
            # Params of the mixer (can substitute ${output_dir})
            sel_params = load_component(
                'mixers',
                finetuning_args.components_cfg_file,
                name,
                runtime_vars={}
            )
            sel_params["mixture_manager"] = self.mixture_manager
            runtime = dict(
                dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                accelerator=self.accelerator,
                data_collator=self.data_collator,
                output_dir=self.args.output_dir,  # for weight logging
            )
            self.mixer = REGISTRY.build("mixer", name, runtime=runtime, cfg=sel_params)
            logger.info(f"[Dataflex] mixer={name}, params={sel_params}")
        elif self.mixture_manager is not None:
            logger.info("[Dataflex] Using static mix proportions during training.")
        else:
            logger.info("[Dataflex] No mixture manager available, using standard training.")

        # SFT data mixer: dynamic MoE
        if getattr(finetuning_args, 'freeze_gate', False):
            frozen_count = 0
            for name_p, param in self.model.named_parameters():
                if "gate" in name_p:
                    param.requires_grad = False
                    frozen_count += 1
            logger.info(f"[Dataflex] Froze {frozen_count} gate parameters (freeze_gate=True)")

        logger.info("[Dataflex] MixTrainer initialized")

    # ------------------------------------------------------------------
    # Schedule
    # ------------------------------------------------------------------

    @property
    def _total_train_batch_size(self) -> int:
        return self.get_total_train_batch_size(self.args)

    def _steps_for_one_pass(self) -> int:
        """Optimizer steps to cover the combined source datasets once."""
        source_size = sum(len(d) for d in self.mixture_manager.sources.values())
        steps_per_epoch = max(1, source_size // self._total_train_batch_size)
        return int(steps_per_epoch * self.args.num_train_epochs)

    def _ensure_initial_snapshot(self) -> None:
        """Draw the opening snapshot if it has not been drawn yet.

        `loader.py` hands this trainer a `None` train_dataset on purpose: the
        first snapshot's size depends on the total batch size, which is not known
        until `train()` has resolved `_train_batch_size`. So the draw happens on
        the first dataloader request rather than in `__init__`.
        """
        if self.train_dataset is not None or self.mixture_manager is None:
            return

        self.print_mixture_info()
        fa = self.finetuning_args
        if fa.static_mix:
            logger.info("[Dataflex] Initial static mix data with initial mixture proportions.")
            steps = fa.train_step if fa.train_step > 0 else self._steps_for_one_pass()
        else:
            logger.info("[Dataflex] Initial warmup data with initial mixture proportions.")
            steps = fa.warmup_step
        self.train_dataset = self.mixture_manager.rebuild(
            num_samples=self._total_train_batch_size * steps
        )

    @override
    def dataflex_step_budget(self) -> Optional[int]:
        fa = self.finetuning_args
        if self.mixture_manager is None:
            return None

        if fa.train_step > 0:
            return fa.train_step

        # update_times < 0 means "keep re-mixing until training ends".
        if fa.static_mix or fa.update_times < 0:
            return self._steps_for_one_pass()

        return fa.warmup_step + fa.update_step * fa.update_times

    @override
    def dataflex_next_indices(self, model, step_id: int) -> Optional[Any]:
        fa = self.finetuning_args
        if fa.static_mix or self.mixer is None or step_id == 0:
            return None
        if step_id >= self.state.max_steps:
            return None

        at_boundary = step_id == fa.warmup_step or (
            step_id > fa.warmup_step and (step_id - fa.warmup_step) % fa.update_step == 0
        )
        if not at_boundary:
            return None

        self.accelerator.wait_for_everyone()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        update_times = (step_id - fa.warmup_step) // fa.update_step + 1
        logger.info(f"[Dataflex] Model training paused, starting the {update_times}th dynamic data mixture...")

        # DoReMi compares per-token losses on a real batch, so the mixer needs one.
        # A planning boundary has no batch of its own; the last one trained on is
        # the closest thing, and it is what the vendored loop passed too.
        batch = self._last_batch
        probs = self.mixer.mix(
            model=model,
            step_id=step_id,
            batch=batch,
            # A collated batch is a BatchEncoding, i.e. a UserDict -- `isinstance(_, dict)`
            # is False for it and would silently drop the domain ids.
            domain_ids=batch.get('domain_id') if hasattr(batch, 'get') else None,
            data_collator=self.data_collator,
            dataset=self.train_dataset,
        )

        self.mixture_manager.set_proportions(probs)
        self.train_dataset = self.mixture_manager.rebuild(
            num_samples=self._total_train_batch_size * fa.update_step
        )
        if self.accelerator.is_main_process:
            logger.info(
                f"[Dataflex] Updated dataloader at step {step_id}, new mixture proportions generated."
            )
            self.print_mixture_info()
        return WHOLE_DATASET

    # ------------------------------------------------------------------
    # Loss space
    # ------------------------------------------------------------------

    @override
    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

        # Remember the batch for the next planning boundary.
        self._last_batch = inputs

        # ODM attributes its bandit reward per batch, which needs the loss and the
        # batch's domain together -- a pairing only `training_step` sees.
        if (
            not self.finetuning_args.static_mix
            and self.mixer is not None
            and hasattr(self.mixer, 'update_batch_info')
        ):
            domain_id = inputs.get('domain_id') if hasattr(inputs, 'get') else None
            if torch.is_tensor(domain_id):
                domain_id = int(domain_id.view(-1)[0].item()) if domain_id.numel() else None
            if domain_id is not None:
                self.mixer.update_batch_info(float(loss.detach().float().cpu()), domain_id)

        return loss

    @override
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """DoReMi weights the loss by the sample's domain; other mixers do not."""
        is_doremi = (
            self.mixer is not None
            and hasattr(self.mixer, 'get_current_doremi_weights')
            and not self.finetuning_args.static_mix
        )
        if not is_doremi:
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )

        domain_ids = inputs.get('domain_id')
        if domain_ids is None:
            logger.warning("[DoremiMixer] No domain_id in batch, using standard loss computation")
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )

        domain_weights = self.mixer.get_current_doremi_weights()

        labels = inputs.pop("labels") if (self.label_smoother is not None and "labels" in inputs) else None
        outputs = model(**inputs)

        if labels is None:
            loss = outputs["loss"] if isinstance(outputs, dict) and "loss" in outputs else outputs[0]
            return (loss, outputs) if return_outputs else loss

        logits = getattr(outputs, "logits", None)
        if logits is None:
            # Nothing to reduce per sample; report the model's own loss unweighted.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            logger.warning("[DoremiMixer] Model returned scalar loss, cannot apply per-sample reweighting")
            return (loss, outputs) if return_outputs else loss

        if self.label_smoother is not None:
            per_sample_loss = self.label_smoother(outputs, labels, shift_labels=True)
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
            per_token_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            ).view(shift_labels.size())
            valid_mask = (shift_labels != -100)
            per_sample_loss = (
                (per_token_loss * valid_mask.float()).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
            )

        domain_weights_tensor = torch.tensor(
            domain_weights, dtype=per_sample_loss.dtype, device=per_sample_loss.device
        )
        sample_weights = domain_weights_tensor[domain_ids]
        weighted_loss = (per_sample_loss * sample_weights).mean()

        if self.accelerator.is_main_process and torch.rand(1).item() < 0.01:  # ~1% of steps
            logger.info(
                f"[DoremiMixer] Loss reweighting - domain_weights: {domain_weights}, "
                f"sample mean loss: {per_sample_loss.mean().item():.4f}, "
                f"weighted loss: {weighted_loss.item():.4f}"
            )

        return (weighted_loss, outputs) if return_outputs else weighted_loss

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    @override
    def train(self, *args, **kwargs):
        out = super().train(*args, **kwargs)
        if self.mixer is not None and hasattr(self.mixer, 'save_average_weights'):
            if self.accelerator.is_main_process:
                logger.info("[Dataflex] Saving DoReMi average weights for Step 2...")
                self.mixer.save_average_weights(self.args.output_dir)
                logger.info("[Dataflex] DoReMi Step 2 complete. Use the saved average weights for Step 2.")
        return out

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    @override
    def _get_train_sampler(self, train_dataset=None) -> Optional[torch.utils.data.Sampler]:
        if train_dataset is None:
            train_dataset = self.train_dataset
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(train_dataset)
        if train_dataset is None or not has_length(train_dataset):
            return None

        if group_by_length_requested(self.args):
            if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
                lengths = (
                    train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = (
                self.processing_class.model_input_names[0] if self.processing_class is not None else None
            )
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )

        return RandomSampler(train_dataset)

    @override
    def get_train_dataloader(self, indices: Optional[List[int]] = None) -> DataLoader:
        """Training DataLoader over the current mixture snapshot."""
        self._ensure_initial_snapshot()
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if indices is not None:
            train_dataset = torch.utils.data.Subset(train_dataset, indices)

        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        if self.mixer is not None:
            self.mixer.data_collator = data_collator

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler(train_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            if self.args.dataloader_num_workers > 0:
                # seed_worker takes (worker_id, num_workers, rank); DataLoader only passes worker_id.
                dataloader_params["worker_init_fn"] = functools.partial(
                    seed_worker,
                    num_workers=self.args.dataloader_num_workers,
                    rank=self.args.process_index,
                )
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def print_mixture_info(self, prefix: str = "[Dataflex]"):
        """Log the mixer's current sampling proportions."""
        if self.mixture_manager is None:
            logger.info(f"{prefix} No mixture manager available")
            return

        rule = self.mixture_manager.sample_rule
        probs = self.mixture_manager._current_probs()
        sources = self.mixture_manager.names

        labels = {
            "mixture": "Mixture proportions",
            "stratified": "Stratified proportions (by dataset size)",
            "uniform": "Uniform proportions",
        }
        label = labels.get(rule, f"Proportions (rule={rule})")
        logger.info(f"{prefix} {label}: {probs} | sources={sources}")
