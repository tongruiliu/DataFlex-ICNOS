# Copyright 2025 the LlamaFactory team.
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

"""DataFlex's `FinetuningArguments`, which `cli.patch_finetune_params` installs over LlamaFactory's.

Subclassing rather than vendoring a copy. LlamaFactory 0.9.6 added ten fields
here, and `train/tuner.py:run_exp` reads `use_hyper_parallel` in its very first
dispatch branch -- so a copy that predates them stops every run before it starts.
Inheriting means new upstream fields arrive on their own.
"""

import importlib
from dataclasses import dataclass, field

from llamafactory.hparams.finetuning_args import FinetuningArguments


@dataclass
class DynamicFinetuningArguments(FinetuningArguments):
    r"""LlamaFactory's finetuning arguments plus the ones dynamic training needs."""

    early_stopping_min_delta: float = field(
        default=0.0,
        metadata={"help": "Minimum improvement (abs) on the monitored metric to reset early stopping patience."},
    )
    # hyperparameter for dynamic training
    train_type: str = field(
        default="static",
        metadata={
            "help": (
                "Specifies the type of training to use when `enable_dynamic_train` is True. "
                "Choices: ['static', 'dynamic_select', 'dynamic_mix', 'dynamic_weighting']. "
                "If static, uses default LlamaFactory trainer."
            )
        }
    )
    components_cfg_file: str = field(
        default="configs/components.yaml",
        metadata={"help": "Path to the components configuration file."},
    )
    component_name: str = field(
        default="Loss",
        metadata={"help": "The component name defined in the components configuration file."},
    )
    warmup_step: int = field(
        default=0,
        metadata={"help": "Warm up steps for dynamic training"},
    )
    update_step: int = field(
        default=0,
        metadata={"help": "Update steps for dynamic select or mix training"},
    )
    update_times: int = field(
        default=1,
        metadata={"help": "Update times per Flex epoch for dynamic selection. Use <= 0 for no fixed update count."},
    )
    static_mix: bool = field(
        default=False,
        metadata={"help": "Whether or not to fix the static mix ratio in dynamic mix training."},
    )
    train_step: int = field(
        default=0,
        metadata={"help": "Optional total training steps. If set, overrides num_train_epochs."},
    )
    freeze_gate: bool = field(
        default=False,
        metadata={"help": "Whether to freeze gate parameters in MoE models during SFT training."},
    )

    def __post_init__(self):
        super().__post_init__()

        if self.early_stopping_min_delta < 0:
            raise ValueError("`early_stopping_min_delta` must be non-negative.")

        if self.early_stopping_steps is not None:
            _patch_early_stopping_callback(self.early_stopping_min_delta)


def _patch_early_stopping_callback(min_delta: float) -> None:
    r"""Teach LlamaFactory's early stopping to respect a minimum improvement."""
    from transformers import EarlyStoppingCallback as HFEarlyStoppingCallback

    try:
        tuner = importlib.import_module("llamafactory.train.tuner")
    except ModuleNotFoundError:
        return

    class DataFlexEarlyStoppingCallback(HFEarlyStoppingCallback):
        def __init__(self, early_stopping_patience: int):
            super().__init__(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=min_delta,
            )

    tuner.EarlyStoppingCallback = DataFlexEarlyStoppingCallback
