# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
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

"""DataFlex's `DataArguments`, which `cli.patch_finetune_params` installs over LlamaFactory's.

Subclassing rather than vendoring a copy: LlamaFactory keeps adding fields to
`DataArguments` (0.9.6 added `preserve_thinking`, which `data/template.py` reads
unconditionally), and a copy has to mirror each one or the run dies on an
`AttributeError` that points at DataFlex rather than at the missing mirror.
"""

from dataclasses import dataclass, field
from typing import Optional

from llamafactory.hparams.data_args import DataArguments as _DataArguments


@dataclass
class DataArguments(_DataArguments):
    r"""LlamaFactory's data arguments plus the ones dynamic mixing needs."""

    mixture_sample_rule: Optional[str] = field(
        default=None,
        metadata={"help": "Rule to sample from per-source datasets, e.g. 'proportional', 'fixed'."},
    )
    init_mixture_proportions: Optional[list[float]] = field(
        default=None,
        metadata={"help": "Initial proportions for sampling from each dataset in mixture."},
    )
    mixer_eval_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "Independent eval dataset(s) for dynamic mixer (e.g. gate load evaluation). Use commas to separate multiple datasets."},
    )

    def __post_init__(self):
        super().__post_init__()

        if isinstance(self.mixer_eval_dataset, str):
            self.mixer_eval_dataset = [item.strip() for item in self.mixer_eval_dataset.split(",")]
