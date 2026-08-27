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

"""Dynamic re-weighting: the loss is scaled per sample, the data is not touched.

This used to carry a copy of `Trainer._inner_training_loop` so it could reach the
`(model, inputs, num_items_in_batch)` triple. `training_step` receives exactly
that triple, so the copy bought nothing: 530 vendored lines existed to change the
two below, and every transformers release then had to be re-diffed against them.
"""

from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from typing_extensions import override

# Registers the built-in weighters into REGISTRY as a side effect.
import dataflex.train.weighter  # noqa: F401
from dataflex.core.registry import REGISTRY
from dataflex.utils.load_component import load_component
from dataflex.utils.logging import logger


class WeightTrainer(CustomSeq2SeqTrainer):
    def __init__(self, finetuning_args, processor=None, gen_kwargs=None, **kwargs):
        super().__init__(finetuning_args=finetuning_args, processor=processor, gen_kwargs=gen_kwargs, **kwargs)
        name = finetuning_args.component_name
        # Params of the weighter (can substitute ${output_dir})
        sel_params = load_component(
            'weighters',
            finetuning_args.components_cfg_file,
            name,
            runtime_vars={}
        )

        # Supply the runtime dependencies; static components ignore what they do not need.
        runtime = dict(
            dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            accelerator=self.accelerator,
            data_collator=self.data_collator,
        )

        self.weighter = REGISTRY.build("weighter", name, runtime=runtime, cfg=sel_params)
        logger.info(f"[Dataflex] weighter={name}, params={sel_params}")
        logger.info("[Dataflex] WeightTrainer initialized")

    @override
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Hand the step to the weighter once warmup is over.

        Before `warmup_step` the weighter is asked to run unweighted rather than
        being bypassed, so that a weighter which accumulates statistics still
        observes the warmup batches.
        """
        use_weighter = self.state.global_step >= self.finetuning_args.warmup_step
        return self.weighter.training_step(self, model, inputs, num_items_in_batch, use_weighter)
