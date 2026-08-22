
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Literal, Optional, Union

import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from torch.utils.data import Dataset as TorchDataset

from llamafactory.extras.constants import FILEEXT2TYPE
from llamafactory.extras.misc import check_version, has_tokenized_data
from llamafactory.data.converter import align_dataset
from llamafactory.data.data_utils import get_dataset_module, merge_dataset, read_cloud_json, split_dataset
from llamafactory.data.parser import get_dataset_list
from llamafactory.data.processor import (
    FeedbackDatasetProcessor,
    PackedSupervisedDatasetProcessor,
    PairwiseDatasetProcessor,
    PretrainDatasetProcessor,
    SupervisedDatasetProcessor,
    UnsupervisedDatasetProcessor,
)
from llamafactory.data.loader import _get_merged_dataset, _get_preprocessed_dataset

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import PreTrainedTokenizer, ProcessorMixin, Seq2SeqTrainingArguments

    from llamafactory.hparams import DataArguments, ModelArguments
    from llamafactory.data.data_utils import DatasetModule
    from llamafactory.data.parser import DatasetAttr
    from llamafactory.data.processor import DatasetProcessor
    from llamafactory.data.template import Template

import logging
import sys
logging.basicConfig(level=logging.INFO)
handler = logging.StreamHandler(sys.stdout)
logger = logging.getLogger(__name__)
logger.addHandler(handler)

from ..dataset.mixed_proportion_manager import MixedProportionManager

def get_dataset(
    template: "Template",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
) -> "DatasetModule":
    r"""Get the train dataset and optionally gets the evaluation dataset."""
    # Load tokenized dataset if path exists
    if data_args.tokenized_path is not None:
        if has_tokenized_data(data_args.tokenized_path):
            logger.warning_rank0("Loading dataset from disk will ignore other data arguments.")
            tokenized_data = load_from_disk(data_args.tokenized_path)
            dataset_module = get_dataset_module(tokenized_data)
            if data_args.streaming:
                dataset_module["train_dataset"] = dataset_module["train_dataset"].to_iterable_dataset()

            logger.info_rank0(f"Loaded tokenized dataset from {data_args.tokenized_path}.")
            return dataset_module

        if data_args.streaming:
            raise ValueError("Turn off `streaming` when saving dataset to disk.")

    # Load and preprocess dataset
    with training_args.main_process_first(desc="load dataset", local=(not data_args.data_shared_file_system)):
        dataset = _get_merged_dataset(data_args.dataset, model_args, data_args, training_args, stage)
        eval_dataset = _get_merged_dataset(
            data_args.eval_dataset,
            model_args,
            data_args,
            training_args,
            stage,
            return_dict=data_args.eval_on_each_dataset,
        )
        per_source_raw = None
        logger.info_rank0("[Dataflex] Mixture enabled: building per-source raw datasets for dynamic mixing.")
        per_source_raw = _get_merged_dataset(
            data_args.dataset, model_args, data_args, training_args, stage, return_dict=True
        )
        logger.info_rank0(f"[Dataflex] Loaded per-source raw datasets: {list(per_source_raw.keys())} "
                            f"(num_sources={len(per_source_raw)})")

    with training_args.main_process_first(desc="pre-process dataset", local=(not data_args.data_shared_file_system)):
        dataset = _get_preprocessed_dataset(
            dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=False
        )
        if isinstance(eval_dataset, dict):
            for eval_name, eval_data in eval_dataset.items():
                eval_dataset[eval_name] = _get_preprocessed_dataset(
                    eval_data, data_args, training_args, stage, template, tokenizer, processor, is_eval=True
                )
        else:
            eval_dataset = _get_preprocessed_dataset(
                eval_dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=True
            )

        dataset_dict = split_dataset(dataset, eval_dataset, data_args, seed=training_args.seed)
        if data_args.tokenized_path is not None:  # save tokenized dataset to disk
            if training_args.should_save:
                dataset_dict.save_to_disk(data_args.tokenized_path)
                logger.info_rank0(f"Tokenized dataset is saved at {data_args.tokenized_path}.")
                logger.info_rank0(f"Please launch the training with `tokenized_path: {data_args.tokenized_path}`.")

        dataset_module = get_dataset_module(dataset_dict)

    logger.info_rank0("[Dataflex] Preprocessing per-source datasets for dynamic mixing...")
    per_source_pp = {
        name: _get_preprocessed_dataset(
            ds, data_args, training_args, stage, template, tokenizer, processor, is_eval=False
        )
        for name, ds in per_source_raw.items()
    }
    sizes_str = {name: len(ds) for name, ds in per_source_pp.items()}
    logger.info_rank0(f"[Dataflex] Per-source preprocessed sizes: {sizes_str}")

    # Print initial proportion configuration
    logger.info_rank0(f"[Dataflex] sample_rule={data_args.mixture_sample_rule} | "
                        f"proportions={data_args.init_mixture_proportions} | "
                        f"seed={training_args.seed}")

    manager = MixedProportionManager(
        per_source=per_source_pp,
        sample_rule=data_args.mixture_sample_rule,
        proportions=data_args.init_mixture_proportions,
        seed=training_args.seed,
        logger=logger,
    )

    # ── Load independent eval datasets for mixer (e.g. gate load evaluation) ──
    mixer_eval_names = getattr(data_args, 'mixer_eval_dataset', None)
    if mixer_eval_names:
        logger.info_rank0(f"[Dataflex] Loading mixer eval datasets: {mixer_eval_names}")
        with training_args.main_process_first(desc="load mixer eval dataset", local=(not data_args.data_shared_file_system)):
            mixer_eval_raw = _get_merged_dataset(
                mixer_eval_names, model_args, data_args, training_args, stage, return_dict=True
            )
        mixer_eval_pp = {}
        for name, ds in mixer_eval_raw.items():
            mixer_eval_pp[name] = _get_preprocessed_dataset(
                ds, data_args, training_args, stage, template, tokenizer, processor, is_eval=True
            )
        # Map eval dataset names back to training domain names:
        # e.g. "code_eval" -> "code", so dynamic_moe_mixer can look up by domain name
        mixer_eval_by_domain = {}
        for eval_name, eval_ds in mixer_eval_pp.items():
            domain = eval_name.replace("_eval", "")
            mixer_eval_by_domain[domain] = eval_ds
            logger.info_rank0(f"[Dataflex] Mixer eval: '{eval_name}' -> domain '{domain}' ({len(eval_ds)} samples)")
        manager.mixer_eval_datasets = mixer_eval_by_domain

    # Optional: expose manager to external code (for callback reconstruction)
    # e.g. attached to dataset_module (Trainer doesn't use this field)
    dataset_module["train_dataset"] = None # Placeholder, trainer will rebuild
    dataset_module["mixture_manager"] = manager
    logger.info_rank0("[Dataflex] Exposed mixture_manager for runtime re-mixing.")

    return dataset_module


# ======================================================================
# Reordering
# ======================================================================


@contextmanager
def _capture_raw_scores(score_field: str):
    """Snapshot a score column from each raw dataset before it is discarded.

    `align_dataset` is the last thing `_load_single_dataset` does, and it maps
    with `remove_columns=column_names`, so any score field in the raw JSONL dies
    there. Wrapping it lets us read the column while reusing LlamaFactory's
    loading path verbatim.

    The capture happens after `num_samples` / `max_samples` truncation, so the
    captured vector lines up with the rows that survive, and datasets are
    captured in `dataset:` order, which is the order `concat` merges them in.
    """
    import llamafactory.data.loader as lf_loader

    original = lf_loader.align_dataset
    captured: List[Optional[np.ndarray]] = []

    def wrapper(dataset, dataset_attr, data_args, training_args):
        try:
            column_names = getattr(dataset, "column_names", None) or []
            if score_field in column_names:
                captured.append(np.asarray(dataset[score_field], dtype=np.float64))
            else:
                captured.append(None)
                logger.warning_rank0(
                    f"[Dataflex][Reorder] dataset '{dataset_attr}' has no field '{score_field}' "
                    f"(available: {list(column_names)})"
                )
        except Exception as exc:  # never let score capture break dataset loading
            captured.append(None)
            logger.warning_rank0(f"[Dataflex][Reorder] could not read '{score_field}': {exc}")
        return original(dataset, dataset_attr, data_args, training_args)

    lf_loader.align_dataset = wrapper
    try:
        yield captured
    finally:
        lf_loader.align_dataset = original


def _concat_raw_scores(captured: List[Optional[np.ndarray]], score_field: str) -> np.ndarray:
    if not captured or any(part is None for part in captured):
        raise ValueError(
            f"[Dataflex][Reorder] score field '{score_field}' is missing from at least one dataset. "
            f"Either add it to every source, or switch the reorder to "
            f"`apply_at: index` with an explicit `score_path`."
        )
    return np.concatenate(captured, axis=0)


def make_reorder_get_dataset(reorder_factory):
    """Build a `get_dataset` that permutes raw rows before tokenization.

    Why here and not in the trainer: the score lives in the raw JSONL and is
    deleted during preprocessing, and preprocessing is not index-preserving
    (malformed rows are dropped, packing merges rows). Permuting the raw dataset
    sidesteps the mapping entirely, because both `align_dataset` and
    `_get_preprocessed_dataset` preserve order, so whatever survives keeps its
    relative position.

    Args:
        reorder_factory: zero-arg callable returning a reorder exposing
            `order_rows(scores) -> permutation` and a `score_params` dict.
    """

    def reorder_get_dataset(
        template: "Template",
        model_args: "ModelArguments",
        data_args: "DataArguments",
        training_args: "Seq2SeqTrainingArguments",
        stage: Literal["pt", "sft", "rm", "ppo", "kto"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"] = None,
    ) -> "DatasetModule":
        if data_args.tokenized_path is not None and has_tokenized_data(data_args.tokenized_path):
            logger.warning_rank0(
                "[Dataflex][Reorder] loading an already tokenized dataset; its stored order is used as is. "
                "Use a distinct `tokenized_path` per ordering variant."
            )
            tokenized_data = load_from_disk(data_args.tokenized_path)
            dataset_module = get_dataset_module(tokenized_data)
            if data_args.streaming:
                dataset_module["train_dataset"] = dataset_module["train_dataset"].to_iterable_dataset()
            return dataset_module

        if data_args.streaming:
            raise ValueError("[Dataflex][Reorder] reordering requires `streaming: false`.")

        reorder = reorder_factory()
        score_field = reorder.score_params.get("score_field", "score")

        with training_args.main_process_first(desc="load dataset", local=(not data_args.data_shared_file_system)):
            with _capture_raw_scores(score_field) as captured:
                dataset = _get_merged_dataset(data_args.dataset, model_args, data_args, training_args, stage)

            eval_dataset = _get_merged_dataset(
                data_args.eval_dataset,
                model_args,
                data_args,
                training_args,
                stage,
                return_dict=data_args.eval_on_each_dataset,
            )

            if dataset is not None:
                score_path = reorder.score_params.get("score_path")
                if score_path:
                    from ..reorder.score_provider import PrecomputedScoreProvider

                    scores = np.asarray(
                        PrecomputedScoreProvider(
                            score_path=score_path, score_field=score_field, expected_size=len(dataset)
                        ).scores,
                        dtype=np.float64,
                    )
                else:
                    scores = _concat_raw_scores(captured, score_field)

                if len(scores) != len(dataset):
                    raise ValueError(
                        f"[Dataflex][Reorder] captured {len(scores)} scores but the merged dataset has "
                        f"{len(dataset)} rows. Check `mix_strategy` (use 'concat') and that every source "
                        f"carries '{score_field}'."
                    )

                permutation = reorder.order_rows(scores)
                dataset = dataset.select(permutation)
                logger.info_rank0(
                    f"[Dataflex][Reorder] applied '{reorder.pattern}' to {len(permutation)} raw rows "
                    f"before preprocessing."
                )

        with training_args.main_process_first(
            desc="pre-process dataset", local=(not data_args.data_shared_file_system)
        ):
            dataset = _get_preprocessed_dataset(
                dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=False
            )
            if isinstance(eval_dataset, dict):
                for eval_name, eval_data in eval_dataset.items():
                    eval_dataset[eval_name] = _get_preprocessed_dataset(
                        eval_data, data_args, training_args, stage, template, tokenizer, processor, is_eval=True
                    )
            else:
                eval_dataset = _get_preprocessed_dataset(
                    eval_dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=True
                )

            if data_args.val_size > 1e-6:
                logger.warning_rank0(
                    "[Dataflex][Reorder] `val_size > 0` splits with a shuffle, which destroys the ordering. "
                    "Use a separate `eval_dataset` instead."
                )

            dataset_dict = split_dataset(dataset, eval_dataset, data_args, seed=training_args.seed)
            if data_args.tokenized_path is not None and training_args.should_save:
                dataset_dict.save_to_disk(data_args.tokenized_path)
                logger.info_rank0(f"[Dataflex][Reorder] tokenized dataset saved at {data_args.tokenized_path}.")

            dataset_module = get_dataset_module(dataset_dict)

        train_size = len(dataset_module["train_dataset"]) if dataset_module.get("train_dataset") is not None else 0
        logger.info_rank0(f"[Dataflex][Reorder] ordered training set ready: {train_size} samples.")
        return dataset_module

    return reorder_get_dataset


# ======================================================================
# Lego
# ======================================================================


class DomainLabeledDataset(TorchDataset):
    """A single dataset that still knows which domain each row came from.

    The mixture path represents a domain as its own dataset object, which forces
    `train_dataset = None` and leaves selectors and reorders with nothing to
    index into. Composition needs the opposite arrangement: one concatenated
    dataset plus a label per row, so that "allocate a quota per domain" is index
    arithmetic over groups and every other family keeps working unchanged.

    `domain_id` is injected per item because DoReMi reads it in `compute_loss`
    and ODM reads it to attribute its bandit reward.
    """

    def __init__(self, dataset, domain_ids):
        self.dataset = dataset
        self.domain_ids = np.asarray(domain_ids, dtype=np.int64)
        if len(self.domain_ids) != len(dataset):
            raise ValueError(
                f"domain_ids has {len(self.domain_ids)} entries but the dataset has {len(dataset)} rows"
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if isinstance(item, dict):
            return {**item, "domain_id": int(self.domain_ids[idx])}
        return item

    @property
    def column_names(self):
        # Some LlamaFactory paths sniff this to decide whether to prune columns.
        names = getattr(self.dataset, "column_names", None)
        return (list(names) + ["domain_id"]) if names else None


def lego_get_dataset(
    template: "Template",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
) -> "DatasetModule":
    r"""Build one concatenated training set plus a per-sample domain label.

    Reuses LlamaFactory's own loading and preprocessing, then concatenates the
    per-source datasets in the order they appear in `dataset:` so that domain
    index order matches `init_mixture_proportions`. Unlike the mixture path this
    returns a real `train_dataset`, samples nothing, and shuffles nothing —
    all of that is the pipeline's job.
    """
    if data_args.streaming:
        raise ValueError("[Dataflex][Lego] composition requires `streaming: false`.")

    with training_args.main_process_first(desc="load dataset", local=(not data_args.data_shared_file_system)):
        per_source_raw = _get_merged_dataset(
            data_args.dataset, model_args, data_args, training_args, stage, return_dict=True
        )
        eval_dataset = _get_merged_dataset(
            data_args.eval_dataset,
            model_args,
            data_args,
            training_args,
            stage,
            return_dict=data_args.eval_on_each_dataset,
        )

    with training_args.main_process_first(desc="pre-process dataset", local=(not data_args.data_shared_file_system)):
        per_source_pp = {
            name: _get_preprocessed_dataset(
                ds, data_args, training_args, stage, template, tokenizer, processor, is_eval=False
            )
            for name, ds in (per_source_raw or {}).items()
        }

        if isinstance(eval_dataset, dict):
            for eval_name, eval_data in eval_dataset.items():
                eval_dataset[eval_name] = _get_preprocessed_dataset(
                    eval_data, data_args, training_args, stage, template, tokenizer, processor, is_eval=True
                )
        else:
            eval_dataset = _get_preprocessed_dataset(
                eval_dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=True
            )

        domain_names = list(per_source_pp.keys())
        sizes = [len(per_source_pp[name]) for name in domain_names]
        domain_ids = np.concatenate(
            [np.full(n, i, dtype=np.int64) for i, n in enumerate(sizes)]
        ) if sizes else np.zeros(0, dtype=np.int64)

        if len(domain_names) == 1:
            train_dataset = per_source_pp[domain_names[0]]
        else:
            train_dataset = concatenate_datasets([per_source_pp[name] for name in domain_names])

        if data_args.val_size > 1e-6:
            raise ValueError(
                "[Dataflex][Lego] `val_size > 0` splits with a shuffle, which breaks the alignment "
                "between domain_ids and train_dataset. Use a separate `eval_dataset`."
            )

        dataset_dict = split_dataset(train_dataset, eval_dataset, data_args, seed=training_args.seed)
        dataset_module = get_dataset_module(dataset_dict)

    plan = list(zip(domain_names, sizes))
    logger.info_rank0(
        f"[Dataflex][Lego] domains in `dataset:` order (name, rows): {plan}; total={sum(sizes)}"
    )

    # Only wrap when there is something to attribute; a single-domain run does
    # not need the per-item dict copy.
    if len(domain_names) > 1:
        dataset_module["train_dataset"] = DomainLabeledDataset(dataset_module["train_dataset"], domain_ids)
        logger.info_rank0("[Dataflex][Lego] train_dataset wrapped to carry domain_id per sample.")

    dataset_module["domain_ids"] = domain_ids
    dataset_module["domain_names"] = domain_names
    return dataset_module
