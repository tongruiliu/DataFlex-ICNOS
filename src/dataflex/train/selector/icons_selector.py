import glob
import os
from typing import Dict, List, Optional, Sequence

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataflex.core.registry import register_selector
from dataflex.utils.logging import logger
from dataflex.utils.selector_io import load_cached_selection, save_selection

from .base_selector import Selector
from .less_selector import _trak_projectors


class IndexedDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, index):
        return index, self.original_dataset[index]


@register_selector("icons")
class IconsSelector(Selector):
    """
    Influence CONsensus data selection (arXiv:2501.00654).

    ICONS scores every training sample against K task validation sets via
    gradient influence, then aggregates the K per-task rankings by majority
    voting instead of by comparing raw scores across tasks. It shares the
    gradient-influence machinery with LESS (projected, normalized gradients;
    cosine influence) but is a distinct method: the consensus vote, not a
    single validation target, is what selects the subset.

    The K tasks are read off a single `eval_dataset` whose rows are the task
    validation sets concatenated in order. `task_boundaries` gives the size of
    each task's slice (summing to len(eval_dataset)); when it is None the whole
    eval set is one task and selection degenerates to plain LESS.
    """

    def __init__(self,
                 dataset,
                 eval_dataset,
                 accelerator,
                 data_collator,
                 cache_dir,
                 gradient_type: str = "adam",
                 proj_dim: int = 8192,
                 save_interval: int = 16,
                 seed: int = 42,
                 task_boundaries: Optional[Sequence[int]] = None):
        super().__init__(dataset, accelerator, data_collator, cache_dir)

        self.eval_dataset = eval_dataset
        self.gradient_type = gradient_type
        self.proj_dim = proj_dim
        self.save_interval = save_interval
        self.seed = seed

        self.device = self.accelerator.device
        self.dtype = torch.float16

        self.task_slices = self._resolve_task_slices(task_boundaries, len(eval_dataset))

        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"IconsSelector initialized with {len(self.task_slices)} task(s). "
                    f"Projected gradients will be saved in {self.cache_dir}")

    def _resolve_task_slices(self, task_boundaries, n_eval) -> List[slice]:
        if not task_boundaries:
            return [slice(0, n_eval)]
        if sum(task_boundaries) != n_eval:
            raise ValueError(
                f"task_boundaries sum to {sum(task_boundaries)} but eval_dataset has {n_eval} samples"
            )
        slices, start = [], 0
        for size in task_boundaries:
            slices.append(slice(start, start + size))
            start += size
        return slices

    # ------------------------------------------------------------------
    # Gradient influence (shared design with LESS)
    # ------------------------------------------------------------------

    def _get_number_of_params(self, model) -> int:
        num_params = 0
        for p in model.parameters():
            if p.requires_grad:
                # ZeRO-3 partitions params; ds_numel is the full size that
                # safe_get_full_grad returns, p.numel() is only the local shard.
                num_params += p.ds_numel if hasattr(p, 'ds_numel') else p.numel()
        if self.accelerator.is_main_process:
            logger.info(f"Total number of parameters that require gradients: {num_params}")
        return num_params

    def _prepare_optimizer_state(self, model, optimizer_state: Optional[Dict] = None):
        avg_list, avg_sq_list = [], []
        if self.accelerator.state.deepspeed_plugin is not None:
            from deepspeed.utils import safe_get_full_optimizer_state
            for param in model.parameters():
                if param.requires_grad:
                    exp_avg = safe_get_full_optimizer_state(param, "exp_avg")
                    exp_avg_sq = safe_get_full_optimizer_state(param, "exp_avg_sq")
                    if exp_avg is not None and exp_avg_sq is not None:
                        avg_list.append(exp_avg.view(-1))
                        avg_sq_list.append(exp_avg_sq.view(-1))
        else:
            if optimizer_state is None:
                raise ValueError("optimizer_state must be provided for non-DeepSpeed 'adam' gradient type.")
            for param in model.parameters():
                if param.requires_grad:
                    avg_list.append(optimizer_state[param]["exp_avg"].view(-1))
                    avg_sq_list.append(optimizer_state[param]["exp_avg_sq"].view(-1))
        avg = torch.cat(avg_list).to(self.device)
        avg_sq = torch.cat(avg_sq_list).to(self.device)
        return avg, avg_sq

    def _obtain_gradients(self, model, batch, gradient_type, m=None, v=None) -> torch.Tensor:
        if self.accelerator.state.deepspeed_plugin is not None:
            loss = model(**batch).loss
            model.backward(loss)
            from deepspeed.utils import safe_get_full_grad
            grads = []
            for _, p in model.named_parameters():
                g = safe_get_full_grad(p)
                if g is not None:
                    grads.append(g.contiguous().view(-1))
            vectorized_grads = torch.cat(grads) if grads else None
        else:
            with self.accelerator.no_sync(model):
                loss = model(**batch).loss
                self.accelerator.backward(loss)
            vectorized_grads = torch.cat(
                [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
            )

        if gradient_type == "adam":
            if m is None or v is None:
                raise ValueError("Adam optimizer states (m, v) must be provided for 'adam' gradient type.")
            beta1, beta2, eps = 0.9, 0.999, 1e-08
            denom = v.mul(beta2)
            denom.addcmul_(vectorized_grads, vectorized_grads, value=(1 - beta2))
            denom.sqrt_().add_(eps)
            vectorized_grads.mul_(1 - beta1).add_(m, alpha=beta1)
            vectorized_grads.div_(denom)
            del denom
        elif gradient_type != "sgd":
            assert False, f"Unknown gradient type: {gradient_type}"

        model.zero_grad()
        return vectorized_grads

    def _get_trak_projector(self):
        BasicProjector, CudaProjector, _ = _trak_projectors()
        try:
            import fast_jl
            num_sms = torch.cuda.get_device_properties(self.device.index).multi_processor_count
            fast_jl.project_rademacher_8(torch.zeros(8, 1_000, device=self.device), 512, 0, num_sms)
            projector = CudaProjector
            if self.accelerator.is_main_process:
                logger.info("Using CudaProjector for gradient projection.")
        except (ImportError, RuntimeError):
            projector = BasicProjector
            if self.accelerator.is_main_process:
                logger.info("CudaProjector not available. Using BasicProjector for gradient projection.")
        return projector

    def _get_max_saved_index(self, save_dir) -> int:
        if not os.path.exists(save_dir) or not self.accelerator.is_main_process:
            return -1
        files = [f for f in os.listdir(save_dir) if f.startswith("grads") and f.endswith(".pt")]
        if not files:
            return -1
        # filename: grads-{count}-rank{rank}.pt
        return max(int(f.split('.')[0].split('-')[1]) for f in files)

    def _collect_and_save_projected_gradients(self, model, save_dir, dataset_to_use, gradient_type, optimizer_state=None):
        """Each process projects its shard of per-sample gradients and saves indexed chunks."""
        num_params = self._get_number_of_params(model)
        projector_class = self._get_trak_projector()
        _, _, ProjectionType = _trak_projectors()
        projector = projector_class(
            grad_dim=num_params,
            proj_dim=self.proj_dim,
            seed=self.seed,
            proj_type=ProjectionType.rademacher,
            max_batch_size=8,
            block_size=128,
            device=self.device,
            dtype=self.dtype,
        )

        m, v = None, None
        if gradient_type == "adam":
            if self.accelerator.state.deepspeed_plugin is None and optimizer_state is None:
                raise ValueError("optimizer_state must be provided for non-DeepSpeed 'adam' gradient type.")
            m, v = self._prepare_optimizer_state(model, optimizer_state)

        indexed_dataset = IndexedDataset(dataset_to_use)

        def indexed_collator_wrapper(features):
            indices = [f[0] for f in features]
            original_data = [f[1] for f in features]
            return {'indices': torch.tensor(indices), 'batch': self.data_collator(original_data)}

        dataloader = DataLoader(
            indexed_dataset, batch_size=1, shuffle=False, num_workers=2,
            collate_fn=indexed_collator_wrapper,
        )
        dataloader = self.accelerator.prepare(dataloader)

        save_interval = self.save_interval
        start_count = self._get_max_saved_index(save_dir) + 1
        if self.accelerator.is_main_process and start_count > 1:
            logger.info(f"Resuming from sample index {start_count}.")
        self.accelerator.wait_for_everyone()

        total_samples_in_loader = len(dataloader)
        model_device = next(model.parameters()).device
        grad_buffer = torch.zeros(save_interval, num_params, device=model_device, dtype=self.dtype)
        idx_buffer = torch.zeros(save_interval, dtype=torch.long)
        buf_pos = 0

        for batch_idx, data in enumerate(tqdm(
            dataloader,
            desc=f"[Process {self.accelerator.process_index}] Calculating Gradients",
            disable=not self.accelerator.is_local_main_process,
            dynamic_ncols=True,
            position=self.accelerator.process_index,
        ), 1):
            vectorized_grads = self._obtain_gradients(model, data['batch'], gradient_type, m, v)
            grad_buffer[buf_pos].copy_(vectorized_grads)
            del vectorized_grads
            idx_buffer[buf_pos] = data['indices'][0]
            buf_pos += 1

            if buf_pos == save_interval or batch_idx == total_samples_in_loader:
                projected = projector.project(grad_buffer[:buf_pos], model_id=0).cpu()
                save_path = os.path.join(
                    save_dir, f"grads-{idx_buffer[:buf_pos].max().item()}-rank{self.accelerator.process_index}.pt",
                )
                torch.save({'grads': projected, 'indices': idx_buffer[:buf_pos].clone()}, save_path)
                del projected
                buf_pos = 0

        del grad_buffer, idx_buffer
        self.accelerator.wait_for_everyone()

    def _merge_and_normalize_info(self, save_dir, total_samples):
        """Main process reorders chunks by original index, then L2-normalizes per row."""
        if not self.accelerator.is_main_process:
            return
        files = glob.glob(os.path.join(save_dir, "grads-*-rank*.pt"))
        if not files:
            logger.warning("No gradient files found to merge.")
            return

        final_grads = torch.zeros(total_samples, self.proj_dim, dtype=torch.float32)
        for file_path in tqdm(files, desc="Merging files"):
            chunk = torch.load(file_path, map_location="cpu")
            final_grads[chunk['indices']] = chunk['grads'].to(torch.float32)

        norms = final_grads.norm(dim=1, keepdim=True).clamp_(min=1e-12)
        final_grads.div_(norms)
        output_file = os.path.join(save_dir, "all_projected_grads.pt")
        torch.save(final_grads, output_file)
        logger.info(f"Saved merged and normalized gradients (Shape: {final_grads.shape}) to {output_file}")

        for file_path in files:
            os.remove(file_path)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(self, model, step_id: int, num_samples: int, **kwargs) -> List[int]:
        os.makedirs(self.cache_dir, exist_ok=True)
        save_path = os.path.join(self.cache_dir, f"step_{step_id}.json")
        if os.path.exists(save_path):
            if self.accelerator.is_main_process:
                cached_indices, _ = load_cached_selection(save_path)
            else:
                cached_indices = None
            cached_indices_list = [cached_indices]
            if dist.is_available() and dist.is_initialized():
                dist.broadcast_object_list(cached_indices_list, src=0)
                cached_indices = cached_indices_list[0]
            else:
                cached_indices = cached_indices or []
            return cached_indices

        now_train_save_dir = os.path.join(self.cache_dir, "train", str(step_id))
        now_eval_save_dir = os.path.join(self.cache_dir, "eval", str(step_id))
        self.step_id = step_id
        train_final_grads_path = os.path.join(now_train_save_dir, "all_projected_grads.pt")
        eval_final_grads_path = os.path.join(now_eval_save_dir, "all_projected_grads.pt")

        # All K task validation sets are concatenated in eval_dataset, so their
        # gradients are projected in a single pass; the tasks are split apart
        # only at scoring time via self.task_slices.
        if not os.path.exists(train_final_grads_path):
            os.makedirs(now_train_save_dir, exist_ok=True)
            optimizer_state = kwargs.get('optimizer_state', None)
            self._collect_and_save_projected_gradients(model, now_train_save_dir, self.dataset, self.gradient_type, optimizer_state)
            self._merge_and_normalize_info(now_train_save_dir, len(self.dataset))
        self.accelerator.wait_for_everyone()

        if not os.path.exists(eval_final_grads_path):
            os.makedirs(now_eval_save_dir, exist_ok=True)
            self._collect_and_save_projected_gradients(model, now_eval_save_dir, self.eval_dataset, "sgd", None)
            self._merge_and_normalize_info(now_eval_save_dir, len(self.eval_dataset))
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            train_projected_grads = torch.load(train_final_grads_path, map_location="cpu")
            eval_projected_grads = torch.load(eval_final_grads_path, map_location="cpu")

            selected_indices, votes = self._vote_select(train_projected_grads, eval_projected_grads, num_samples)
            logger.info(f"Selecting top {num_samples} samples by influence consensus over {len(self.task_slices)} tasks.")

            metric_payload = {"icons_votes": [int(votes[i].item()) for i in selected_indices]}
            save_selection(save_path, selected_indices, metric_payload, self.accelerator)
        else:
            selected_indices = None

        obj_list = [selected_indices]
        if dist.is_initialized():
            dist.broadcast_object_list(obj_list, src=0)
        selected_indices = obj_list[0]

        return selected_indices

    def _vote_select(self, train_grads, eval_grads, num_samples):
        """Per-task influence -> per-task threshold vote -> top by total votes.

        For each task, a sample votes if its mean influence on that task ranks
        in the top p = num_samples / N. Voting on the binary "above threshold"
        signal, not on the raw score, is what removes the need to calibrate
        influence magnitudes across tasks (paper Eqn. 6, Alg. 2).
        """
        n_train = train_grads.shape[0]
        p = num_samples / n_train
        votes = torch.zeros(n_train, dtype=torch.long)

        for sl in self.task_slices:
            task_influence = (train_grads @ eval_grads[sl].T).mean(dim=1)
            tau = torch.quantile(task_influence, 1.0 - p)  # (1 - p) quantile = top-p cutoff for this task
            votes += (task_influence >= tau).long()

        selected_indices = torch.topk(votes, k=num_samples, largest=True).indices.tolist()
        return selected_indices, votes
