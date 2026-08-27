from abc import ABC, abstractmethod
from typing import Any, Union
from torch import nn
import torch


class Weighter(ABC):
    """
    Abstract base class for data weighting, defining the basic interface and common functionality.
    """
    
    def __init__(self, **kwargs):
        """
        Base class constructor
        
        Args:
            **kwargs: Subclass-specific parameters
        """
        # Subclasses can define common initialization logic here
        pass
    
    def _per_sample_loss_from_logits(self, logits, labels, ignore_index: int = -100):
        """
        Calculate the per-sample loss from logits and labels
        
        Args:
            logits: Model output logits
            labels: True labels
            ignore_index: Ignored label index
            
        Returns:
            torch.Tensor: The per-sample loss (B,)
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        num_active = (shift_labels != ignore_index).sum(dim=1)  # (B,)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        tok_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1).long()
        )
        return tok_loss.view(shift_logits.size(0), -1).sum(dim=1) / torch.clamp(num_active, min=1)
    
    @abstractmethod
    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        *,
        ctx: Any = None,
        model: nn.Module | None = None,
        inputs: dict[str, Union[torch.Tensor, Any]] | None = None,
    ) -> torch.Tensor:
        """
        Core weighting method, subclasses must implement this method
        
        Args:
            losses: Per-sample loss on this card (B,)
            ctx: Trainer context, can get global_step information
            model: Current model
            inputs: Input data
            
        Returns:
            torch.Tensor: The weighted total loss (scalar)
        """
        pass
    
    def training_step(self, ctx, model, inputs, num_items_in_batch=None, use_weighter=False):
        """
        Execute the training step, including forward propagation, loss calculation, weighting, and backpropagation
        
        Args:
            ctx: Trainer context
            model: Model
            inputs: Input data
            num_items_in_batch: Number of samples in the batch
            use_weighter: Whether to use the weighter
            
        Returns:
            The loss value for this step
        """
        from dataflex.utils.logging import logger
        from transformers.utils import is_apex_available
        from accelerate.utils import DistributedType
        
        model.train()
        if hasattr(ctx.optimizer, "train") and callable(ctx.optimizer.train):
            ctx.optimizer.train()

        inputs = ctx._prepare_inputs(inputs)

        # Save a copy of labels (to prevent them from being popped in some implementations)
        labels_for_weighter = inputs.get("labels", None)

        with ctx.compute_loss_context_manager():
            # Key: get the outputs
            loss, outputs = ctx.compute_loss(
                model, inputs, num_items_in_batch=num_items_in_batch, return_outputs=True
            )

        # Whether the model's scalar loss really got replaced by our own per-sample one.
        # If it did, the model's per-token normalization no longer applies and this
        # method has to do the gradient accumulation scaling itself.
        reweighted = False

        if use_weighter:
            # 1) If compute_loss already returned a (B,) vector, use it directly
            if torch.is_tensor(loss) and loss.dim() == 1:
                per_sample = loss
            else:
                # 2) Otherwise, calculate the per-sample loss from logits and labels (no need for second forward pass)
                logits = getattr(outputs, "logits", None) if outputs is not None else None
                labels = inputs.get("labels", None)
                if labels is None:
                    labels = labels_for_weighter
                per_sample = None
                if logits is not None and labels is not None:
                    per_sample = self._per_sample_loss_from_logits(logits, labels)

            if per_sample is not None:
                # Log only from the main process
                if ctx.args.local_rank in [-1, 0]:
                    ps = per_sample.detach().float().cpu().view(-1)[0]
                    logger.info(f"[Dataflex] Before weighting per-sample (first sample): {ps}")
                # Distributed weighting
                loss = self.get_weighted_loss(per_sample, ctx=ctx, model=model, inputs=inputs)
                reweighted = True
                if ctx.args.local_rank in [-1, 0]:
                    logger.info(f"[Dataflex] After weighting (first sample): {float(loss.detach().cpu())}")
            else:
                if ctx.args.local_rank in [-1, 0]:
                    logger.info("[Dataflex] Could not form per-sample losses; fallback to scalar loss (no reweight).")

        del inputs

        if ctx.args.torch_empty_cache_steps is not None and ctx.state.global_step % ctx.args.torch_empty_cache_steps == 0:
            ctx._empty_cache()

        kwargs = {}
        if ctx.args.n_gpu > 1:
            loss = loss.mean()

        if getattr(ctx, "use_apex", False):
            if is_apex_available():
                from apex import amp
                with amp.scale_loss(loss, ctx.optimizer) as scaled_loss:
                    scaled_loss.backward()
        else:
            # Gradient accumulation scaling, aligned with upstream Trainer.training_step.
            #
            # Upstream's rule: skip the division when the model already normalized by
            # num_items_in_batch, otherwise divide by the group's *actual* micro-batch
            # count. This used to divide unconditionally, and by the static
            # args.gradient_accumulation_steps, so an already-normalized loss got
            # divided twice and a short final group got divided too much.
            #
            # Reweighting is a third case: the per-sample loss is recomputed here from
            # logits, so the model's normalization does not apply and we must scale.
            if reweighted or (
                (not getattr(ctx, "model_accepts_loss_kwargs", False) or num_items_in_batch is None)
                and getattr(ctx, "compute_loss_func", None) is None
            ):
                loss = loss / getattr(
                    ctx, "current_gradient_accumulation_steps", ctx.args.gradient_accumulation_steps
                )
            if ctx.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False
            ctx.accelerator.backward(loss, **kwargs)

        return loss.detach()
