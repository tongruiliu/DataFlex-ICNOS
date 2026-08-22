"""Per-sample loss, in one place.

Three call sites needed this independently — the weighters, the reorder score
providers and the loss-based selectors — and a fourth (the compose ScoreBoard)
would have made four. Keeping one implementation also keeps one definition of
what "the loss of a sample" means: token-mean cross-entropy, so a long sample is
not scored higher merely for having more tokens.
"""

from typing import Any, Dict

import torch

IGNORE_INDEX = -100

#: The score of a sample whose loss is unknown, either because nothing scored it
#: or because it has no trainable token to score. NaN rather than an infinity for
#: two reasons. It propagates: the difference of two scores stays unknown, where
#: `inf - x` would come out as `+inf` and read as an enormous improvement. And
#: every comparison against it is False, so an unscored sample cannot fall into a
#: quantile bucket or a top-k even if a caller forgets to mask it out. Consumers
#: filter it with `torch.isfinite` or collapse it with `torch.nan_to_num`.
UNSCORED = float("nan")


def per_sample_loss_from_logits(logits, labels, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """Token-mean cross-entropy for each row of the batch. Detached.

    Logits are upcast to float32 first, which is what HuggingFace's own causal-LM
    loss does. It is not optional under bf16: a loss near 1.28 lands on a bf16
    grid roughly 0.008 wide, so samples whose true losses differ by less than
    that become tied and any ranking built on the value degrades. Long sequences
    could also overflow to NaN.
    """
    shift_logits = logits[..., :-1, :].float().contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=ignore_index)
    tok_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1).long(),
    ).view(shift_labels.size(0), -1)
    active = (shift_labels != ignore_index).sum(dim=1)
    per_sample = tok_loss.sum(dim=1) / torch.clamp(active, min=1)
    # A sample with no trainable token has no defined loss, which is also what
    # HuggingFace's mean reduction reports there. Returning 0.0 instead would
    # make it look like the easiest data in the pool.
    return per_sample.masked_fill(active == 0, UNSCORED).detach()


def per_sample_loss_from_outputs(outputs, inputs: Dict[str, Any]) -> torch.Tensor:
    """Per-sample loss from a forward pass.

    Falls back to broadcasting the model's scalar loss when logits or labels are
    unavailable. Note that `outputs.loss` is a *batch mean*, so that fallback is
    only meaningful at batch size 1 — which is exactly why anything that batches
    has to go through the logits path.
    """
    logits = getattr(outputs, "logits", None)
    labels = inputs.get("labels", None)
    if logits is None or labels is None:
        loss = getattr(outputs, "loss", None)
        if loss is None:
            raise ValueError("model returned neither logits+labels nor a loss")
        batch = next(iter(inputs.values())).size(0)
        return loss.detach().view(1).expand(batch).clone()
    return per_sample_loss_from_logits(logits, labels)
