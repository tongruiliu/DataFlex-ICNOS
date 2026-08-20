import json
import os
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple
from dataflex.utils.logging import logger

def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
        
def load_cached_selection(
    save_path: str
) -> Tuple[Optional[List[int]], Optional[Dict[str, List]]]:
    indices = None
    metric = None
    with open(save_path, "r") as f:
        payload = json.load(f)
    indices = payload.get("indices", [])
    metric = payload.get("metric", {})

    logger.info(f"[Dataflex] Loaded cached selection from {save_path}: {indices is not None}.")
    return indices, metric

def save_selection(
    save_path: str,
    indices: List[int],
    metric: Dict[str, List],
    accelerator,
) -> None:
    """
    Save in a unified format and only by the main process.
    Stored as standard JSON format.
    """
    if accelerator.is_main_process:
        _ensure_parent_dir(save_path)
        payload = {
            "indices": list(map(int, indices)),
            "metric": metric,
        }
        with open(save_path, "w") as f:
            json.dump(payload, f, indent=4)  # Save in a pretty JSON format
        logger.info(f"[Dataflex] Saved selection to {save_path}.")
