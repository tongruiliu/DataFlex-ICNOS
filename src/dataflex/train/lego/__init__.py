"""Lego: chain several data-centric strategies in one training run.

Importing this module also imports the four component families so their registry entries exist
before a pipeline tries to build one by name.
"""

import dataflex.train.mixer  # noqa: F401
import dataflex.train.reorder  # noqa: F401
import dataflex.train.selector  # noqa: F401
import dataflex.train.weighter  # noqa: F401

from .context import PlanContext
from .domain_view import DomainView
from .pipeline import SchedulePipeline
from .plan import DataPlan
from .scoreboard import ScoreBoard, per_sample_loss_from_outputs
from .stages import (
    INDEX_ORDER,
    STAGE_TYPES,
    DomainWeightStage,
    IndexStage,
    LossStage,
    MixStage,
    ReorderStage,
    SelectStage,
    Stage,
    WeightStage,
)

__all__ = [
    "DataPlan",
    "PlanContext",
    "DomainView",
    "ScoreBoard",
    "per_sample_loss_from_outputs",
    "SchedulePipeline",
    "Stage",
    "IndexStage",
    "LossStage",
    "SelectStage",
    "MixStage",
    "ReorderStage",
    "WeightStage",
    "DomainWeightStage",
    "INDEX_ORDER",
    "STAGE_TYPES",
]
