"""olive_oil — CAC2026 olive oil musty-defect classification toolkit."""

from .data import load_dataset, get_spectral_matrix, average_replicates
from .blocks import BlockConfig, PreparedData, prepare_blocks, BLOCK_KEYS
from .models import (
    BlockSet,
    PLSScorer,
    PLSDAClassifier,
    MidLevelFusionTransformer,
    MidLevelFusionClassifier,
)
from .evaluation import nested_cv, tune_final_model, predict_test, make_cv

__all__ = [
    "load_dataset",
    "get_spectral_matrix",
    "average_replicates",
    "BlockConfig",
    "PreparedData",
    "prepare_blocks",
    "BLOCK_KEYS",
    "BlockSet",
    "PLSScorer",
    "PLSDAClassifier",
    "MidLevelFusionTransformer",
    "MidLevelFusionClassifier",
    "nested_cv",
    "tune_final_model",
    "predict_test",
    "make_cv",
]
