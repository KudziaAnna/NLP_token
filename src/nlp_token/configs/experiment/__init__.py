from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from omegaconf.omegaconf import MISSING


# Experiment settings validation schema & default values
@dataclass
class ExperimentSettings:
    # ----------------------------------------------------------------------------------------------
    # General experiment settings
    # ----------------------------------------------------------------------------------------------
    # wandb tags
    _tags_: Optional[List[str]] = None

    # Seed for all random number generators
    seed: int = 1

    # Path to resume from. Two formats are supported:
    # - local checkpoints: path to checkpoint relative from run (results) directory
    # - wandb artifacts: wandb://ARTIFACT_PATH/ARTIFACT_NAME:VERSION@CHECKPOINT_NAME
    resume_checkpoint: Optional[str] = None

    # Enable checkpoint saving
    save_checkpoints: bool = True

    # Enable initial validation before training
    validate_before_training: bool = True

    model: Any = MISSING

    # ----------------------------------------------------------------------------------------------
    # Data loading settings
    # ----------------------------------------------------------------------------------------------
    # Training batch size
    batch_size: int = 2

    # Number of folds
    num_workers: int = 8

    data_dir: Any = MISSING

    # ----------------------------------------------------------------------------------------------
    # Dataset specific settings
    # ----------------------------------------------------------------------------------------------
    # PyTorch Lightning datamodule class
    # e.g.: `pl_bolts.datamodules.binary_mnist_datamodule.BinaryMNISTDataModule`

    embedding_dim: int = 50
    input_size: int = 684
    #output_size: int = 1026
    output_size: int = 2347

    # RTransformer parameters to explore d_model, h, ksize, n_level, n
    model_spec: Tuple[int, int, int, int] = (2, 7, 8, 2)

    # RNN type
    rnn_type: str = 'GRU'

    # Dropout values
    dropout: Tuple[float, float] = (0.05, 0.02)

    #  GRU number of layers and hidden layer size
    hidden_size: int = 64
    n_layers: int = 1

    dict_size: int = 2347