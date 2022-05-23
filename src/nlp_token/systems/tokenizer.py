from __future__ import annotations

import random
from typing import Any, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from pytorch_lightning.loggers.base import LoggerCollection
from pytorch_lightning.loggers.wandb import WandbLogger
from torchtext.data.metrics import bleu_score
from torchmetrics import BLEUScore
from rich import print
import pytorch_lightning as pl
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from wandb.sdk.wandb_run import Run

from ..configs import Config
from ..models.rtransformer import RT
from ..models.gru_model import GRUBasedSeq2Seq
from ..datamodules.token_datamodule import prepare_dict, get_key


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


class TextTokenizer(pl.LightningModule):
    """
    Basic text tokenizer.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()  # type: ignore

        self.logger: Union[LoggerCollection, WandbLogger, Any]
        self.wandb: Run

        self.cfg = cfg

        if self.cfg.experiment.model == 'RT':
            self.model = RT(self.cfg)
        elif self.cfg.experiment.model == 'GRU':
            self.model = GRUBasedSeq2Seq(cfg)
        elif self.cfg.experiment.model == 'LSTM':
            pass
        else:
            raise Exception("No such model name")
        self.criterion = nn.CrossEntropyLoss(ignore_index = 0)

        self.words_dict, _ = prepare_dict(self.cfg.experiment.data_dir)
        self.get_word = get_key

        # Metrics
        self.metric = BLEUScore(n_gram=1)
        self.train_bleu = []
        self.val_bleu = []

        self.model.apply(init_weights)

    # -----------------------------------------------------------------------------------------------
    # Default PyTorch Lightning hooks
    # -----------------------------------------------------------------------------------------------
    def on_fit_start(self) -> None:
        """
        Hook before `trainer.fit()`.
        Attaches current wandb run to `self.wandb`.
        """
        if isinstance(self.logger, LoggerCollection):
            for logger in self.logger:  # type: ignore
                if isinstance(logger, WandbLogger):
                    self.wandb = logger.experiment  # type: ignore
        elif isinstance(self.logger, WandbLogger):
            self.wandb = self.logger.experiment  # type: ignore

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """
        Hook on checkpoint saving.
        Adds config and RNG states to the checkpoint file.
        """
        checkpoint['cfg'] = self.cfg
        checkpoint['rng_torch'] = torch.default_generator.get_state()
        checkpoint['rng_numpy'] = np.random.get_state()
        checkpoint['rng_random'] = random.getstate()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """
        Hook on checkpoint loading.
        Loads RNG states from the checkpoint file.
        """
        torch.default_generator.set_state(checkpoint['rng_torch'])
        np.random.set_state(checkpoint['rng_numpy'])
        random.setstate(checkpoint['rng_random'])

    # ----------------------------------------------------------------------------------------------
    # Optimizers
    # ----------------------------------------------------------------------------------------------
    def configure_optimizers(self) -> Union[Optimizer, Tuple[List[Optimizer], List[_LRScheduler]]]:  # type: ignore
        """
        Define system optimization procedure.
        See https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers.
        Returns
        -------
        Union[Optimizer, Tuple[List[Optimizer], List[_LRScheduler]]]
            Single optimizer or a combination of optimizers with learning rate schedulers.
        """
        optimizer: Optimizer = instantiate(
            self.cfg.optim.optimizer,
            params=self.parameters(),
            _convert_='all'
        )

        if self.cfg.optim.scheduler is not None:
            scheduler: _LRScheduler = instantiate(  # type: ignore
                self.cfg.optim.scheduler,
                optimizer=optimizer,
                _convert_='all'
            )
            print(optimizer, scheduler)
            return [optimizer], [scheduler]
        else:
            print(optimizer)
            return optimizer

    # ----------------------------------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor, y:torch.Tensor) -> torch.Tensor:  # type: ignore
        """
        Forward pass of the whole system.
        In this simple case just calls the main model.
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        y: torch.Tensor
            Target tensor.
        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        return self.model(x, y)

    # ----------------------------------------------------------------------------------------------
    # Loss
    # ----------------------------------------------------------------------------------------------
    def calculate_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss value of a batch.
        In this simple case just forwards computation to default `self.criterion`.
        Parameters
        ----------
        outputs : torch.Tensor
            Network outputs with shape (batch_size, n_classes).
        targets : torch.Tensor
            Targets (ground-truth labels) with shape (batch_size).
        Returns
        -------
        torch.Tensor
            Loss value.
        """
        return self.criterion(outputs, targets)

    # ----------------------------------------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------------------------------------
    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor]:  # type: ignore
        """
        Train on a single batch with loss defined by `self.criterion`.
        Parameters
        ----------
        batch : list[torch.Tensor]
            Training batch.
        batch_idx : int
            Batch index.
        Returns
        -------
        dict[str, torch.Tensor]
            Metric values for a given batch.
        """
        inputs, targets = batch["text"], batch["label"]
        inputs = inputs.reshape(inputs.shape[0], inputs.shape[2]).T
        targets = targets.reshape(targets.shape[0], targets.shape[2]).T
        outputs = self(inputs, targets)  # basically equivalent to self.forward(data)

        output_dim = outputs.shape[-1]
        
        to_loss_output = outputs[1:].reshape(-1, output_dim)
        to_loss_target = targets[1:].reshape(-1)

        loss = self.calculate_loss(to_loss_output, to_loss_target)

        outputs = torch.argmax(outputs, dim=2)

        for batch in range(outputs.shape[1]):
            tmp_outputs = []
            tmp_targets = []
            for i in range(outputs.shape[0]):
                tmp_outputs.append(self.get_word(self.words_dict, outputs[i, batch]))
                tmp_targets.append(self.get_word(self.words_dict, targets[i, batch]))
                
            self.train_bleu.append(self.metric(tmp_targets, tmp_targets))

        return {
            'loss': loss,
            # no need to return 'train_acc' here since it is always available as `self.train_acc`
        }

    def training_epoch_end(self, outputs: list[Any]) -> None:
        """
        Log training metrics.
        Parameters
        ----------
        outputs : list[Any]
            List of dictionaries returned by `self.training_step` with batch metrics.
        """
        step = self.current_epoch + 1

        metrics = {
            'epoch': float(step),
            'train_bleu': float(np.average(self.train_bleu)),
        }

        print("Bleu score: ")
        print(np.average(self.train_bleu))
        # Average additional metrics over all batches
        for key in outputs[0]:
            metrics[key] = float(self._reduce(outputs, key).item())

        self.train_bleu = []

        self.logger.log_metrics(metrics, step=step)

    def _reduce(self, outputs: list[Any], key: str):
        return torch.stack([out[key] for out in outputs]).mean().detach()

    # ----------------------------------------------------------------------------------------------
    # Validation
    # ----------------------------------------------------------------------------------------------
    def validation_step(self, batch: list[torch.Tensor], batch_idx: int) -> dict[str, Any]:  # type: ignore
        """
        Compute validation metrics.
        Parameters
        ----------
        batch : list[torch.Tensor]
            Validation batch.
        batch_idx : int
            Batch index.
        Returns
        -------
        dict[str, torch.Tensor]
            Metric values for a given batch.
        """

        inputs, targets = batch["text"], batch["label"]
        inputs = inputs.reshape(inputs.shape[0], inputs.shape[2]).T
        targets = targets.reshape(targets.shape[0], targets.shape[2]).T

        outputs = self(inputs, targets)  # basically equivalent to self.forward(data)

        outputs = torch.argmax(outputs, dim=2)

        for batch in range(outputs.shape[1]):
            tmp_outputs = []
            tmp_targets = []
            for i in range(outputs.shape[0]):
                tmp_outputs.append(self.get_word(self.words_dict, outputs[i, batch]))
                tmp_targets.append(self.get_word(self.words_dict, targets[i, batch]))
                
            self.val_bleu.append(self.metric(tmp_targets, tmp_targets))



        return {
            # 'additional_metric': ...
            # no need to return 'val_acc' here since it is always available as `self.val_acc`
        }

    def validation_epoch_end(self, outputs: list[Any]) -> None:
        """
        Log validation metrics.
        Parameters
        ----------
        outputs : list[Any]
            List of dictionaries returned by `self.validation_step` with batch metrics.
        """
        step = self.current_epoch + 1 if not self.trainer.running_sanity_check else self.current_epoch  # type: ignore

        metrics = {
            'epoch': float(step),
            'val_bleu': float(np.average(self.val_bleu)),
        }

        print("Bleu score: ")
        print(np.average(self.val_bleu))
        # Average additional metrics over all batches
        for key in outputs[0]:
            metrics[key] = float(self._reduce(outputs, key).item())
        
        self.val_bleu = []

        self.logger.log_metrics(metrics, step=step)