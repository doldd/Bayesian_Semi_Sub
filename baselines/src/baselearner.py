"""Training routine for experiments."""

from typing import (
    Any,
    Dict,
    Optional,
)

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader, random_split

from .utils import custom_loss_fun, init_weights


class NNLearner(pl.LightningModule):
    """Vanilla network training."""

    def __init__(
        self,
        device: str,
        model: nn.Module,
        hyperparams: Dict,
        dataset: Any,
        test_dataset: Optional[Any] = None,
        prop_val: float = 0.3,
        member: int = 0,
        seed: int = 1,
    ) -> None:
        """Set up learner object."""
        super().__init__()
        self._device = device
        self.member = member
        self.model = model.to(self._device)
        self.seed = seed

        self.prop_val = prop_val
        val_set_size = int(len(dataset) * self.prop_val)
        train_set_size = len(dataset) - val_set_size
        seed = torch.Generator().manual_seed(seed)
        train_set, val_set = random_split(
            dataset, [train_set_size, val_set_size], generator=seed
        )
        self.data_train = train_set
        self.data_val = val_set
        self.data_test = test_dataset

        self.optimizer = None
        self.scheduler = None
        self.hyperparams = hyperparams

        self.loss_fun_train = torch.nn.MSELoss()
        self.loss_train = torchmetrics.MeanMetric(prefix='train')
        self.loss_val = torchmetrics.MeanMetric(prefix='val')

        self.rmse_train = torchmetrics.MeanSquaredError(squared=False)
        self.rmse_val = torchmetrics.MeanSquaredError(squared=False)

    def on_fit_start(self) -> None:
        """Set global seed."""
        pl.seed_everything(seed=self.seed)

    def train_dataloader(self) -> DataLoader:
        """Set up data loader for training."""
        return DataLoader(
            self.data_train,
            batch_size=self.hyperparams.get('batch_size_train') or 32,
            shuffle=True,
            num_workers=1,
        )

    def val_dataloader(self) -> DataLoader:
        """Set up data loader for validation."""
        return DataLoader(
            self.data_val,
            batch_size=self.hyperparams.get('batch_size_test') or 32,
            num_workers=1,
        )

    def configure_optimizers(self) -> Dict:
        """Set up optimization-related objects."""
        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(),
            # momentum=0.01,
            lr=self.hyperparams.get('learning_rate') or 0.01,
            weight_decay=self.hyperparams.get('weight_decay') or 0.0,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.hyperparams.get('step_size') or 10,
            gamma=self.hyperparams.get('gamma') or 0.1,
        )

        return {
            'optimizer': self.optimizer,
          }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define standard forward pass."""
        return self.model(x)

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Define training routine."""
        x, y = batch
        preds = self.model(x.float())
        weights = parameters_to_vector(self.get_model().parameters())
        print(weights)
        sigma = self.get_model().sigma
        loss = custom_loss_fun(preds, y, sigma, weights)
        self.rmse_train.update(preds, y)
        self.loss_train.update(loss)

        return loss

    def training_epoch_end(self, outputs) -> None:
        """Collect metrics after each training step."""
        loss = self.loss_train.compute()
        rmse = self.rmse_train.compute()
        self.log('loss_train', loss)
        self.log('rmse_train', rmse)
        self.loss_train.reset()
        self.rmse_train.reset()

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        """Define validation routine."""
        x, y = batch
        preds = self.model(x.float())
        sigma = self.get_model().sigma
        weights = parameters_to_vector(self.get_model().parameters())
        loss = custom_loss_fun(preds, y, sigma, weights)
        self.loss_val.update(loss)
        self.rmse_val.update(preds, y)

    def validation_epoch_end(self, outputs) -> None:
        """Collect metrics after each validation step."""
        loss = self.loss_val.compute()
        rmse = self.rmse_val.compute()
        self.log('loss_val', loss)
        self.log('rmse_val', rmse)
        self.loss_val.reset()
        self.rmse_val.reset()

    def init_model_weights(self, seed: Optional[int] = None) -> None:
        """Initialize model weights."""
        if seed is not None:
            torch.manual_seed(seed)
        self.model.apply(init_weights)

    def get_model_id(self) -> str:
        """Get model identifier as handed to model factory."""
        return self.model.model_id

    def get_model(self) -> nn.Module:
        """Get model."""
        return self.model

    def set_member_number(self, number: int) -> None:
        """Assign member number in ensemble."""
        self.member = number

    def set_seed(self, seed: int) -> None:
        """Set seed to custom value."""
        self.seed = seed
