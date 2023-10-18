import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import MetricCollection, AUROC, CalibrationError, SumMetric, Accuracy, AveragePrecision, F1Score, \
    MeanSquaredError, MeanAbsoluteError
import argparse
# from subspace import curves
from torch import distributions as t_dist


class OutcomeDistributionLayerAutoConstrain(torch.nn.Module):
    def __init__(self, distribution: torch.distributions):
        self.dist_ = distribution
        super().__init__()

    def forward(self, x) -> torch.distributions:
        if len(x.shape) > 1:
            params = {key: t_dist.transform_to(constrain)(x[..., i]) for i, (key, constrain) in
                      enumerate(self.dist_.arg_constraints.items())}
        else:
            key, constrain = self.dist_.arg_constraints.copy().popitem()
            params = {key: t_dist.transform_to(constrain)(x)}
        return self.dist_(**params)

    def __repr__(self):
        return f"{self.__class__.__name__}(dist={self.dist_})"


class OutcomeDistributionLayer(torch.nn.Module):
    def __init__(self,
                 distribution: torch.distributions,
                 constrain_fns: dict = {'loc': lambda x: x, 'scale': torch.exp},
                 **dist_kwargs):
        super().__init__()
        self.dist_kwargs = dist_kwargs
        self.dist_ = distribution
        self.constrains = constrain_fns

    def forward(self, x) -> torch.distributions:
        assert x.shape[-1] == len(
            self.constrains), f"Got not enough distribution parameters, Got {x.shape[-1]} expected {len(self.constrains)}"
        params = {k: constrain(x[..., i]) for i, (k, constrain) in enumerate(self.constrains.items())}
        return self.dist_(**params, **self.dist_kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}(dist={self.dist_}; constrains={self.constrains})"


class NllLoss(torch.nn.modules.loss._Loss):
    def __init__(self, distribution: torch.distributions, constrains: dict = {}, reduction='mean', **dist_kwargs):
        super().__init__(reduction=reduction)
        if isinstance(distribution, OutcomeDistributionLayer):
            self.dist_ = distribution
        else:
            self.dist_ = OutcomeDistributionLayer(distribution, constrains,
                                                  **dist_kwargs)  # OutcomeDistributionLayer will instantiate a distribution with constrained support of the parameters

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        outcome_dist = self.dist_(
            input)  # constrain the inputs (shape: (#b, number parameters)) into parameter support and return the distribution
        assert (outcome_dist.batch_shape + outcome_dist.event_shape)[-1] == target.shape[-1], f"Expected target shape {outcome_dist.batch_shape + outcome_dist.event_shape} but got shape {target.shape} "    
        if self.reduction == 'mean':
            return -outcome_dist.log_prob(target).mean()
        elif self.reduction == 'sum':
            return -outcome_dist.log_prob(target).sum()
        else:
            return -outcome_dist.log_prob(target)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.dist_}) with reduction={self.reduction}"


class BaseModel(pl.LightningModule):
    def __init__(self,
                 loss_fn:NllLoss,
                 metric_collection=MetricCollection([CalibrationError(task='binary'),
                                                     AUROC(task='binary'),
                                                     Accuracy(task='binary'),
                                                     AveragePrecision(task='binary'),
                                                     F1Score(task='binary', threshold=0.1)
                                                     ]),
                 classification=False,
                 **kwargs
                 ):
        super(BaseModel, self).__init__()
        self.save_hyperparameters('loss_fn', 'metric_collection', 'classification')
        self.loss_fn = loss_fn
        self.valid_metrics = metric_collection.clone(prefix="valid/")
        self.test_metrics = metric_collection.clone(prefix="test/")
        self.classification=classification

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def configure_optimizers(self):
        lr = self.hparams['lr']
        wd = self.hparams['weight_decay']
        opti = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        return opti

    def training_step(self, train_batch, batch_idx) -> torch.Tensor:
        data, y = train_batch
        # y = y.squeeze(-1)
        if y.device != self.device:
            y = y.to(device=self.device)
            if type(data) is tuple:
                data = (d.to(device=self.device) for d in data)
            else:
                data = data.to(device=self.device)
        eta_prime = self(data)
        loss = self.loss_fn(eta_prime, y)
        self.log('train/loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx) -> dict:
        data, y = batch
        # y = y.squeeze(-1)
        eta_prime = self(data)
        loss = self.loss_fn(eta_prime, y)
        self.log('valid/loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        # ToDo refactor valid_metric (evaluate a model specific function instead of metric collection)
        expected = self.loss_fn.dist_(eta_prime).mean
        if self.classification:
            self.valid_metrics(expected, y.to(dtype=torch.int32))
        else:
            if (expected.dim() == 1) and (y.dim() == 2):
                self.valid_metrics(expected, y.squeeze(-1))
            else:
                self.valid_metrics(expected, y)
        self.log_dict(self.valid_metrics, on_step=False, on_epoch=True)
        return expected.detach()

    def test_step(self, batch, batch_idx) -> dict:
        data, y = batch
        # y = y.squeeze(-1)
        eta_prime = self(data)
        loss = self.loss_fn(eta_prime, y)
        self.log('test/loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        expected = self.loss_fn.dist_(eta_prime).mean
        if self.classification:
            self.test_metrics(expected, y.to(dtype=torch.int32))
        else:            
            if (expected.dim() == 1) & (y.dim() == 2):
                self.test_metrics(expected, y.squeeze(-1))
            else:
                self.test_metrics(expected, y)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)
        return expected.detach()


class SimpleCnn(BaseModel):
    def __init__(self, weight_decay: float = 0., lr: float = 1e-4, seed: int = None, **kwargs):
        kwargs.pop('classification', 0)
        super(SimpleCnn, self).__init__(classification=True, **kwargs)
        if seed is not None:
            pl.seed_everything(seed)
        # self.save_hyperparameters('num_bends', 'lr', 'seed')
        self.save_hyperparameters('weight_decay', 'lr', 'seed')
        self.conv = nn.Sequential(nn.Conv2d(3, 32, 3, ),  # 128
                                  nn.Tanh(),
                                  nn.MaxPool2d(2, ),  # 126
                                  nn.Conv2d(32, 32, 3),  # 63
                                  nn.Tanh(),
                                  nn.MaxPool2d(2, ),  # 61
                                  nn.Conv2d(32, 64, 3),  # 30
                                  nn.Tanh(),
                                  nn.MaxPool2d(2, ),  # 28
                                  nn.Conv2d(64, 64, 3),  # 14
                                  nn.Tanh(),
                                  nn.MaxPool2d(2, ),  # 12
                                  nn.Conv2d(64, 128, 3),  # 6
                                  nn.Tanh(),
                                  nn.MaxPool2d(2, )  # 4
                                  )
        self.lin = nn.Sequential(nn.Flatten(),  # 2
                                 nn.Linear(2 * 2 * 128, 128),
                                 nn.Tanh(),
                                 nn.Linear(128, 128),
                                 nn.Tanh(),
                                 nn.Linear(128, 1)
                                 )
        self.dnn = nn.Sequential(self.conv, self.lin)

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument('--lr', type=float, default=argparse.SUPPRESS, required=True)
        parser.add_argument('--weight_decay', type=float, default=argparse.SUPPRESS)
        parser.add_argument('--seed', type=int, default=argparse.SUPPRESS)
        return parent_parser

    def forward(self, x) -> torch.Tensor:
        out = self.dnn(x)
        return out


class RegNet(BaseModel):
    def __init__(self, input_dim, dimensions, output_dim: int = 1, weight_decay: float = 0., lr: float = 1e-4,
                 seed: int = None, loss_fn=nn.MSELoss(), 
                 metric_collection=MetricCollection([MeanSquaredError(),
                                                     MeanAbsoluteError()]),
                 activation: str = 'relu',
                 **kwargs):
        print("RegNet")
        super(RegNet, self).__init__(metric_collection=metric_collection,
                                     loss_fn=loss_fn,
                                     classification=False,
                                     **kwargs)
        if seed is not None:
            pl.seed_everything(seed)
        # self.save_hyperparameters('num_bends', 'lr', 'seed')
        self.save_hyperparameters('weight_decay', 'lr', 'seed', 'input_dim', 'dimensions', 'output_dim')

        # Define Model architecture
        dimensions = [input_dim, *dimensions, output_dim]
        modules = []
        for i in range(len(dimensions) - 1):
            modules.append(torch.nn.Linear(dimensions[i], dimensions[i + 1]))
            if i < len(dimensions) - 2:
                if activation == 'tanh':
                    modules.append(torch.nn.Tanh())
                else:
                    modules.append(torch.nn.ReLU())
        self.dnn = nn.Sequential(*modules)

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument('--lr', type=float, default=argparse.SUPPRESS, required=True)
        parser.add_argument('--weight_decay', type=float, default=argparse.SUPPRESS)
        parser.add_argument('--seed', type=int, default=argparse.SUPPRESS)
        return parent_parser

    def forward(self, x) -> torch.Tensor:
        out = self.dnn(x)
        return out