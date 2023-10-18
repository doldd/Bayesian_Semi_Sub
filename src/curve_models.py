import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pytorch_lightning as pl
from torchmetrics import MetricCollection, AUROC, CalibrationError, SumMetric
import argparse
import wandb
from pytorch_lightning.callbacks import Callback
# from ./dnn import curves
from dnn-mode-connectivity import curves
from copy import deepcopy

from src.base_models import BaseModel


class CurveArchitectureFromSequential(nn.Module):
    def __init__(self, fix_points: list, base_sequential_model: nn.Sequential):
        """
        # Base Class for simple Curve Architecture based on a non nested sequential container
        :type base_sequential_model: nn.Sequential
        :type fix_points: list
        :param fix_points:
        :param base_sequential_model: 
        """
        super().__init__()
        modules = []
        for m in base_sequential_model.modules():
            if type(m) is not type(base_sequential_model):
                if type(m) is nn.Linear:
                    modules.append(
                        eval(f"curves.{type(m).__name__}")(m.in_features, m.out_features, fix_points=fix_points))
                elif type(m) is nn.Conv2d:
                    modules.append(
                        eval(f"curves.{type(m).__name__}")(in_channels=m.in_channels, 
                                                           out_channels=m.out_channels, 
                                                           kernel_size=m.kernel_size, 
                                                           stride=m.stride, 
                                                           fix_points=fix_points))
                elif type(m) is nn.BatchNorm2d:
                    modules.append(
                        eval(f"curves.{type(m).__name__}")(num_features=m.num_features, 
                                                           eps=m.eps, 
                                                           momentum=m.momentum, 
                                                           affine=m.affine, 
                                                           track_running_stats=m.track_running_stats, 
                                                           fix_points=fix_points))
                else:
                    modules.append(deepcopy(m))
        self.net_ = nn.Sequential(*modules)

    def forward(self, x, coeffs_t):
        for m in self.net_.modules():
            if issubclass(m.__class__, curves.CurveModule):
                x = m(x, coeffs_t)
            elif m != self.net_:
                x = m(x)
        return x


class CurveArchitectureSemiBlackBox(nn.Module):
    def __init__(self, fix_points: list, base_model: nn.Sequential):
        """
        # Base Class for simple Curve Architecture based on a non nested sequential container
        :type base_sequential_model: nn.Sequential
        :type fix_points: list
        :param fix_points:
        :param base_sequential_model:
        """
        super().__init__()
        modules = []
        for m in base_model.dnn.modules():
            if type(m) is not type(base_model.dnn):
                if type(m) is nn.Linear:
                    modules.append(
                        eval(f"curves.{type(m).__name__}")(m.in_features, m.out_features, fix_points=fix_points))
                else:
                    modules.append(type(m)())
        self.lin_curve_ = nn.Sequential(*modules)
        self.structural_curve_ = curves.Linear(base_model.structure_lin.in_features,
                                               base_model.structure_lin.out_features, bias=False, fix_points=fix_points)

    def forward(self, data, coeffs_t):
        u, x = data
        for m in self.lin_curve_.modules():
            if issubclass(m.__class__, curves.CurveModule):
                u = m(u, coeffs_t)
            elif m != self.lin_curve_:
                u = m(u)
        return u + self.structural_curve_(x, coeffs_t)


class CurveModel(BaseModel):
    def __init__(self, model_start, model_end, num_bends: int = 3, lr: float = 1e-4, seed: int = None,
                 weight_decay: float = 0., architecture=CurveArchitectureFromSequential, architecture_kwargs={},
                 fix_start=True, fix_end=True, **kwargs):
        """
        # wraps curve model into lightning module

        :param model_start:
        :param model_end:
        :param num_bends:
        :param lr:
        :param seed:
        :param weight_decay:
        :param architecture:
        :param architecture_kwargs:
        :param kwargs:
        """
        super(CurveModel, self).__init__(**kwargs)
        if seed is not None:
            pl.seed_everything(seed)
        self.save_hyperparameters('num_bends', 'lr', 'seed', 'weight_decay', 'architecture')

        # do Curve stuff
        self.model = curves.CurveNet(curves.Bezier, architecture, num_bends, fix_start=fix_start, fix_end=fix_end,
                                     architecture_kwargs=architecture_kwargs)
        if model_start is not None and model_end is not None:
            # set start/ end point of curve
            self.model.import_base_parameters(model_start, 0)
            self.model.import_base_parameters(model_end, num_bends - 1)
            # initiate additional support points such that the curve is a line
            self.model.init_linear()

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument('--lr', type=float, default=argparse.SUPPRESS, required=True)
        parser.add_argument('--weight_decay', type=float, default=argparse.SUPPRESS)
        parser.add_argument('--seed', type=int, default=argparse.SUPPRESS)
        parser.add_argument('--num_bends', type=int, default=argparse.SUPPRESS)
        return parent_parser

    def forward(self, *args) -> torch.Tensor:
        t = args[0].data.new(1).uniform_()
        out = self.model(args if len(args) > 1 else args[0], t)
        return out


class CurveLogger(Callback):
    def __init__(self, t_space=np.linspace(0, 1, 20)):
        super().__init__()
        self.metric_collection = MetricCollection([
            CalibrationError(task='binary'),
            AUROC(task='binary'),
        ])
        self.t_space_ = t_space
        self.len_ds = 0
        self.metrics_for_t = None
        self.logged_y_lim = None

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.logged_y_lim = None
        self.metrics_for_t = [(self.metric_collection.clone().to(device=pl_module.device),
                               SumMetric()) for i in range(len(self.t_space_))]

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        [(mm.reset(), ml.reset()) for mm, ml in self.metrics_for_t]
        self.len_ds = 0

    def on_validation_batch_start(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch, batch_idx: int, dataloader_idx: int=0
    ) -> None:
        (u, x), y = batch
        self.len_ds += len(y)
        for t_curve, (mm, ml) in zip(self.t_space_, self.metrics_for_t):
            t = y.data.new_full((1,), t_curve)
            param = pl_module.forward_structure(x)
            param += pl_module.model(u, t)
            expected = pl_module.loss_fn.dist_(param).mean
            nll = pl_module.loss_fn(param, y).item()
            ml.update(nll)
            mm.update(expected, y.to(dtype=torch.int32))

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        all_metrics = [{**mm.compute(), 'nll': ml.compute().item() / self.len_ds} for mm, ml in self.metrics_for_t]
        df_t_space = pd.DataFrame(all_metrics)
        df_t_space['BinaryCalibrationError'] = df_t_space['BinaryCalibrationError'].apply(lambda x: x.item())
        df_t_space['BinaryAUROC'] = df_t_space['BinaryAUROC'].apply(lambda x: x.item())
        df_t_space['t'] = self.t_space_
        table = wandb.Table(data=df_t_space.to_numpy().tolist(), columns=df_t_space.columns.tolist())
        pl_module.logger.experiment.log(dict(curve_points=table))
        fig, ax = plt.subplots(dpi=150)
        ax2 = ax.twinx()
        ax2.scatter(x=df_t_space['t'].to_numpy(), y=df_t_space['BinaryAUROC'].to_numpy(), label='BinaryAUROC',
                    color=sns.color_palette('tab10')[1])
        ax2.legend()
        ax2.set_ylabel('BinaryAUROC')
        ax.scatter(x=df_t_space['t'].to_numpy(), y=df_t_space['nll'].to_numpy(), label='nll',
                   color=sns.color_palette('tab10')[0])
        ax.legend(loc='lower left')
        ax.set_ylabel('nll')
        ax.set_xlabel("t")
        plt.title("Walk along Bézier curve")
        if self.logged_y_lim is None:
            self.logged_y_lim = [ax.get_ylim(), ax2.get_ylim()]
        else:
            if ((self.logged_y_lim[0][1] + self.logged_y_lim[0][0]) / 2 > ax.get_ylim()[1]) or (
                    self.logged_y_lim[0][1] < ax.get_ylim()[1]):
                self.logged_y_lim[0] = ax.get_ylim()
            else:
                ax.set_ylim(self.logged_y_lim[0])
            if ((self.logged_y_lim[1][1] + self.logged_y_lim[1][0]) / 2 < ax2.get_ylim()[0]) or (
                    self.logged_y_lim[1][0] > ax2.get_ylim()[0]):
                self.logged_y_lim[1] = ax2.get_ylim()
            else:
                ax2.set_ylim(self.logged_y_lim[1])
        # plt.savefig("plots/nll_auroc_along_bezier.svg")
        pl_module.logger.experiment.log({'Nll and AUROC on curve': wandb.Image(fig)})