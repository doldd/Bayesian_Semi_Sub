import numpy as np
import torch
import math
from torch.utils.data._utils import collate, pin_memory
import pytorch_lightning as pl
from torchmetrics import Metric
from pytorch_lightning.callbacks import Callback
from torch.utils.data import TensorDataset, Subset, Dataset

__all__ = ['FastFillTensorDataLoader', 'BrierScore', 'NllClassification', 'LogSigmaBatch', 'LogSigmaEpoch']


class JitBatchTransform(torch.nn.Module):
    def __init__(self, transform: torch.nn.Sequential):
        super().__init__()
        self.trans = transform

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.stack([self.trans(x) for x in input], dim=0)


# class FastFillTensorDataLoader(torch.utils.data.dataloader.DataLoader):
class FastFillTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).

    See: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, data, batch_size=32, shuffle=False, drop_last=False, fill_last=False, pin_memory=False,
                 transform_per_iter: torch.nn.Sequential = None):
        """
        Initialize a FastTensorDataLoader.

        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.

        :returns: A FastTensorDataLoader.
        """
        self.num_workers = 0
        self.worker_init_fn = None
        if type(data) == tuple:
            tensors = list(data)
            assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
            self.dataset = TensorDataset(*tensors)
        elif isinstance(data, TensorDataset):
            self.dataset = data
        elif isinstance(data, Subset) & isinstance(data.dataset, TensorDataset):
            self.dataset = data
        elif isinstance(data, Dataset):
            self.dataset = self.dataset_to_tensordataset(data)

        self.dataset_len = len(self.dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.fill_last = fill_last
        self.pin_memory = pin_memory
        self.batch_sampler = None
        self.jit_transform = torch.jit.script(
            JitBatchTransform(transform_per_iter)) if transform_per_iter is not None else False
        self.n_max_ = self.len__()

    @staticmethod
    def dataset_to_tensordataset(dataset):
        with torch.no_grad():
            test1 = dataset[0]
            x_stack = torch.zeros((len(dataset), *(test1[0].shape)), dtype=test1[0].dtype)
            if type(test1[1]) is int:
                y_stack = torch.zeros((len(dataset)), dtype=torch.int64)
            else:
                y_stack = torch.zeros((len(dataset), *(test1[1].shape)), dtype=test1[1].dtype)
            for i in range(len(dataset)):
                x_stack[i], y_stack[i] = dataset[i]
            return TensorDataset(x_stack, y_stack)

    def len__(self):
        if self.drop_last:
            return self.dataset_len // self.batch_size  # type: ignore
        elif self.fill_last:
            return int(math.ceil(self.dataset_len / float(self.batch_size)))
        else:
            return (self.dataset_len + self.batch_size - 1) // self.batch_size  # type: ignore

    def __len__(self):
        return self.n_max_

    def __iter__(self):
        self.index = torch.arange(self.dataset_len)
        if self.shuffle:
            self.index = torch.randperm(self.dataset_len)
        else:
            self.index = torch.arange(self.dataset_len)
        if self.drop_last:
            max_ = (self.dataset_len // self.batch_size) * self.batch_size
            self.index = self.index[:max_]
        elif self.fill_last:
            missing = self.batch_size - (self.dataset_len % self.batch_size)
            r = torch.randperm(self.dataset_len)[:missing]
            self.index = torch.hstack((self.index, r))
        self.i = 0
        self.index = self.index.tolist()
        return self

    def __next__(self):
        # if (self.i // self.batch_size) >= len(self):
        #     raise StopIteration
        if self.i < self.n_max_:
            data = self.dataset[self.index[self.i * self.batch_size: (self.i + 1) * self.batch_size]]
            self.i += 1
            if len(data) == 2:
                x = data[0]  # x is single datasets
            else:
                x = data[:-1]  # x contain multiple datasets
            y = data[-1]
            if self.jit_transform:
                x = self.jit_transform(x)
            # batch = [t[self.i:self.i+self.batch_size] for t in self.tensors]
            if self.pin_memory:
                x = pin_memory.pin_memory(x)
                y = pin_memory.pin_memory(y)
            return x, y
        else:
            raise StopIteration


class BrierScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        target_one_hot = torch.zeros_like(preds)
        target_one_hot[np.arange(0, len(target)), target] = 1.

        self.correct += torch.square(preds - target_one_hot).sum()
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total


class NllClassification(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.nll_ = torch.nn.CrossEntropyLoss(reduction="sum")
        self.add_state("nll_s", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.nll_s += self.nll_(preds, target)
        self.total += target.numel()

    def compute(self):
        return self.nll_s.float() / self.total


# class LogSigma(Callback):
#     def __init__(self, pl_module: "pl.LightningModule"):
#         mns = [(mn, m) for mn, m in pl_module.feature_extractor.named_modules() if hasattr(m, "weight_sigma") or isinstance(m, _SpectralBatchNorm)]
#         self.sn_modules_ = mns
#
#     def on_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
#         with torch.no_grad():
#             for mn, m in pl_module.named_modules():
#                 if hasattr(m, "weight_sigma"):
#                     pl_module.log(f"Sigma/{mn}", m.weight_sigma, on_step=True, on_epoch=True)
#                 if isinstance(m, _SpectralBatchNorm):
#                     lipschitz = torch.max(torch.abs(m.weight * (m.running_var + m.eps) ** -0.5))
#                     pl_module.log(f"Sigma/{mn}", lipschitz, on_step=True, on_epoch=True)


class _LogSigma(Callback):
    def __init__(self, pl_module: "pl.LightningModule", on_batch=False):
        from due.layers.spectral_batchnorm import _SpectralBatchNorm
        mns = [(mn, m) for mn, m in pl_module.feature_extractor.named_modules() if
               hasattr(m, "weight_sigma") or isinstance(m, _SpectralBatchNorm)]
        self.sn_modules_ = mns


class LogSigmaBatch(_LogSigma):
    def on_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        with torch.no_grad():
            def log_fn(mn, value):
                pl_module.log(f"Sigma/{mn}", value, on_step=True, on_epoch=False)

            [log_fn(mn, m.weight_sigma) if hasattr(m, "weight_sigma") else log_fn(mn, torch.max(
                torch.abs(m.weight * (m.running_var + m.eps) ** -0.5))) for mn, m in self.sn_modules_]


class LogSigmaEpoch(_LogSigma):
    def on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        with torch.no_grad():
            def log_fn(mn, value):
                pl_module.log(f"Sigma/{mn}", value, on_epoch=True, on_step=False)

            [log_fn(mn, m.weight_sigma) if hasattr(m, "weight_sigma") else log_fn(mn, torch.max(
                torch.abs(m.weight * (m.running_var + m.eps) ** -0.5))) for mn, m in self.sn_modules_]

# class LogSigma2(Callback):
#     def __init__(self, pl_module: "pl.LightningModule"):
#         mns = [(mn, m) for mn, m in pl_module.feature_extractor.named_modules() if hasattr(m, "weight_sigma") or isinstance(m, _SpectralBatchNorm)]
#         self.sn_modules_ = mns
#         self.accum_ = []
#
#     def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
#         self.accum_ = []
#
#     def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
#         pl_module.register_buffer("lipschitz_history", torch.stack(self.accum_))
#
#     def on_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
#         with torch.no_grad():
#             lipschitz = [m.weight_sigma if hasattr(m, "weight_sigma") else torch.max(torch.abs(m.weight * (m.running_var + m.eps) ** -0.5)) for mn, m in self.sn_modules_]
#             self.accum_.append(lipschitz)

def r_hat_from_samples(samples):
    # compute R hat
    # samples_split = df_samples[r'$\theta_{cat2}$'].to_numpy().reshape(-1,250)
    samples_split = samples # ieine shape mit (z,y)
    W = samples_split.var(ddof=1, axis=1).mean()
    B = ((samples_split.mean(1)-samples_split.mean())**2).sum() * samples_split.shape[1]/(samples_split.shape[0] -1)
    B
    var_hat_plus = 249/250*W + 1/250*B
    np.sqrt(var_hat_plus)
    r_hat = np.sqrt(var_hat_plus/W)
    r_hat