"""Datasets for experiments."""

import os
from typing import (
    Dict,
    Final,
    List,
    Optional,
    Tuple,
)

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, TensorDataset

from .utils import load_json, save_as_json

ROOT: Final[str] = 'data/'
DATASETS_BENCHMARK: Final[List] = [
    'airfoil',
    'concrete',
    'diabetes',
    'energy',
    'forest_fire',
    'yacht',
]
DATASETS_TOY: Final[List] = ['izmailov', 'regression2d', 'sinusoidal']


class RegrDataset(TensorDataset):
    """Torch dataset for benchmark data."""

    def __init__(self, x: np.ndarray, y: np.ndarray, device:torch.device) -> None:
        """Instantiate dataset."""
        x = torch.tensor(x, dtype=torch.float32, device=device)
        y = torch.tensor(y, dtype=torch.float32, device=device)
        self.params: Dict = {str: float}
        self.n_features = x.shape[1]
        super().__init__(x,y)


    def set_params(self, params: Dict[str, float]):
        """Set mean and std."""
        self.params.update(params)


class DatasetFactory:
    """Custom class for experiments."""

    @staticmethod
    def get(
        dataset_id: str,
        dataset_pth: str = ROOT,
        val_size: Optional[float] = 0.3,
        splits: Optional[str] = None,
        seed: int = 1,
        device: torch.device = torch.device('cpu'),
    ) -> Tuple[RegrDataset, RegrDataset]:
        """Return dataset from an identifier."""
        if dataset_id in DATASETS_BENCHMARK + DATASETS_TOY:
            data = np.loadtxt(os.path.join(dataset_pth, dataset_id + '.data'))
            x, y = data[:, :-1], data[:, -1]

            if splits is not None:
                splits_dict = load_json(splits)
                idx_train = splits_dict[dataset_id]['train']
                idx_test = splits_dict[dataset_id]['validate']
                x_train = np.take(x, idx_train, axis=0)
                x_test = np.take(x, idx_test, axis=0)
                y_train = np.take(y, idx_train, axis=0)
                y_test = np.take(y, idx_test, axis=0)
            else:
                x_train, x_test, y_train, y_test = train_test_split(
                    x, y, test_size=val_size, random_state=seed
                )
            scaler_x = StandardScaler().fit(x_train)
            scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
            x_train = scaler_x.transform(x_train)
            y_train = scaler_y.transform(y_train.reshape(-1, 1))
            x_test = scaler_x.transform(x_test)
            y_test = scaler_y.transform(y_test.reshape(-1, 1))

            data_train = RegrDataset(x_train, y_train, device=device)
            data_train.set_params({'mean': scaler_x.mean_, 'var': scaler_x.var_})
            data_test = RegrDataset(x_test, y_test, device=device)
            data_test.set_params({'mean': scaler_x.mean_, 'var': scaler_x.var_})

            if os.path.exists('mu_sigma.json'):
                mu_sigma = load_json('mu_sigma.json')
            else:
                mu_sigma = {}
            mu_sigma.update(
                {
                    dataset_id: {
                        'mean_x': list(scaler_x.mean_),
                        'var_x': list(scaler_x.var_),
                        'mean_y': list(scaler_y.mean_),
                        'var_y': list(scaler_y.var_),
                    }
                }
            )
            save_as_json(mu_sigma, 'mu_sigma.json')

        else:
            raise NotImplementedError(f'Dataset `{dataset_id}` not available.')

        return data_train, data_test
