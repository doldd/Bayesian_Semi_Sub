"""Wrappers to make existing (deterministic) NN Bayesian."""
import os.path
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from laplace import Laplace
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader

from src.baselearner import NNLearner


class ProbabilisticPredictor(ABC):
    """Abstract class for probabilistic prediction."""

    @abstractmethod
    def get_id(self) -> str:
        """Return predictor OD."""
        pass

    @abstractmethod
    def get_models(self) -> List[nn.Module]:
        """Return trained models."""
        pass

    @abstractmethod
    def get_weights(self) -> Dict:
        """Return model weights."""
        pass

    def get_sigma(self) -> List[float]:
        """Return sigmas."""
        pass

    @abstractmethod
    def train(self, run_prefix: str, epochs: int) -> None:
        """Train predictor."""
        pass

    @abstractmethod
    def predict(self, dataloader: DataLoader, avg: bool = False) -> torch.Tensor:
        """Predict on given data."""
        pass


class DeepEnsemble(ProbabilisticPredictor):
    """Ensemble wrapper for NN learners."""

    def __init__(
        self,
        base_learner: NNLearner,
        ensemble_size: int,
        ckpt: str,
    ) -> None:
        """Instantiate ensemble."""
        self.base_learner_type = base_learner.get_model_id()
        self.ensemble_size = ensemble_size
        self.ensemble = []
        for _ in range(self.ensemble_size):
            self.ensemble.append(deepcopy(base_learner))
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.ckpt = ckpt

        self._init()

    def _init(self):
        """Initialize NN weights."""
        for idx, bl in enumerate(self.ensemble):
            bl.set_seed(idx)
            bl.init_model_weights(seed=idx)

    def get_id(self) -> str:
        """Return predictor ID."""
        return 'de'

    def get_models(self) -> List[nn.Module]:
        """Return models."""
        return [bl.get_model() for bl in self.ensemble]

    def get_weights(self) -> Dict:
        """Return state dicts of base learners."""
        return {
            f'de_{idx}': parameters_to_vector(bl.parameters()).tolist()[1:]
            for idx, bl in enumerate(self.ensemble)
        }

    def get_sigma(self) -> List[float]:
        """Return sigmas."""
        return [m.sigma.detach().numpy() for m in self.get_models()]

    def train(self, run_prefix: str, epochs: int) -> None:
        """Train ensemble."""
        print('---> Training ensemble...')

        for idx, bl in enumerate(self.ensemble):
            logger = pl.loggers.WandbLogger(
                project='src',
                name=f'{run_prefix}_{idx}',
                save_dir='wandb_logs',
            )
            trainer = pl.Trainer(
                max_epochs=epochs,
                logger=logger,
                gpus=torch.cuda.device_count(),
                num_sanity_val_steps=0,
                deterministic=True,
            )
            print(f'---> Training ensemble member {idx + 1}...')
            trainer.fit(bl)
            trainer.save_checkpoint(os.path.join(self.ckpt, f'trainer_{idx}.ckpt'))
            wandb.finish()  # necessary for each run to be actually logged

    def predict(self, dataloader: DataLoader, avg: bool = False) -> torch.Tensor:
        """Predict with ensemble."""
        print('---> Computing ensemble prediction...')

        ensemble_prediction = []
        for idx, bl in enumerate(self.ensemble):
            bl_predictions = torch.empty(0, device=self.device)
            for x, _ in dataloader:
                prediction = bl(x.float()).to(self.device)
                bl_predictions = torch.cat((bl_predictions, prediction), dim=0)
            ensemble_prediction.append(bl_predictions)
        ensemble_prediction = torch.stack(tuple(ensemble_prediction))
        if avg:
            ensemble_prediction = torch.mean(ensemble_prediction, dim=0)

        return ensemble_prediction

    def predict_array(self, x: torch.Tensor, avg: bool = False) -> torch.Tensor:
        """Predict with Laplace approximator."""
        ensemble_prediction = []
        for idx, bl in enumerate(self.ensemble):
            prediction = bl(x.float()).to(self.device)
            ensemble_prediction.append(prediction)
        ensemble_prediction = torch.stack(tuple(ensemble_prediction))
        if avg:
            ensemble_prediction = torch.mean(ensemble_prediction, dim=0)

        return ensemble_prediction


class LaplaceApproximator(ProbabilisticPredictor):
    """LA wrapper for NN learners."""

    def __init__(
        self,
        base_learner: NNLearner,
        ensemble_size: int,
        ckpt: str,
    ) -> None:
        """Instantiate Laplace approximator."""
        self.base_learner_type = base_learner.get_model_id()
        self.bl = base_learner
        self.ensemble_size = ensemble_size
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.ckpt = ckpt
        self.laplace_learner = None
        self.sigma = None

    def get_id(self) -> str:
        """Return predictor ID."""
        return 'la'

    def get_models(self) -> List[nn.Module]:
        """Return models."""
        return [self.bl.get_model()]

    def get_weights(self) -> Dict:
        """Return state dicts of base learners."""
        if self.laplace_learner is None:
            raise ValueError
        else:
            post_samples = self.laplace_learner.sample(self.ensemble_size)
            return {
                f'la_{idx}': post_samples[idx].tolist()
                for idx, _ in enumerate(post_samples)
            }

    def get_sigma(self) -> List[float]:
        """Return sigmas."""
        if self.sigma is not None:
            return [self.sigma]
        else:
            raise ValueError

    def train(self, run_prefix: str, epochs: int) -> None:
        """Train Laplace approximator."""
        print('---> Training base model...')

        logger = pl.loggers.WandbLogger(
            reinit=True,
            project='src',
            name=f'{run_prefix}',
            save_dir='wandb_logs',
        )
        trainer = pl.Trainer(
            max_epochs=epochs,
            logger=logger,
            gpus=torch.cuda.device_count(),
            num_sanity_val_steps=0,
            deterministic=True,
        )
        trainer.fit(self.bl)
        trainer.save_checkpoint(os.path.join(self.ckpt, 'trainer.ckpt'))

        print('---> Training Laplace approximator...')
        model = self.bl.get_model()
        self.sigma = model.sigma.detach().numpy()
        model.sigma = None
        self.laplace_learner = Laplace(
            model,
            likelihood='regression',
            subset_of_weights='all',
            hessian_structure='full',
        )
        if self.laplace_learner is not None:
            self.laplace_learner.fit(self.bl.train_dataloader())
            self.laplace_learner.optimize_prior_precision(
                pred_type='glm',
                method='marglik',
                verbose=True,
            )
            wandb.finish()

    def predict(self, dataloader: DataLoader, avg: bool = False) -> torch.Tensor:
        """Predict with Laplace approximator."""
        predictions = torch.empty(0, device=self.device)
        print('---> Drawing from Laplace approximate posterior...')
        for x, _ in dataloader:
            torch.manual_seed(123)
            if self.laplace_learner is None:
                raise ValueError
            else:
                next_prediction = self.laplace_learner.predictive_samples(
                    x.float(),
                    pred_type='glm',
                    n_samples=self.ensemble_size,
                )
                predictions = torch.cat(
                    (predictions, next_prediction.to(self.device)), dim=1
                )
            return predictions

    def predict_array(self, x: torch.Tensor, avg: bool = False) -> torch.Tensor:
        """Predict with Laplace approximator."""
        print('---> Drawing from Laplace approximate posterior...')
        random.seed(123)
        if self.laplace_learner is None:
            raise ValueError
        else:
            predictions = self.laplace_learner.predictive_samples(
                x.float(),
                pred_type='glm',
                n_samples=self.ensemble_size,
            )
            return predictions
