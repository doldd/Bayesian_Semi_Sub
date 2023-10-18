"""
Perform experiments.
Folder baselines cloned from https://github.com/jgwiese/mcmc_bnn_symmetry
"""

import os
from copy import deepcopy

import click
import torch.cuda

from src.baselearner import NNLearner
from src.datasets import (
    DATASETS_BENCHMARK,
    DATASETS_TOY,
    DatasetFactory,
)
from src.evaluator import get_posterior_samples
from src.models import MLP
from src.probabilistic_extensions import DeepEnsemble, LaplaceApproximator


@click.command()
@click.option('-lr', '--learning-rate', required=True, default=0.0001)
@click.option('-esd', '--size-de', required=True, default=10)
@click.option('-esl', '--size-la', required=True, default=1274)
@click.option('-ep', '--epochs', required=True, default=500)
def main(
    learning_rate: float,
    size_de: int,
    size_la: int,
    epochs: int,
):
    """Run experiments."""
    for dataset_id in DATASETS_BENCHMARK + DATASETS_TOY:

        if dataset_id in DATASETS_BENCHMARK:
            split_file = os.path.join('data', 'dataset_indices_0.2.json')
        else:
            split_file = os.path.join('data/', 'toy_dataset_indices_0.2.json')
        data_train, data_test = DatasetFactory.get(dataset_id, splits=split_file)

        models = [
            MLP(input_size=data_train.n_features, hidden_sizes=[3]),
            MLP(input_size=data_train.n_features, hidden_sizes=[16, 16, 16]),
        ]
        model_names = ['mlp_1x3', 'mlp3x16']

        for md, mdname in zip(models, model_names):

            learner = NNLearner(
                device='cuda:0' if torch.cuda.is_available() else 'cpu',
                model=md,
                prop_val=0.0,
                hyperparams={
                    'learning_rate': int(learning_rate),
                    'batch_size_train': len(data_train),
                    'weight_decay': 0.0,
                },
                dataset=data_train,
            )

            de = DeepEnsemble(deepcopy(learner), ensemble_size=int(size_de), ckpt='de')
            la = LaplaceApproximator(
                deepcopy(learner), ensemble_size=int(size_la), ckpt='la'
            )

            effective_epochs = int(epochs) if mdname == 'mlp_1x3' else 2 * int(epochs)
            de.train(run_prefix=f'de_{mdname}_{dataset_id}', epochs=effective_epochs)
            la.train(run_prefix=f'la_{mdname}_{dataset_id}', epochs=effective_epochs)

            get_posterior_samples(
                dataset_id=dataset_id,
                model_id=mdname,
                learners=[de, la],
            )


if __name__ == '__main__':
    main()
