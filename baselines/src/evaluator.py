"""Evaluation protocol for experiments."""

import os.path
from typing import List

from src.probabilistic_extensions import ProbabilisticPredictor
from src.utils import save_as_json


def get_posterior_samples(
    model_id: str,
    dataset_id: str,
    learners: List[ProbabilisticPredictor],
) -> None:
    """Evaluate learners on datasets w.r.t. several metrics."""
    for learner in learners:
        weights = learner.get_weights()
        save_as_json(
            weights,
            os.path.join(
                'weights', f'weights_{learner.get_id()}_{model_id}_{dataset_id}.json'
            ),
        )
        sigmas = {
            f'{learner.get_id()}_{idx}': float(sigma)
            for idx, sigma in enumerate(learner.get_sigma())
        }
        save_as_json(
            sigmas,
            os.path.join(
                'weights', f'sigmas_{learner.get_id()}_{model_id}_{dataset_id}.json'
            ),
        )
