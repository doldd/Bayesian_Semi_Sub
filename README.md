# Bayesian Semi-structured Subspace Inference

This repository contrains the code for the Bayesian Semi-structured Subspace Inference paper.
All worke was logged trough a local installed Weights & Biases instences.

The 

As the logged files are not instances of this repository, users must recomuptate the reuslts.

In the following gives an overview over the used files per experiements. 

### Experiment 1 (Toy Data)
 - semi_regression_Laplace.ipynb
 - semi_regression_train.ipynb
 - semi_subspace_vs_blackBox.ipynb

### Experiment 2 (Simulation Study)
- simulation_coverage_analyse.ipynb (To analyse)
- simulation_coverage.py (To train)
    We used the following configurations:
    - simulation_coverage.py -n 75 --lr 5e-3 --num_runs 50 --num_chains 10 --num_warmup 400 --num_samples 800 --dist normal_mu --use_hmc --p_struct=3 --num_bends {3, 5, 9,13, 17}
    - simulation_coverage.py -n 75 --lr 5e-4 --num_runs 50 --num_chains 10 --num_warmup 400 --num_samples 800 --dist poisson --use_hmc --p_struct=3 --max_epochs=2000 --num_bends={3, 5, 9,13, 17}

### Experiment 3 (UCI Benchmark)
 - UCI_benchmark.py
 - UCI_subspace.ipynb

### Experiment 4 (Melanoma Dataset)
 - train_semi_subspace_melanoma_ess.py
 - train_semi_subspace_melanoma.ipynb
 - melanoma_laplace.py
 - melanoma_laplace.ipynb