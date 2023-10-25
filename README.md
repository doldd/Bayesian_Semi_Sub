# Bayesian Semi-structured Subspace Inference

This repository contains the code for the Bayesian Semi-structured Subspace Inference paper. 
All work was tracked using locally installed Weights & Biases instances. 
Please note that the log files are not included in this repository.

We also incorporated code snippets from the following sources:

    https://github.com/wjmaddox/drbayes/
    https://github.com/timgaripov/dnn-mode-connectivity
    https://github.com/jgwiese/mcmc_bnn_symmetry

The following listing provides an overview of the files used in each experiment:

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