import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pytorch_lightning as pl
from torchmetrics import MetricCollection, AUROC, CalibrationError, SumMetric, Accuracy
import wandb
from utils_datamodel.wandb_utils import load_model, parse_runs, wandb_table_to_dataframe
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

import argparse
from src.ess import SemiSubEllipticalSliceSampling
import os

from tqdm import trange
from src.model import RegNet, CurveModel, CurveArchitectureFromSequential, getModel, cudaIter, OutcomeDistributionLayer, \
    NllLoss, pyroSemiSubspace, NumpyroModel
from src.semi_sub_utils import get_curve_space_torch, span_space_from_curve_model
from torchmetrics import MeanSquaredError, MeanAbsoluteError
from tqdm.notebook import tqdm
from functools import partial
import torch.distributions as t_dist
from sklearn.model_selection import StratifiedKFold
from numpyro.diagnostics import summary
from src.semi_sub_utils import features, get_ds_test_from_df, get_ds_from_df, base_net_kwargs, expCollector, log_pointwise_predictive_likelihood
from src.semi_subspace import *
from utils_datamodel.utils import FastFillTensorDataLoader
from joblib import Parallel, delayed
import pyro
from pyro.infer.autoguide.initialization import init_to_sample
from pyro.poutine.indep_messenger import IndepMessenger
from src.plot import plot_subspace, plot_subspace_solution_regression_pyro

from numpyro.contrib.module import flax_module, _update_params
from copy import deepcopy
import arviz as az
from arviz import labels as azl
import xarray as xr

# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=12"
import jax
import jax.numpy as jnp
from numpyro.infer.initialization import init_to_sample as np_init_to_sample
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import numpyro
from pyro import poutine

numpyro.set_platform('cpu')
numpyro.set_host_device_count(40)
from numpyro import distributions as np_dist
import flax.linen as nn

from numpyro import handlers
from jax.scipy.special import logsumexp


def xgen(n, p, rng, name_prefix="x_struct_"):
    data = pd.DataFrame(rng.normal(size=(n, p)))
    data.columns = [name_prefix + str(i) for i in range(1, p + 1)]
    return data


def data_generation_simulation1(n, p_struct, outcome_dist_layer: OutcomeDistributionLayer, base_net_kwargs_g, seed,
                                dir_to_save, prior_dist, device):
    rng = np.random.default_rng(seed=seed)
    n_all = int(n * 1.7)  # use 20% for validation and 50% for testing
    # generate data
    Xstruct_df = xgen(n_all, p_struct, rng=rng)
    Xstruct = torch.from_numpy(Xstruct_df.to_numpy().astype(np.float32))
    X_df = xgen(n_all, base_net_kwargs_g['input_dim'], rng=rng, name_prefix="x_unstruct_")
    X = torch.from_numpy(X_df.to_numpy().astype(np.float32))

    net_kwargs_g = dict(lr=5e-2,
                        weight_decay=1e-5,
                        loss_fn=torch.nn.MSELoss(reduction='mean'),
                        num_structure=p_struct,
                        ortho_layer_name_nn_head=None
                        )

    with torch.no_grad():
        model = getModel(RegNet, seed=rng.integers(100000), **base_net_kwargs_g, **net_kwargs_g)
        # draw true thetas from prior dist
        true_thetas = prior_dist.sample(model.structure_lin.weight.shape)
        print("true_thetas: ", true_thetas)
        model.structure_lin.weight.data = true_thetas
        lambda_struct = model.structure_lin(Xstruct)
        lambda_unstruct = model.dnn(X)
        # lambda_struct = (lambda_struct - lambda_struct.mean()) / lambda_struct.std()
        # lambda_unstruct = (lambda_unstruct - lambda_unstruct.mean()) / lambda_unstruct.std()
        # epsilon = 0.01 * torch.randn_like(lambda_unstruct)
        # lambda_ = lambda_struct + tau * lambda_unstruct
        lambda_ = lambda_struct + lambda_unstruct
        y = outcome_dist_layer(lambda_).sample()

    true_thetas = model.structure_lin.weight.detach().clone().squeeze().numpy()

    shuffle_idx = rng.permutation(np.arange(n_all))
    dataset = torch.utils.data.TensorDataset(X.to(device=device),
                                             Xstruct.to(device=device),
                                             y.to(device=device))
    train_dataset = torch.utils.data.TensorDataset(*dataset[shuffle_idx[:n]])
    valid_dataset = torch.utils.data.TensorDataset(*dataset[shuffle_idx[n:n + int(n * 0.2)]])
    test_dataset = torch.utils.data.TensorDataset(*dataset[shuffle_idx[n + int(n * 0.2):]])
    print("length train dataset:", len(train_dataset))
    print("length valid dataset:", len(valid_dataset))
    print("length test dataset:", len(test_dataset))

    if dir_to_save is not None:
        fname = os.path.join(dir_to_save, "dataset_simulation.pt")
        dict_to_save = dict(train=train_dataset,
                            valid=valid_dataset,
                            test=test_dataset,
                            true_thetas=true_thetas,
                            model_state_dict=model.state_dict(),
                            model_kwargs=dict(**base_net_kwargs_g, **net_kwargs_g))
        torch.save(dict_to_save, fname)
        print("saved all datasets. Filename:", fname)

    train_loader = FastFillTensorDataLoader(train_dataset, batch_size=n, shuffle=True, pin_memory=False)
    valid_loader = FastFillTensorDataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=False,
                                            pin_memory=False)
    test_loader = FastFillTensorDataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, pin_memory=False)

    return [train_loader, valid_loader, test_loader], true_thetas, model


def plot_curve_solution(exp_col: expCollector, best_curve_model, wandb_logger):
    plt.figure()
    u_train, x_train, y_train = exp_col.train_dataloader.dataset[:]
    u_valid, x_valid, y_valid = exp_col.valid_dataloader.dataset[:]
    u_test, x_test, y_test = exp_col.test_dataloader.dataset[:]
    t_space = torch.linspace(0, 1., 101, device=u_train.device)
    nll_vs_t = {'train': [], 'valid': [], 'test': []}
    nll_fn = exp_col.nll_fn
    for t in t_space:
        out_t = best_curve_model.model(u_train, t) + best_curve_model.structure_lin(x_train)
        out_v = best_curve_model.model(u_valid, t) + best_curve_model.structure_lin(x_valid)
        out_test = best_curve_model.model(u_test, t) + best_curve_model.structure_lin(x_test)
        nll_vs_t['train'].append(nll_fn(out_t, y_train).detach().cpu().item())
        nll_vs_t['valid'].append(nll_fn(out_v, y_valid).detach().cpu().item())
        nll_vs_t['test'].append(nll_fn(out_test, y_test).detach().cpu().item())
    for dataset, nll in nll_vs_t.items():
        plt.plot(t_space.cpu().numpy(), nll, label=dataset)
    plt.xlabel("Bézier curve t-space")
    plt.ylabel(r'nll ~ $N(y|\mu=DNN(u)+\theta x,\sigma=1.)$')
    plt.legend()
    wandb_logger.experiment.log({"Bézier Curve NLL": wandb.Image(plt.gcf())})


def run_ess(exp_col, curve_model, curve_wandb_id, loss_fn, prior_scale, num_chains, num_warmup, num_samples, num_data,
            **kwargs):
    # create subspace model
    subspace_model, wandb_logger = load_subspace_model(exp_col,
                                                       curve_model,
                                                       curve_wandb_id)
    # %% Run the ESS on the subspace
    dataset = exp_col.train_dataloader.dataset
    cuda_loader_no_shuffle = FastFillTensorDataLoader(dataset, batch_size=len(dataset), shuffle=False, pin_memory=False)
    loss_fn.reduction = None
    ess = SemiSubEllipticalSliceSampling(subspace_model,
                                         prior_scale=prior_scale,
                                         prior_scale_subspace=5.,
                                         temperature=1.,
                                         temperature_only_on_nn=False,
                                         num_integration_samples=2000,
                                         num_chains=num_chains,
                                         num_warmup=num_warmup,
                                         integration_range=20.,
                                         device=device,
                                         criterion=loss_fn,
                                         num_samples=num_samples,
                                         seed=exp_col.seed)
    wandb_logger.experiment.config.update({'ess_param': ess.hparams_initial})
    logprobs = ess.fit(dataset=cuda_loader_no_shuffle)

    # compute predictive performance
    xt, mt, yt = exp_col.test_dataloader.dataset[:]
    preds = ess.predict(ess.all_samples.T, xt, mt)
    lppd = (torch.logsumexp(-loss_fn(preds, yt), dim=0) - np.log(preds.shape[0])).sum()
    wandb_logger.experiment.summary["test_lppd"] = lppd

    u_valid, x_valid, y_valid = exp_col.valid_dataloader.dataset[:]
    preds = ess.predict(ess.all_samples.T, u_valid, x_valid)
    lppd = (torch.logsumexp(-loss_fn(preds, y_valid), dim=0) - np.log(preds.shape[0])).sum()
    wandb_logger.experiment.summary["valid_lppd"] = lppd

    # create arviz inference object
    labeller = azl.MapLabeller(var_name_map={"theta": r"$\theta$", "varphi": r"$\varphi$"})
    samples = ess.all_samples.T.reshape(ess.num_chains_, ess.num_samples, -1)
    theta_shape = ess.base_model.structure_lin.weight.shape
    varphi_dim = exp_col.num_bends - 1
    coords = {'theta_dim_0': np.arange(theta_shape[0]), 'theta_dim_1': np.arange(theta_shape[1]),
              'varphi_dim': np.arange(varphi_dim)}
    dims = {"theta": ['theta_dim_0', 'theta_dim_1'], "varphi": ['varphi_dim']}
    data = {'theta': samples[..., varphi_dim:].reshape(*samples.shape[:-1], *theta_shape),
            'varphi': samples[..., :varphi_dim]}
    az_inference_data = az.convert_to_inference_data(data, coords=coords, dims=dims)
    y_obs = xr.Dataset({'y_obs': ({'obs': ['obs']}, dataset[:][2].detach().cpu().numpy())},
                       coords={'obs': np.arange(num_data)})
    az_inference_data.add_groups(observed_data=y_obs)
    az_inference_data.add_groups(log_likelihood={'obs': logprobs.reshape(num_chains, num_samples)})

    if exp_col.num_bends == 3:
        # Subspace plot
        # Compute grid
        x = np.linspace(-10, 10, 40, dtype=np.float32)
        y = np.linspace(-8, 8, 40, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        grid = np.vstack([xx.flatten(), yy.flatten()]).T
        prior_dist = torch.distributions.Normal(0.,
                                                torch.tensor([5., 5.] + [1.] * len(true_thetas.flatten()),
                                                             device=device))
        log_prob_joint = []
        u_train, x_train, y_obs = exp_col.train_dataloader.dataset[:]
        # cond_theta = torch.from_numpy(true_thetas.flatten()).to(device=device)
        cond_theta = torch.from_numpy(
            az_inference_data.posterior['theta'].mean(['chain', 'draw']).to_numpy().flatten()).to(
            device=device, dtype=torch.float32)
        with torch.no_grad():
            for p in tqdm(grid):
                p_t = torch.from_numpy(p).to(device=device)
                p_t = torch.concat([p_t, cond_theta])
                subspace_model.set_parameter_vector(p_t)
                params = subspace_model(img=u_train, meta=x_train)
                ll = loss_fn(params, y_obs).sum()
                ll += prior_dist.log_prob(p_t).sum()
                # ll = dist.Normal(mu, 1.).log_prob(y_train)
                log_prob_joint.append(-ll.detach().cpu().numpy())
        log_prob_joint = np.array(log_prob_joint)
        log_prob_joint = np.nan_to_num(log_prob_joint)
        # df = pd.DataFrame({'log_prob_joint':log_prob_joint.sum(1)})
        df = pd.DataFrame({'log_prob_joint': log_prob_joint})
        df['xx'] = xx.flatten()
        df['yy'] = yy.flatten()
        # reconstruct Bezier curve
        w0 = torch.tensor([])
        w12 = torch.tensor([])
        w2 = torch.tensor([])
        for n, p in curve_model.named_parameters():
            if '_0' in n.split('.')[-1]:
                w0 = torch.hstack([w0, p.detach().cpu().clone().flatten()])
            elif '_1' in n.split('.')[-1]:
                w12 = torch.hstack([w12, p.detach().cpu().clone().flatten()])
            elif '_2' in n.split('.')[-1]:
                w2 = torch.hstack([w2, p.detach().cpu().clone().flatten()])
        # p_inv = np.linalg.pinv(cov.cpu().numpy().T)
        p_inv = subspace_model.cov_factor.cpu().numpy().T
        t0 = (w0.cpu().numpy() - subspace_model.mean.cpu().numpy()) @ p_inv
        t12 = (w12.cpu().numpy() - subspace_model.mean.cpu().numpy()) @ p_inv
        t2 = (w2.cpu().numpy() - subspace_model.mean.cpu().numpy()) @ p_inv
        fig = plot_subspace(df, "log_prob_joint", t0, t12, t2, linear_color=False, interpolate=False,
                            vmin=np.quantile(log_prob_joint, 0.8))
        post_varphi = az_inference_data['posterior']['varphi'].to_numpy().reshape(-1, 2)
        sns.scatterplot(x=post_varphi[:, 0], y=post_varphi[:, 1], alpha=0.2, linewidth=0., s=1.)
        ax = plt.gca()
        ax.get_legend().remove()
        ax.set_xlabel(r"$\varphi_1$")
        ax.set_ylabel(r"$\varphi_2$")
        wandb_logger.experiment.log({'Subspace plot': wandb.Image(plt.gcf())})
    return az_inference_data, wandb_logger


def run_hmc_subspace(exp_col, curve_model, curve_wandb_id, loss_fn, prior_scale, num_chains, num_warmup, num_samples,
                     **kwargs):
    # create subspace model
    subspace_model, wandb_logger = load_subspace_model(exp_col,
                                                       curve_model,
                                                       curve_wandb_id)
    subspace_model_ = deepcopy(subspace_model)

    # %% Run HMC on the subspace
    torch.set_default_dtype(torch.float32)
    pyro.clear_param_store()

    pyro_model = pyroSemiSubspace(
        mean=subspace_model_.mean.to(dtype=torch.float32),
        cov_factor=subspace_model_.cov_factor.to(dtype=torch.float32),
        sequential_dnn=subspace_model_.dnn,
        module_structure=subspace_model_.structure_lin,
        outcome_dist=loss_fn.dist_)

    pyro.set_rng_seed(8)
    u_train, x_train, y_train = exp_col.train_dataloader.dataset[:]
    nuts_kernel = pyro.infer.NUTS(pyro_model,
                                  jit_compile=False,
                                  adapt_step_size=True,
                                  step_size=1e-5,
                                  target_accept_prob=0.6,
                                  init_strategy=init_to_sample)
    mcmc = pyro.infer.MCMC(nuts_kernel,
                           num_samples=num_samples,
                           warmup_steps=num_warmup,
                           num_chains=num_chains,
                           mp_context='spawn')
    mcmc.run(u_train, x_train, y_train)
    wandb_logger.experiment.config.update({'num_chains': num_chains,
                                           'num_warmup': num_warmup,
                                           'num_samples': num_samples,
                                           'prior_scale': prior_scale,
                                           'seed': exp_col.seed})
    # create arviz inference object
    az_post_hmc = az.from_pyro(mcmc, log_likelihood=False)
    az_post_hmc = az_post_hmc.rename({
        'structure_nn.weight': 'theta',
        'structure_nn.weight_dim_0': 'theta_dim_0',
        'structure_nn.weight_dim_1': 'theta_dim_1'})

    # logprobs = ess.fit(dataset=cuda_loader_no_shuffle)
    #
    # compute predictive performance
    u_valid, x_valid, y_valid = exp_col.valid_dataloader.dataset[:]
    lppd_valid = log_pointwise_predictive_likelihood(pyro_model, mcmc.get_samples(), u=u_valid, x=x_valid, y=y_valid)
    lppd_valid = (torch.logsumexp(lppd_valid, dim=0) - np.log(lppd_valid.shape[0])).sum()
    wandb_logger.experiment.summary["valid_lppd"] = lppd_valid

    u_test, x_test, y_test = exp_col.test_dataloader.dataset[:]
    lppd_test = log_pointwise_predictive_likelihood(pyro_model, mcmc.get_samples(), u=u_test, x=x_test, y=y_test)
    lppd_test = (torch.logsumexp(lppd_test, dim=0) - np.log(lppd_test.shape[0])).sum()
    wandb_logger.experiment.summary["test_lppd"] = lppd_test

    if exp_col.num_bends == 3:
        plot_subspace_solution_regression_pyro(
            az_post_hmc = az_post_hmc,
            pyro_model = pyro_model,
            dataset = exp_col.train_dataloader.dataset,
            curve_model = curve_model,
            mean = subspace_model_.mean,
            cov = subspace_model_.cov_factor
        )
        wandb_logger.experiment.log({'Subspace plot': wandb.Image(plt.gcf())})
    return az_post_hmc, wandb_logger


def link_scale(x):
    return torch.exp(x) + 1e-30


def run_exp(seed, dir, p_struct, num_data, dist, lr, device, use_hmc, num_bends, max_epochs, **kwargs):
    # Hyperparameters for grid search
    # p_struct = 3  # number structured parameters
    # n = 75  # number of data points "range([75,500,2000])"
    # tau = 0.5  # ratio between structure and unstructure "range([0, 0.5, 2])"

    prior_scale = 1.
    if dist == 'normal':
        outcome_dist = pyro.distributions.Normal  # outcome distributions
        loss_fn = NllLoss(outcome_dist,
                          constrains={'loc': torch.nn.Identity(), 'scale': link_scale},
                          reduction='mean')
        base_net_kwargs_g = {"dimensions": [4, 4],
                             "output_dim": 2,
                             "input_dim": 4}
    elif dist == 'normal_mu':
        outcome_dist = pyro.distributions.Normal  # outcome distributions
        loss_fn = NllLoss(outcome_dist,
                          constrains={'loc': torch.nn.Identity()},
                          scale=1.,
                          reduction='mean')
        base_net_kwargs_g = {"dimensions": [16, 16],
                             "output_dim": 1,
                             "input_dim": 4}
    elif dist == 'poisson':
        outcome_dist = pyro.distributions.Poisson  # outcome distributions
        loss_fn = NllLoss(outcome_dist,
                          constrains={'rate': link_scale},
                          reduction='mean')
        base_net_kwargs_g = {"dimensions": [16, 16],
                             "output_dim": 1,
                             "input_dim": 4}
    else:
        raise f"Distribution {dist} is not supported"
    use_ortho = False

    # data generation
    loaders, true_thetas, true_model = data_generation_simulation1(num_data, p_struct, loss_fn.dist_,
                                                                   base_net_kwargs_g,
                                                                   seed=seed,
                                                                   dir_to_save=dir,
                                                                   prior_dist=torch.distributions.Normal(0,
                                                                                                         prior_scale),
                                                                   device=device)
    net_kwargs = dict(lr=lr,
                      weight_decay=1e-3,
                      loss_fn=loss_fn,
                      num_structure=p_struct,
                      ortho_layer_name_nn_head="lin[4]" if use_ortho else None,
                      metric_collection=MetricCollection([])
                      )
    exp_col = expCollector(wandb_project=wandb_project,
                           use_ortho=use_ortho,
                           seed=seed,
                           base_net_kwargs=base_net_kwargs_g,
                           net_kwargs=net_kwargs,
                           num_bends=num_bends,
                           max_epochs=max_epochs,
                           nll_fn=loss_fn)
    exp_col.train_dataloader = loaders[0]
    exp_col.valid_dataloader = loaders[1]
    exp_col.test_dataloader = loaders[2]

    # initialize subspace model
    curve_model, curve_wandb_id, fix_points_wandb_id, wandb_logger = initialize_subspace_model_v2(
        exp_col=exp_col,
        plot_predictive_f=lambda *x, **xx: None,
        plot_curve_solution_f=plot_curve_solution)
    print("*" * 27)
    print("* Initialisation finished *")
    print("*" * 27)
    art = wandb.Artifact(f"data_{wandb_logger.experiment.id}", type="data",
                         description="data, ground truth model and state_dict")
    art.add_file(os.path.join(dir, "dataset_simulation.pt"))
    wandb_logger.experiment.log_artifact(art)
    wandb_logger.experiment.config.update({'dist': dist,
                                           'Subspace_dimension': exp_col.num_bends - 1})
    wandb.finish()

    # Run MCMC
    if use_hmc:
        az_inference_data, wandb_logger = run_hmc_subspace(exp_col=exp_col,
                                                           curve_model=curve_model,
                                                           curve_wandb_id=curve_wandb_id,
                                                           loss_fn=loss_fn,
                                                           prior_scale=prior_scale,
                                                           **kwargs)
    else:
        az_inference_data, wandb_logger = run_ess(exp_col=exp_col,
                                                  curve_model=curve_model,
                                                  curve_wandb_id=curve_wandb_id,
                                                  loss_fn=loss_fn,
                                                  prior_scale=prior_scale,
                                                  num_data=num_data,
                                                  **kwargs)
    wandb_logger.experiment.config.update({'dist': dist,
                                           'Subspace_dimension': exp_col.num_bends - 1})
    # save samples with wandb
    fname = os.path.join(dir, "az_subspace_posterior.nc")
    az_inference_data.to_netcdf(fname)
    art = wandb.Artifact(f"data_{wandb_logger.experiment.id}", type="xarray",
                         description="posterior from subspace model")
    art.add_file(fname)
    wandb_logger.experiment.log_artifact(art)

    # Description
    labeller = azl.MapLabeller(var_name_map={"theta": r"$\theta$", "varphi": r"$\varphi$"})
    summary_subspace = az.summary(az_inference_data)
    wandb_logger.experiment.log({"description": wandb.Table(dataframe=summary_subspace.reset_index())})
    max_r_hat = az.rhat(az_inference_data, var_names='theta')['theta'].to_numpy().max()
    wandb_logger.experiment.summary["r_hat_max_structure"] = max_r_hat
    r_hat_smaller_1_1 = max_r_hat < 1.1
    wandb_logger.experiment.summary["r_hat<1.1"] = r_hat_smaller_1_1

    # coverage comparison
    axes = az.plot_posterior(az_inference_data, var_names=("theta",), labeller=labeller,
                             backend_kwargs={'layout': 'tight'})
    for i, ax in enumerate(axes.flat):
        ax.vlines(true_thetas.flatten()[i], ymin=0, ymax=ax.get_ylim()[1], color="red", label=rf"$\Theta_{i}$")
    wandb_logger.experiment.log({'Coverage_comparison': wandb.Image(plt.gcf())})

    # Trace plot
    az.plot_trace(az_inference_data, compact=False, legend=False, labeller=labeller,
                  backend_kwargs={'layout': 'tight'})
    wandb_logger.experiment.log({'Trace plot': wandb.Image(plt.gcf())})
    wandb.finish()
    return true_thetas, exp_col, curve_wandb_id, az_inference_data

def run_hmc(seed, dir, dist, num_warmup, num_chains, num_samples, exp_col, curve_wandb_id, **kwargs):
    # -> data preparation <-
    unp, xnp, ynp = exp_col.train_dataloader.dataset[:]
    unp = jnp.array(unp.cpu().numpy().astype(np.float32))
    xnp = jnp.array(xnp.cpu().numpy().astype(np.float32))
    ynp = jnp.array(ynp.cpu().numpy().astype(np.float32))

    dist_kwargs = {}
    if dist == 'normal':
        outcome_dist = np_dist.Normal  # outcome distributions
        constrain_fn = {'loc': lambda x: x, 'scale': jnp.exp}
        base_net_kwargs_g = {"dimensions": [4, 4],
                             "output_dim": 2,
                             "input_dim": 4}
    elif dist == 'normal_mu':
        outcome_dist = np_dist.Normal  # outcome distributions
        constrain_fn = {'loc': lambda x: x}
        dist_kwargs = {'scale': 1.}
        base_net_kwargs_g = {"dimensions": [16, 16],
                             "output_dim": 1,
                             "input_dim": 4}
    elif dist == 'poisson':
        outcome_dist = np_dist.Poisson  # outcome distributions
        constrain_fn = {'rate': jnp.exp}
        base_net_kwargs_g = {"dimensions": [16, 16],
                             "output_dim": 1,
                             "input_dim": 4}
    else:
        raise f"Distribution {dist} is not supported"

    # -> inference loop <-
    model = NumpyroModel(dist=outcome_dist,
                         base_net_kwargs=base_net_kwargs_g,
                         constrains=constrain_fn,
                         **dist_kwargs)
    prng = jax.random.PRNGKey(seed + 1)
    kernel = numpyro.infer.NUTS(model,
                                step_size=1e-5,
                                adapt_step_size=True,
                                adapt_mass_matrix=True,
                                dense_mass=True,
                                init_strategy=np_init_to_sample,
                                target_accept_prob=0.6)
    mcmc = numpyro.infer.MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="parallel",
        progress_bar=True,
    )
    # mcmc.warmup(prng, unp, xnp, ynp, extra_fields=("diverging", "adapt_state.step_size", 'accept_prob','potential_energy', 'energy', 'r'), collect_warmup=True)
    # mcmc.run(prng, unp, xnp, ynp, extra_fields=("diverging", "adapt_state.step_size", 'accept_prob','potential_energy', 'energy', 'r', "num_steps"))
    mcmc.run(prng, unp, xnp, ynp, extra_fields=['energy', "num_steps"])

    # -> analyze results <-
    posterior_samples = mcmc.get_samples()
    posterior_predictive = numpyro.infer.Predictive(model, posterior_samples)(
        jax.random.PRNGKey(1), unp, xnp
    )
    prior = numpyro.infer.Predictive(model, num_samples=500)(
        jax.random.PRNGKey(2), unp, xnp
    )
    mcmc_data = az.from_numpyro(
        mcmc,
        prior=prior,
        posterior_predictive=posterior_predictive,
        log_likelihood=False
    )
    # save in wandb
    wandb_logger = WandbLogger(project=exp_col.wandb_project, log_model=False,
                               name=f'hmc_{curve_wandb_id}',
                               group="HMC", resume='never')
    fname = os.path.join(dir, "az_hmc_posterior.nc")
    mcmc_data.to_netcdf(fname)
    art = wandb.Artifact(f"data_{wandb_logger.experiment.id}", type="xarray",
                         description="posterior, posterior predictive and prior predictive with HMC\nReload with from_netcdf('fname')")
    art.add_file(fname)
    wandb_logger.experiment.log_artifact(art)

    # compute valid lppd
    unp_t, xnp_t, ynp_t = exp_col.valid_dataloader.dataset[:]
    unp_t = jnp.array(unp_t.cpu().numpy().astype(np.float32))
    xnp_t = jnp.array(xnp_t.cpu().numpy().astype(np.float32))
    ynp_t = jnp.array(ynp_t.cpu().numpy().astype(np.float32))
    log_likelihood = numpyro.infer.util.log_likelihood(handlers.seed(model, prng), posterior_samples, unp_t, xnp_t,
                                                       y=ynp_t)
    lppd = (logsumexp(log_likelihood['y'], axis=0) - np.log(log_likelihood['y'].shape[0])).sum()
    wandb_logger.experiment.summary["valid_lppd"] = lppd.item()

    # compute test lppd
    unp_t, xnp_t, ynp_t = exp_col.test_dataloader.dataset[:]
    unp_t = jnp.array(unp_t.cpu().numpy().astype(np.float32))
    xnp_t = jnp.array(xnp_t.cpu().numpy().astype(np.float32))
    ynp_t = jnp.array(ynp_t.cpu().numpy().astype(np.float32))
    log_likelihood = numpyro.infer.util.log_likelihood(handlers.seed(model, prng), posterior_samples, unp_t, xnp_t,
                                                       y=ynp_t)
    lppd = (logsumexp(log_likelihood['y'], axis=0) - np.log(log_likelihood['y'].shape[0])).sum()
    wandb_logger.experiment.summary["test_lppd"] = lppd.item()

    # plot posterior with ground truth
    labeller = azl.MapLabeller(var_name_map={"theta": r"$\theta$", "varphi": r"$\varphi$"})
    axes = az.plot_posterior(mcmc_data, var_names=("theta",), labeller=labeller,
                             backend_kwargs={'layout': 'tight'})
    for i, ax in enumerate(axes.flat):
        ax.vlines(true_thetas.flatten()[i], ymin=0, ymax=ax.get_ylim()[1], color="red", label=rf"$\Theta_{i}$")
    wandb_logger.experiment.log({'Coverage_comparison': wandb.Image(plt.gcf())})

    # plot trace plot
    az.plot_trace(mcmc_data, var_names=("theta"), compact=False, labeller=labeller,
                  backend_kwargs={'layout': 'tight'})
    wandb_logger.experiment.log({'Trace plot': wandb.Image(plt.gcf())})

    # Description
    summary_subspace = az.summary(mcmc_data)
    wandb_logger.experiment.log({"description": wandb.Table(dataframe=summary_subspace.reset_index())})
    max_r_hat = az.rhat(mcmc_data, var_names='theta')['theta'].to_numpy().max()
    wandb_logger.experiment.summary["r_hat_max_structure"] = max_r_hat
    r_hat_smaller_1_1 = max_r_hat < 1.1
    wandb_logger.experiment.summary["r_hat<1.1"] = r_hat_smaller_1_1
    wandb_logger.experiment.config.update({'num_chains': num_chains,
                                           'num_warmup': num_warmup,
                                           'num_samples': num_samples,
                                           'seed': exp_col.seed,
                                           'dist': dist,
                                           'Subspace_dimension': exp_col.num_bends - 1})
    mcmc.print_summary()
    return mcmc, mcmc_data, wandb_logger


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run coverage simulation')
    parser.add_argument('-d', '--dir', type=str, default='./simulation/results/', metavar='DIR',
                        help='result directory (default: ./simulation/results/)')
    parser.add_argument('-n', '--num_data', type=int, default=75, help='Number of datapoints')
    parser.add_argument('--num_runs', type=int, default=3, help='Number of repeated runs')
    parser.add_argument('--p_struct', type=int, default=3,
                        help='Number of structured parameters per distribution parameter')
    # parser.add_argument('--tau', type=float, default=0.5, help='Fraction between DNN and structured model')
    parser.add_argument('--dist', type=str, default='normal',
                        help="Output distribution either 'normal' or 'poisson' supported")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--wandb_project', type=str, default='semiSub_simulation1_coverage', help="WandB project name")
    parser.add_argument('--num_chains', type=int, default=40, help="number of markov chains to draw")
    parser.add_argument('--num_samples', type=int, default=2000, help="number of ESS samples to draw")
    parser.add_argument('--num_warmup', type=int, default=200, help="number of warmup samples to draw")
    parser.add_argument('--max_epochs', type=int, default=1000, help="number of epochs")
    parser.add_argument('--num_bends', type=int, default=3, help="number of bendpoints")
    parser.add_argument('--use_hmc', action='store_true', help="Use HMC to get the posterior from the subspace")
    args = parser.parse_args()
    os.makedirs(args.dir, exist_ok=True)

    wandb_project = args.wandb_project
    use_ortho = False
    seed = 0
    # style stuff ->
    sns.set_style("darkgrid")
    if torch.cuda.is_available() and 1:
        print("Use cuda. Device: ", torch.cuda.get_device_name(torch.cuda.current_device()))
        device = torch.device('cuda', torch.cuda.current_device())
    else:
        device = torch.device('cpu')
    print("Device: ", device)
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": False,
        # "text.latex.preamble": r'\usepackage{amsfonts}',
        "font.family": "Nimbus Sans",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 24,
        "font.size": 24,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 18,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24
    }
    plt.rcParams.update(tex_fonts)

    # run exp ->
    df_ci = []
    df_ci_hmc = []
    for i in trange(args.num_runs, desc="Runs: "):
        # try:
        true_thetas, exp_col, curve_wandb_id, az_subspace_data = run_exp(seed=i, **vars(args), device=device)
        theta_samples = az_subspace_data.posterior['theta'].to_numpy()  # only with structure parameters (thetas)
        theta_samples = theta_samples.reshape(np.prod(theta_samples.shape[:2]), -1)
        alphas = np.arange(0.05, 1., 0.025)
        lower_q = np.quantile(theta_samples, (1 - alphas) / 2, axis=0)
        upper_q = np.quantile(theta_samples, 1 - (1 - alphas) / 2, axis=0)
        in_ci = np.greater_equal(true_thetas.flatten(), lower_q) & np.greater_equal(upper_q, true_thetas.flatten())
        alphas = alphas[:, None]
        df_ci.append(np.hstack([np.full_like(alphas, i), alphas, in_ci]))
        # run HMC

        mcmc, mcmc_data, wandb_logger = run_hmc(**vars(args), seed=i, curve_wandb_id=curve_wandb_id, exp_col=exp_col)
        samples = mcmc.get_samples()['theta']  # only with structure parameters (thetas)
        alphas = np.arange(0.05, 1., 0.025)
        lower_q = np.quantile(samples, (1 - alphas) / 2, axis=0)
        upper_q = np.quantile(samples, 1 - (1 - alphas) / 2, axis=0)
        in_ci = np.greater_equal(true_thetas, lower_q) & np.greater_equal(upper_q, true_thetas)
        in_ci = in_ci.reshape(*in_ci.shape[:-2], -1)  # flatten of last two dimension
        alphas = alphas[:, None]
        df_ci_hmc.append(np.hstack([np.full_like(alphas, i), alphas, in_ci.squeeze()]))

        # forest plot compare
        labeller = azl.MapLabeller(var_name_map={"theta": r"$\theta$", "varphi": r"$\varphi$"})
        az.plot_forest([mcmc_data, az_subspace_data], model_names=['HMC', 'Subspace'], var_names=("theta"),
                       figsize=(8, 8), labeller=labeller, backend_kwargs={'dpi': 150, 'layout': 'tight'})
        wandb_logger.experiment.log({'Forest plot': wandb.Image(plt.gcf())})
        wandb.finish()

        # except Exception as e:
        #     print(e)
        #     print(f"A Problem encountered in run {i}")

    run = wandb.init(entity="ddold", project=args.wandb_project, name="qq_plot")

    structure_labels = ["$\\theta_{" + str(ss_p) + "}$" for ss_p in range(len(true_thetas.flatten()))]
    df_ci = pd.DataFrame(np.vstack(df_ci), columns=['run', 'alpha'] + structure_labels)
    fname = os.path.join(args.dir, "df_theta_in_ci.pkl")
    df_ci.to_pickle(fname)
    w_table = wandb.Table(dataframe=df_ci)
    run.log({"df_theta_in_ci": w_table})

    df_ci_hmc = pd.DataFrame(np.vstack(df_ci_hmc), columns=['run', 'alpha'] + structure_labels)
    fname = os.path.join(args.dir, "df_theta_in_ci_hmc.pkl")
    df_ci_hmc.to_pickle(fname)
    w_table = wandb.Table(dataframe=df_ci_hmc)
    run.log({"df_theta_hmc_in_ci": w_table})

    df_ci_ = df_ci.copy(deep=True)
    df_ci_ = df_ci_.groupby('alpha').mean()
    df_ci_.drop(columns='run', inplace=True)
    df_ci_ = df_ci_.reset_index().melt(id_vars=['alpha'], var_name='parameter')
    fig = plt.figure(figsize=(8., 8.), dpi=100)
    ax = sns.lineplot(df_ci_, x='alpha', y='value', hue='parameter', markers=True, dashes=False)
    ax.set_xlim(0, 1.)
    ax.set_ylim(0, 1.)
    ax.plot([0, 1], [0, 1], '--k', alpha=0.8)
    run.log({'qq-plot true_value in posterior Subspace': wandb.Image(plt.gcf())})

    df_ci_hmc_ = df_ci_hmc.copy(deep=True)
    df_ci_hmc_ = df_ci_hmc_.groupby('alpha').mean()
    df_ci_hmc_.drop(columns='run', inplace=True)
    df_ci_hmc_ = df_ci_hmc_.reset_index().melt(id_vars=['alpha'], var_name='parameter')
    fig = plt.figure(figsize=(8., 8.), dpi=100)
    ax = sns.lineplot(df_ci_hmc_, x='alpha', y='value', hue='parameter', markers=True, dashes=False)
    ax.set_xlim(0, 1.)
    ax.set_ylim(0, 1.)

    ax.plot([0, 1], [0, 1], '--k', alpha=0.8)
    run.log({'qq-plot true_value in posterior HMC': wandb.Image(plt.gcf())})
    wandb.finish()
