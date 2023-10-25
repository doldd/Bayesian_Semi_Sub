# %%
import torch
import torchvision
from torch import nn
import wandb
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import pyro
from src.semi_sub_utils import expCollector
from src.plot import plot_curve_solutions_regression, plot_predictive_regression
from src.semi_subspace import initialize_subspace_model_v2
from utils_datamodel.utils import FastFillTensorDataLoader
from torch.utils.data import random_split
from baselines.src.datasets import DatasetFactory
from torchmetrics import MetricCollection, MeanAbsoluteError, MeanSquaredError, MeanMetric
from src.base_models import NllLoss
import matplotlib.pyplot as plt
import numpy as np
from src.model import getModel, RegNet
from src.semi_sub_utils import span_space_from_curve_model
from src.plot import plot_subspace
from tqdm import tqdm
import pandas as pd
from src.semi_sub_utils import log_pointwise_predictive_likelihood
from src.model import pyroSubspaceUCI
from src.plot import plot_subspace_solution_regression_pyro
from pyro.infer.autoguide import init_to_sample
import arviz as az
from pyro.poutine.indep_messenger import IndepMessenger
import seaborn as sns
from src.semi_subspace import load_subspace_model
from copy import deepcopy
from baselines.src.datasets import DATASETS_BENCHMARK, DATASETS_TOY
from src.pyro_models import pyroSubspaceUCI_Zt
import json
from pyro import poutine

if torch.cuda.is_available() and 1:
    print("Use cuda. Device: ", torch.cuda.get_device_name(torch.cuda.current_device()))
    device = torch.device('cuda', torch.cuda.current_device())
else:
    device = torch.device('cpu')
print("Device: ", device)

def plot_curve_solutions_uci_regression(exp_col: expCollector, best_curve_model, wandb_logger):
    plt.figure(figsize=(9., 7.))
    # %% plot performance along curve
    dataset_test = exp_col.test_dataloader.dataset
    x_train, y_train = dataset_test[:]
    device = x_train.device
    best_curve_model = best_curve_model.to(device=device)
    t_space = torch.linspace(0, 1., 101, device=device)
    nll_vs_t = []
    for t in t_space:
        out = best_curve_model.model(x_train, t)
        nll = exp_col.nll_fn(out, y_train.squeeze(-1)).mean()
        nll_vs_t.append(nll.detach().cpu().item())
    plt.plot(t_space.cpu().numpy(), nll_vs_t)
    plt.xlabel("Bézier curve t-space")
    plt.ylabel(r'nll ~ $N(y|\mu=DNN(),\sigma=0.005)$')
    wandb_logger.experiment.log({"Bézier Curve NLL": wandb.Image(plt.gcf())})

    # %% plot subspace
    mean, cov, S = span_space_from_curve_model(best_curve_model.model.cpu(), exp_col.num_bends)
    subspace_model = getModel(RegNet, mean=mean, cov_factor=cov, **exp_col.base_net_kwargs, **exp_col.net_kwargs).to(
        device=device)
    metric_collection_base = MetricCollection([MeanSquaredError(squared=False),
                                               MeanAbsoluteError(),
                                               ]).to(device=device)
    # do grid for the first two dimensions of the subspace
    x = np.linspace(-6, 6, 40, dtype=np.float32)
    y = np.linspace(-6, 6, 40, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    grid = np.vstack([xx.flatten(), yy.flatten()]).T
    all_metrics = []
    metric_collection = metric_collection_base.clone()
    with torch.no_grad():
        for p in tqdm(grid):
            metric_collection.reset()
            p_t = torch.from_numpy(p).to(device=device)
            p_t = torch.concat([p_t, torch.zeros(exp_col.num_bends - 3, device=device)])
            subspace_model.set_parameter_vector(p_t)
            nll = 0.
            for data, y in exp_col.train_dataloader:
                # y = y.cuda() if y.device is not device else y
                p_pred = subspace_model(data)
                # nll += subspace_model.loss_fn(p_pred, y).item()
                nll += exp_col.nll_fn(p_pred, y.squeeze(-1)).mean().item()
                metric_collection.update(exp_col.nll_fn.dist_(p_pred).mean, y.squeeze(-1))
            metrics = metric_collection.compute()
            metrics['nll'] = nll / len(exp_col.train_dataloader)
            all_metrics.append(metrics)
    df_grid = pd.DataFrame(all_metrics)
    df_grid['MeanSquaredError'] = df_grid['MeanSquaredError'].apply(lambda x: x.item())
    df_grid['MeanAbsoluteError'] = df_grid['MeanAbsoluteError'].apply(lambda x: x.item())
    df_grid['xx'] = xx.flatten()
    df_grid['yy'] = yy.flatten()
    all_curve_params = [np.array([])] * exp_col.num_bends
    for n, p in best_curve_model.named_parameters():
        control_point_i = n.split('.')[-1]
        if '_' in control_point_i:
            control_point_i = int(control_point_i.split('_')[1])
            all_curve_params[control_point_i] = np.hstack(
                [all_curve_params[control_point_i], p.detach().clone().flatten().numpy()])
    all_curve_params = torch.as_tensor(all_curve_params)
    p_inv = np.linalg.pinv(cov.cpu().numpy().T)
    cp = (all_curve_params.cpu().numpy() - mean.cpu().numpy()) @ p_inv.T  # control points in the subspace
    fig = plot_subspace(df_grid, "nll", cp[0, :2], cp[1:-1, :2], cp[-1, :2], linear_color=False, interpolate=True)
    wandb_logger.experiment.log({"train_grid_nll": wandb.Image(fig)})
    fig = plot_subspace(df_grid, "MeanSquaredError", cp[0, :2], cp[1:-1, :2], cp[-1, :2], linear_color=True,
                        interpolate=False)
    wandb_logger.experiment.log({"train_grid_MSE": wandb.Image(fig)})
    fig = plot_subspace(df_grid, "MeanAbsoluteError", cp[0, :2], cp[1:-1, :2], cp[-1, :2], linear_color=True,
                        interpolate=False)
    wandb_logger.experiment.log({"train_grid_MAE": wandb.Image(fig)})
    w_table = wandb.Table(dataframe=df_grid)
    wandb_logger.experiment.log({"train_grid": w_table})

def run_hmc_on_subspace_no_struct(num_chains, num_warmup, num_samples, prior_scale_subspace, exp_col:expCollector, curve_model, curve_wandb_id):
    # create subspace model
    subspace_model, wandb_logger = load_subspace_model(exp_col,
                                                       curve_model,
                                                       curve_wandb_id)
    subspace_model_ = deepcopy(subspace_model)

    # %% Run HMC on the subspace
    torch.set_default_dtype(torch.float32)
    pyro.clear_param_store()

    pyro_model = pyroSubspaceUCI(
        mean=subspace_model_.mean.to(dtype=torch.float32),
        cov_factor=subspace_model_.cov_factor.to(dtype=torch.float32),
        sequential_dnn=subspace_model_.dnn,
        outcome_dist=exp_col.nll_fn.dist_,
        prior_scale_subspace=prior_scale_subspace)
    print(pyro_model.device)

    pyro.set_rng_seed(exp_col.seed+3)
    u_train = torch.vstack([exp_col.train_dataloader.dataset[:][0], exp_col.valid_dataloader.dataset[:][0]])
    y_train = torch.vstack([exp_col.train_dataloader.dataset[:][1], exp_col.valid_dataloader.dataset[:][1]])
    nuts_kernel = pyro.infer.NUTS(pyro_model,
                                    jit_compile=False,
                                    adapt_step_size=True,
                                    step_size=1e-5,
                                    target_accept_prob=0.8,
                                    init_strategy=init_to_sample)
    mcmc = pyro.infer.MCMC(nuts_kernel,
                        num_samples=num_samples,
                        warmup_steps=num_warmup,
                        num_chains=num_chains,
                        mp_context='spawn')
    mcmc.run(u_train, y_train.squeeze())
    wandb_logger.experiment.config.update({'num_chains': num_chains,
                                            'num_warmup': num_warmup,
                                            'num_samples': num_samples,
                                            'prior_scale_subspace': prior_scale_subspace,
                                            'seed': exp_col.seed,
                                            'subspace_dimension': exp_col.num_bends - 1, 
                                            'dnn_large': True if len(exp_col.base_net_kwargs['dimensions']) > 1 else False})
    # create arviz inference object
    az_post_hmc = az.from_pyro(mcmc, log_likelihood=False)
    # save samples with wandb
    az_post_hmc.to_netcdf("az_subspace_posterior.nc")
    art = wandb.Artifact(f"data_{wandb_logger.experiment.id}", type="xarray",
                            description="posterior from subspace model")
    art.add_file("az_subspace_posterior.nc")
    wandb_logger.experiment.log_artifact(art)
    # save pyro model state dict
    torch.save(pyro_model.state_dict(), "model_state_dict.pt")
    art = wandb.Artifact(f"model_state_{wandb_logger.experiment.id}", type="pyroSemiSubspace",
                            description="pyro model state dict")
    art.add_file("model_state_dict.pt")
    wandb_logger.experiment.log_artifact(art)

    # compute test lppd
    if exp_col.test_dataloader is not None:
        u_test, y_test = exp_col.test_dataloader.dataset[:]
        lppd_test = log_pointwise_predictive_likelihood(pyro_model, mcmc.get_samples(), u=u_test, y=y_test.squeeze())
        lppd_test = (torch.logsumexp(lppd_test, dim=0) - np.log(lppd_test.shape[0])).sum()
        wandb_logger.experiment.summary["test_mlppd"] = lppd_test / len(y_test)

        # compute test lppd unnormalized
        with open('mu_sigma.json', 'r') as f:
            mu_Simga = json.load(f)
        mu = mu_Simga[ds_name]['mean_y'][0]
        sigma = np.sqrt(mu_Simga[ds_name]['var_y']).item()
        print(f"mu: {mu}, sigma:{sigma}")
        y_test_scale = y_test*sigma + mu

        num_samples = list(mcmc.get_samples().values())[0].shape[0]
        log_probs = []
        samples = [
            {k: v[i] for k, v in mcmc.get_samples().items()} for i in range(num_samples)
        ]
        pyro_model_z = pyroSubspaceUCI_Zt(mean=pyro_model.mean,
                                            cov_factor=pyro_model.cov,
                                            sequential_dnn=getModel(RegNet, **exp_col.base_net_kwargs, **exp_col.net_kwargs).to(device=pyro_model.device).dnn,
                                            z_mean=mu,
                                            z_scale=sigma)

        for i in range(num_samples):
            trace = poutine.trace(poutine.condition(pyro_model_z, samples[i])).get_trace(u=u_test, y=y_test_scale.squeeze())
            trace.compute_log_prob()
            log_probs.append(trace.nodes['obs']["log_prob"])
        ll_test_unscale = torch.stack(log_probs)
        lppd_test_unscale = (torch.logsumexp(ll_test_unscale, dim=0) - np.log(ll_test_unscale.shape[0])).sum()
        wandb_logger.experiment.summary["test_mlppd_unscaled"] = lppd_test_unscale / len(y_test)

    # compute grid
    if exp_col.num_bends == 3:
        # compute grid
        device = pyro_model.device
        x = np.linspace(-10, 10, 40, dtype=np.float32)
        y = np.linspace(-8, 8, 40, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        grid = np.vstack([xx.flatten(), yy.flatten()]).T
        with IndepMessenger("grid", size=grid.shape[0], dim=-2):
            cond_model = pyro.condition(pyro_model, data={"varphi": torch.from_numpy(grid).to(device=device)})
            trace = pyro.poutine.trace(cond_model).get_trace(u_test, y_test.squeeze(-1))
            trace.compute_log_prob()
        log_like = trace.nodes['obs']['log_prob'].sum(1).detach().cpu().numpy()
        log_prob_joint = log_like.copy()
        # log_prob_joint += trace.nodes['structure_nn.weight']['log_prob'].item()  # wasn't broadcasted => single value
        log_prob_joint += trace.nodes['varphi']['log_prob'].detach().cpu().numpy()
        log_prob_joint = np.nan_to_num(log_prob_joint, nan=np.nan_to_num(-np.inf))
        df = pd.DataFrame.from_dict(dict(xx=xx.flatten(),
                                            yy=yy.flatten(),
                                            log_like=log_like,
                                            log_prob_joint=log_prob_joint))
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
        p_inv = pyro_model.cov.detach().cpu().numpy().T
        t0 = (w0.cpu().numpy() - pyro_model.mean.detach().cpu().numpy()) @ p_inv
        t12 = (w12.cpu().numpy() - pyro_model.mean.detach().cpu().numpy()) @ p_inv
        t2 = (w2.cpu().numpy() - pyro_model.mean.detach().cpu().numpy()) @ p_inv
        fig = plot_subspace(df, "log_prob_joint", t0, t12, t2, linear_color=False, interpolate=False,
                            vmin=np.quantile(log_prob_joint, 0.8))
        post_varphi = az_post_hmc['posterior']['varphi'].to_numpy().reshape(-1, 2)
        sns.scatterplot(x=post_varphi[:, 0], y=post_varphi[:, 1], alpha=np.min((0.75, 100./post_varphi.shape[0])), linewidth=0., s=3)
        ax = plt.gca()
        ax.get_legend().remove()
        ax.set_xlabel(r"$\varphi_1$")
        ax.set_ylabel(r"$\varphi_2$")
        wandb_logger.experiment.log({'Subspace plot': wandb.Image(plt.gcf())})
    wandb_logger.experiment.config['dataset'] = ds_name
    wandb.finish()
    return az_post_hmc, mcmc, pyro_model


if '__main__' == __name__:
    for ds_name in DATASETS_TOY + DATASETS_BENCHMARK:
        # load dataset
        if ds_name in DATASETS_BENCHMARK:
            split_file = 'baselines/data/dataset_indices_0.2.json'
        else:
            split_file = 'baselines/data/toy_dataset_indices_0.2.json'
        data_train, data_test = DatasetFactory.get(ds_name, splits=split_file, dataset_pth='baselines/data/', device=device)
        
        for nn_size_large in [False, True]:
        # define hyper parameters
            if nn_size_large:
                base_net_kwargs = {"dimensions": [16, 16, 16],
                           "output_dim": 1,
                           "input_dim": data_train.n_features}
            else:
                base_net_kwargs = {"dimensions": [3],
                                "output_dim": 1,
                                "input_dim": data_train.n_features}
            outcome_dist = pyro.distributions.Normal  # outcome distributions
            loss_fn = NllLoss(outcome_dist,
                            constrains={'loc': torch.nn.Identity()},
                            reduction='mean')
            loss_fn.sigma = torch.nn.Parameter(torch.ones(1)*0.01) # register parameter
            loss_fn.dist_.dist_kwargs = {'scale': loss_fn.sigma} # use sigma parameter in distribution
            net_kwargs = dict(lr=5e-3,
                            weight_decay=0.,
                            loss_fn=loss_fn,
                            num_structure=0,
                            activation='tanh',
                            ortho_layer_name_nn_head=None
                            )
            seed = torch.randint(0,1000,(1,)).item()
            # define metrics for val and test data
            metric_col = MetricCollection(MeanAbsoluteError(), MeanSquaredError(squared=False))
            for num_subspace_dim in [2,5]:
                # collect all parameters
                exp_col = expCollector(wandb_project='uci_reg_benchmark',
                                    use_ortho=False,
                                    seed=seed,
                                    base_net_kwargs=base_net_kwargs,
                                    net_kwargs=net_kwargs,
                                    nll_fn=loss_fn,
                                    max_epochs=1500,
                                    num_bends=num_subspace_dim+1,
                                    metric_collection=metric_col)

                # define train val and test dataloader
                prop_val = 0.3
                val_set_size = int(len(data_train) * prop_val)
                train_set_size = len(data_train) - val_set_size
                train_set, val_set = random_split(
                    data_train, [train_set_size, val_set_size], generator=torch.Generator().manual_seed(seed)
                )
                exp_col.train_dataloader  = FastFillTensorDataLoader(train_set, batch_size=len(train_set), shuffle=True,
                                                    pin_memory=False)
                exp_col.valid_dataloader  = FastFillTensorDataLoader(val_set, batch_size=len(val_set), shuffle=False,
                                                    pin_memory=False)
                exp_col.test_dataloader  = FastFillTensorDataLoader(data_test, batch_size=len(data_test), shuffle=False,
                                                            pin_memory=False)

                # %%
                # define Projection matrix (Train curve model)
                best_curve_model, wandb_curve_exp_id, _, wandb_logger = initialize_subspace_model_v2(
                    exp_col=exp_col,
                    plot_predictive_f=lambda *x, **xargs: None,
                    plot_curve_solution_f=plot_curve_solutions_uci_regression)
                wandb_logger.experiment.config['base_net_kwargs'] = base_net_kwargs
                wandb_logger.experiment.config['net_kwargs'] = net_kwargs
                wandb_logger.experiment.config['dataset'] = ds_name

                # %%
                trainer = pl.Trainer(devices=1)
                trainer.test(best_curve_model, dataloaders=exp_col.test_dataloader)
                print("*" * 27)
                print("* Initialisation finished *")
                print("*" * 27)
                wandb.finish()

                # %% [markdown]
                # # lppd
                # lppd = $\sum^n_{i=1} log (\frac{1}{S} \sum^S_{s=1}p(y_i|\theta_s))$
                # 
                # lppd = $\sum^n_{i=1} log (\frac{1}{S} \sum^S_{s=1} e^{log(p(y_i|\theta_s)}))$
                # 
                # lppd = $\sum^n_{i=1} log (\sum^S_{s=1} e^{log(p(y_i|\theta_s)}) - log(S))$
                # 
                # 


                # %%
                az_post_hmc, mcmc, pyro_model = run_hmc_on_subspace_no_struct(num_chains=10, 
                                                                num_warmup=200, 
                                                                num_samples=600, 
                                                                prior_scale_subspace=1.,
                                                                exp_col=exp_col,
                                                                curve_model=best_curve_model,
                                                                curve_wandb_id=wandb_curve_exp_id)
                # %%
                print(az.summary(az_post_hmc))


