import torch
import torch.nn as nn
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
from pytorch_lightning.callbacks import TQDMProgressBar

from src.base_models import RegNet
from src.semiSub_model import getModel
from src.curve_models import CurveArchitectureFromSequential, CurveModel
from src.pyro_models import pyroSemiSubspace

from src.semi_sub_utils import features, get_ds_test_from_df, get_ds_from_df, expCollector, log_pointwise_predictive_likelihood, get_curve_space_torch, span_space_from_curve_model
from src.plot import plot_subspace, exclude_project_code_dirs, run_metrics_on_samples
from src.plot import plot_predictive_regression, plot_curve_solutions_regression, plot_subspace_solution_regression, plot_data, plot_subspace_solution_regression_pyro
import os
from torchmetrics import MeanSquaredError, MeanAbsoluteError
from tqdm.notebook import tqdm
from functools import partial
import torch.distributions as dist
from sklearn.model_selection import StratifiedKFold
from numpyro.diagnostics import summary
from numpy.random import default_rng
from utils_datamodel.utils import FastFillTensorDataLoader
from copy import deepcopy
import pyro
from pyro.infer.autoguide import init_to_sample
import arviz as az



__all__ = ['prepare_data', 'initialize_subspace_model', 'initialize_subspace_model_v2', 'load_subspace_model',
           'NoValProgressBar', 'run_hmc_on_subspace']

# global vars
sns.set_style("darkgrid")
if torch.cuda.is_available() and 1:
    print("Use cuda. Device: ", torch.cuda.get_device_name(torch.cuda.current_device()))
    device = torch.device('cuda', torch.cuda.current_device())
else:
    device = torch.device('cpu')
print("Device: ", device)


def gaussian_nll(loc, y, scale=0.05):
    return -dist.Normal(loc, scale).log_prob(y)


def prepare_data(exp_col: expCollector, reload=False):
    if reload:
        df = pd.read_csv('data_cat_train.csv', header=[0, 1], index_col=[0])
        df_test = pd.read_csv('data_cat_test.csv', header=[0, 1], index_col=[0])
        # df = pd.read_csv('data_cat.csv', header=[0,1], index_col=[0])
    else:
        # raise Exception("Do you really wand to generate new data?")
        data = np.load("drbayes/experiments/synthetic_regression/ckpts/data.npy")
        x, y = data[:, 0], data[:, 1]
        f = features(x)
        rng = default_rng(exp_col.seed)
        subset_train = rng.choice(np.arange(0, len(data)), 35, replace=False)
        subset_test = np.setdiff1d(np.arange(0, len(data)), subset_train, assume_unique=True)
        data_train = data[subset_train]
        data_test = data[subset_test]

        # train df
        cat = pd.Series(rng.choice(exp_col.hue_order, len(data_train), replace=True, p=[0.7, 0.2, 0.1]))
        intercept_cat = pd.get_dummies(cat, drop_first=True).astype(np.float32)
        meta_df = intercept_cat.copy()
        intercept_cat.loc[:, 'cat2'] *= -0.5
        intercept_cat.loc[:, 'cat3'] *= 1.
        y = data_train[:, 1]
        intercept_cat = intercept_cat.to_numpy().sum(1).reshape(y.shape)
        y_cat = intercept_cat + y
        df = pd.DataFrame.from_dict(dict(x=data_train[:, 0], y=y_cat.squeeze(), category=cat.to_numpy()))
        df = pd.concat([df, meta_df], axis=1, keys=['data', 'meta'])

        # test df
        cat = pd.Series(rng.choice(exp_col.hue_order, len(data_test), replace=True, p=[0.7, 0.2, 0.1]))
        intercept_cat = pd.get_dummies(cat, drop_first=True).astype(np.float32)
        meta_df = intercept_cat.copy()
        intercept_cat.loc[:, 'cat2'] *= -0.5
        intercept_cat.loc[:, 'cat3'] *= 1.
        y = data_test[:, 1]
        intercept_cat = intercept_cat.to_numpy().sum(1).reshape(y.shape)
        y_cat = intercept_cat + y
        df_test = pd.DataFrame.from_dict(dict(x=data_test[:, 0], y=y_cat.squeeze(), category=cat.to_numpy()))
        df_test = pd.concat([df_test, meta_df], axis=1, keys=['data', 'meta'])
        df.to_csv('data_cat_train.csv')
        df_test.to_csv('data_cat_test.csv')

    exp_col.df = df
    exp_col.df_test = df_test
    return df, df_test


class NoValProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = tqdm(
            disable=True,
        )
        return bar

    def _update_n(bar, value: int) -> None:
        if not bar.disable:
            bar.n = value
            bar.update(0)  # let tqdm decide when to trigger a display refresh


def initialize_subspace_model_v2(exp_col: expCollector,
                                 plot_predictive_f,
                                 plot_curve_solution_f):
    """
    :param plot_predictive_f: plot function with format f(model, wandb_logger)
    :param plot_curve_solution_f: plot function with format f(exp_col, best_curve_model, wandb_logger)
    """
    use_valid = exp_col.valid_dataloader is not None
    bar = NoValProgressBar()

    # %% train other control points of Bézier curve
    # wandb stuff ->
    # load start and end-point from wandb (previous trained models)
    model_structure = getModel(RegNet, seed=exp_col.seed, **exp_col.base_net_kwargs, **exp_col.net_kwargs).to(
        device=device)
    wandb_logger = WandbLogger(project=exp_col.wandb_project, log_model=False, name="model_curve", group="RegNetCurve")
    # log code
    wandb_logger.experiment.log_code("./", name=f"project_code_{wandb_logger.experiment.id}",
                                     exclude_fn=exclude_project_code_dirs)
    ckp_dir = os.path.join(wandb_logger.experiment.dir, "checkpoints")
    if use_valid:
        wandb_logger.experiment.define_metric("valid/loss", summary="min")
        if exp_col.metric_collection:
            [wandb_logger.experiment.define_metric(f"valid/{m}", summary="min") for m in exp_col.metric_collection]
        # wandb_logger.experiment.define_metric("valid/MeanSquaredError", summary="min")
        # wandb_logger.experiment.define_metric("valid/MeanAbsoluteError", summary="min")
        callbacks = [ModelCheckpoint(dirpath=ckp_dir, save_top_k=1, monitor="valid/loss"), bar]
    else:
        callbacks = [ModelCheckpoint(dirpath=ckp_dir, save_top_k=1, monitor="train/loss_epoch"), bar]
    # <- wandb stuff end

    # train curve model
    trainer = pl.Trainer(devices=1,
                         max_epochs=exp_col.max_epochs * 3,
                         # max_epochs=30,
                         logger=wandb_logger,
                         callbacks=callbacks
                         )
    kwargs = exp_col.net_kwargs.copy()
    kwargs["ortho_layer_name_nn_head"] = "model.net.net_[4]" if exp_col.use_ortho else None
    kwargs["metric_collection"] = MetricCollection([]) if exp_col.metric_collection is None else exp_col.metric_collection
    # kwargs["lr"] /= 2.
    model_curve = getModel(CurveModel, model_start=None, model_end=None, fix_start=False, fix_end=False,
                           architecture=CurveArchitectureFromSequential,
                           architecture_kwargs=dict(base_sequential_model=model_structure.dnn),
                           seed=exp_col.seed,
                           num_bends=exp_col.num_bends,
                           output_dim=exp_col.base_net_kwargs['output_dim'],
                           **kwargs)

    trainer.fit(model_curve, train_dataloaders=exp_col.train_dataloader, val_dataloaders=exp_col.valid_dataloader)
    print("Sigma: ", model_curve.loss_fn.sigma)

    # wandb stuff ->
    # save best model
    art = wandb.Artifact(f"model_state_{wandb_logger.experiment.id}", type=model_curve.__class__.__name__,
                         description="Regression Base Model")
    art.add_dir(ckp_dir)
    wandb_logger.experiment.log_artifact(art)
    # load best curve model according train loss
    best_curve_model = type(model_curve).load_from_checkpoint(checkpoint_path=callbacks[0].best_model_path,
                                                              model_start=None, model_end=None,
                                                              architecture_kwargs=dict(
                                                                  base_sequential_model=model_structure.dnn),
                                                              **kwargs
                                                              ).to(device=device)
    print("Sigma: ", model_curve.loss_fn.sigma)
    plot_curve_solution_f(exp_col=exp_col,
                          best_curve_model=best_curve_model,
                          wandb_logger=wandb_logger)
    wandb_exp_id = wandb_logger.experiment.id
    # <- wandb stuff end

    return best_curve_model, wandb_exp_id, None, wandb_logger


def initialize_subspace_model(exp_col: expCollector,
                              plot_predictive_f,
                              plot_curve_solution_f):
    """
    :param plot_predictive_f: plot function with format f(model, wandb_logger)
    """
    use_valid = exp_col.valid_dataloader is not None
    bar = NoValProgressBar()

    # %% train two independent models
    # Instantiate two different randomized initialized models
    models = []
    for se in [exp_col.seed + 11, exp_col.seed + 33]:
        model = getModel(RegNet, seed=se, **exp_col.base_net_kwargs, **exp_col.net_kwargs).to(device=device)
        # model = torch.jit.trace(model, example_inputs=(torch.randn(10,2, device=device), torch.randn(10,3, device=device)))
        models.append(model)

    # train models
    wandb_id_control_point = []
    for model in models:
        # wandb stuff ->
        wandb_logger = WandbLogger(project=exp_col.wandb_project, log_model=False, name="model_control_point",
                                   group="RegNet")
        wandb_logger.experiment.define_metric("train/loss_epoch", summary="min")
        ckp_dir = os.path.join(wandb_logger.experiment.dir, "checkpoints")
        if use_valid:
            wandb_logger.experiment.define_metric("valid/loss", summary="min")
            # wandb_logger.experiment.define_metric("valid/MeanSquaredError", summary="min")
            # wandb_logger.experiment.define_metric("valid/MeanAbsoluteError", summary="min")
            callbacks = [ModelCheckpoint(dirpath=ckp_dir, save_top_k=1, monitor="valid/loss"), bar]
        else:
            callbacks = [ModelCheckpoint(dirpath=ckp_dir, save_top_k=1, monitor="train/loss_epoch"), bar]
        # log code
        wandb_logger.experiment.log_code("./", name=f"project_code_{wandb_logger.experiment.id}",
                                         exclude_fn=exclude_project_code_dirs)
        # <- wandb stuff end

        # train loop
        trainer = pl.Trainer(accelerator="auto",
                             devices=1,
                             max_epochs=exp_col.max_epochs,
                             # max_epochs=30,
                             logger=wandb_logger,
                             callbacks=callbacks
                             )
        trainer.fit(model, train_dataloaders=exp_col.train_dataloader, val_dataloaders=exp_col.valid_dataloader)

        # wandb stuff ->
        # save best model
        art = wandb.Artifact(f"model_state_{wandb_logger.experiment.id}", type=model.__class__.__name__,
                             description="Regression Base Model")
        art.add_dir(ckp_dir)
        wandb_logger.experiment.log_artifact(art)
        # load best model
        model = model.load_from_checkpoint(checkpoint_path=callbacks[0].best_model_path).eval()
        plot_predictive_f(exp_col,
                          model=model,
                          wandb_logger=wandb_logger)
        # store id (to be able to reload from wandb such that wandb is aware of model hierarchy)
        wandb_id_control_point.append(wandb_logger.experiment.id)
        wandb.finish()
        # <- wandb stuff end

    # %% train other control points of Bézier curve
    # wandb stuff ->
    # load start and end-point from wandb (previous trained models)
    model_type = type(models[0])
    wandb_logger = WandbLogger(project=exp_col.wandb_project, log_model=False, name="model_curve", group="RegNetCurve")
    model1, config1 = load_model(wandb_logger.experiment,
                                 f'ddold/{wandb_logger.experiment.project}/{wandb_id_control_point[0]}', strict=True,
                                 model_cls=model_type, file_name=None, metric_collection=MetricCollection([]))
    model2, config2 = load_model(wandb_logger.experiment,
                                 f'ddold/{wandb_logger.experiment.project}/{wandb_id_control_point[1]}', strict=True,
                                 model_cls=model_type, file_name=None, metric_collection=MetricCollection([]))
    # log code
    wandb_logger.experiment.log_code("./", name=f"project_code_{wandb_logger.experiment.id}",
                                     exclude_fn=exclude_project_code_dirs)
    ckp_dir = os.path.join(wandb_logger.experiment.dir, "checkpoints")
    if use_valid:
        wandb_logger.experiment.define_metric("valid/loss", summary="min")
        # wandb_logger.experiment.define_metric("valid/MeanSquaredError", summary="min")
        # wandb_logger.experiment.define_metric("valid/MeanAbsoluteError", summary="min")
        callbacks = [ModelCheckpoint(dirpath=ckp_dir, save_top_k=1, monitor="valid/loss"), bar]
    else:
        callbacks = [ModelCheckpoint(dirpath=ckp_dir, save_top_k=1, monitor="train/loss_epoch"), bar]
    # <- wandb stuff end

    # train curve model
    trainer = pl.Trainer(devices=1,
                         max_epochs=exp_col.max_epochs * 3,
                         # max_epochs=30,
                         logger=wandb_logger,
                         callbacks=callbacks
                         )
    kwargs = exp_col.net_kwargs.copy()
    kwargs["ortho_layer_name_nn_head"] = "model.net.net_[4]" if exp_col.use_ortho else None
    kwargs["metric_collection"] = MetricCollection([])
    kwargs["lr"] /= 2.
    model_curve = getModel(CurveModel, model_start=model1, model_end=model2,
                           architecture=CurveArchitectureFromSequential,
                           architecture_kwargs=dict(base_sequential_model=model1.dnn),
                           seed=exp_col.seed,
                           num_bends=exp_col.num_bends,
                           output_dim=exp_col.base_net_kwargs['output_dim'],
                           **kwargs)

    trainer.fit(model_curve, train_dataloaders=exp_col.train_dataloader, val_dataloaders=exp_col.valid_dataloader)

    # wandb stuff ->
    # save best model
    art = wandb.Artifact(f"model_state_{wandb_logger.experiment.id}", type=model_curve.__class__.__name__,
                         description="Regression Base Model")
    art.add_dir(ckp_dir)
    wandb_logger.experiment.log_artifact(art)
    # load best curve model according train loss
    best_curve_model = type(model_curve).load_from_checkpoint(checkpoint_path=callbacks[0].best_model_path,
                                                              model_start=model1, model_end=model2,
                                                              architecture_kwargs=dict(
                                                                  base_sequential_model=model1.dnn),
                                                              ).to(device=device)
    plot_curve_solution_f(exp_col=exp_col,
                          best_curve_model=best_curve_model,
                          wandb_logger=wandb_logger)
    wandb_exp_id = wandb_logger.experiment.id
    # <- wandb stuff end

    return best_curve_model, wandb_exp_id, wandb_id_control_point, wandb_logger


def load_subspace_model(exp_col: expCollector, curve_model, curve_wandb_id, name_post=""):
    wandb_logger = WandbLogger(project=exp_col.wandb_project, log_model=False,
                               name=f'HMC_subspace_from_{curve_wandb_id}' + name_post,
                               group="SemiSub", resume='never')
    # log code
    wandb_logger.experiment.log_code("./", name=f"project_code_{wandb_logger.experiment.id}",
                                     exclude_fn=exclude_project_code_dirs)
    # instantiate basemodel (curve model requires model structure)
    model = getModel(RegNet, seed=exp_col.seed, **exp_col.base_net_kwargs, **exp_col.net_kwargs)
    # reload such that wandb is aware of the dependency
    best_curve_model, config = load_model(wandb_logger.experiment,
                                          f'ddold/{wandb_logger.experiment.project}/{curve_wandb_id}',
                                          strict=True, file_name=None,
                                          model_cls=type(curve_model),
                                          model_start=None, model_end=None,
                                          metric_collection=MetricCollection([]),
                                        loss_fn=exp_col.nll_fn,
                                          architecture=CurveArchitectureFromSequential,
                                          architecture_kwargs=dict(
                                              base_sequential_model=model.dnn))  # to check architecture_kwargs=dict(base_sequential_model=models[0].lin))
    mean, cov, S = span_space_from_curve_model(best_curve_model.model, exp_col.num_bends)
    print("Instantiate semi subspace model for ess")
    subspace_model = getModel(RegNet, mean=mean, cov_factor=cov, **exp_col.base_net_kwargs, **exp_col.net_kwargs).to(
        device=device)
    S = S.detach().cpu().numpy()[None,:]
    print(S)
    wandb_logger.experiment.log({"singular_values": wandb.Table(data=S, columns=list(range(S.shape[1])))})
    return subspace_model, wandb_logger

def run_hmc_on_subspace(num_chains, num_warmup, num_samples, prior_scale, exp_col:expCollector, curve_model, curve_wandb_id):
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
        outcome_dist=exp_col.nll_fn.dist_)

    pyro.set_rng_seed(exp_col.seed+3)
    u_train, x_train, y_train = exp_col.train_dataloader.dataset[:]
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
    mcmc.run(u_train, x_train, y_train)
    wandb_logger.experiment.config.update({'num_chains': num_chains,
                                            'num_warmup': num_warmup,
                                            'num_samples': num_samples,
                                            'prior_scale': prior_scale,
                                            'seed': exp_col.seed,
                                            'Subspace_dimension': exp_col.num_bends - 1})
    # create arviz inference object
    az_post_hmc = az.from_pyro(mcmc, log_likelihood=False)
    az_post_hmc = az_post_hmc.rename({
        'structure_nn.weight': 'theta',
        'structure_nn.weight_dim_0': 'theta_dim_0',
        'structure_nn.weight_dim_1': 'theta_dim_1'})
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

    # compute valid lppd
    if exp_col.valid_dataloader is not None:
        u_valid, x_valid, y_valid = exp_col.valid_dataloader.dataset[:]
        lppd_valid = log_pointwise_predictive_likelihood(pyro_model, mcmc.get_samples(), u=u_valid, x=x_valid, y=y_valid)
        lppd_valid = (torch.logsumexp(lppd_valid, dim=0) - np.log(lppd_valid.shape[0])).sum()
        wandb_logger.experiment.summary["valid_lppd"] = lppd_valid

    # compute test lppd
    if exp_col.test_dataloader is not None:
        u_test, x_test, y_test = exp_col.test_dataloader.dataset[:]
        lppd_test = log_pointwise_predictive_likelihood(pyro_model, mcmc.get_samples(), u=u_test, x=x_test, y=y_test)
        lppd_test = (torch.logsumexp(lppd_test, dim=0) - np.log(lppd_test.shape[0])).sum()
        wandb_logger.experiment.summary["test_lppd"] = lppd_test

    # compute grid
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
    wandb.finish()
    return az_post_hmc, mcmc


if __name__ == '__main__':
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": False,
        "text.latex.preamble": r'\usepackage{amsfonts}',
        # "font.family": "Helvetica",
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

    # Hyper-Parameters
    # kwargs = {"dimensions": [200, 50, 50, 50],
    use_ortho = False
    net_kwargs = dict(lr=5e-2,
                      weight_decay=1e-5,
                      loss_fn=nn.MSELoss(reduction='mean'),
                      num_structure=2,
                      ortho_layer_name_nn_head="lin[4]" if use_ortho else None
                      )
    base_net_kwargs = {"dimensions": [16, 16],
                       "output_dim": 1,
                       "input_dim": 2}
    exp_col = expCollector(wandb_project='Semi_Testte',
                           use_ortho=use_ortho,
                           seed=0,
                           base_net_kwargs=base_net_kwargs,
                           net_kwargs=net_kwargs,
                           max_epochs=1000)

    # generate dataset
    df, df_test = prepare_data(exp_col)
    plt.figure(figsize=(9., 7.), dpi=150)
    plot_data(exp_col)
    dataset, (f_mean, f_std) = get_ds_from_df(df, device)
    exp_col.f_mean = f_mean  # store data mean and data std
    exp_col.f_std = f_std
    dataset_test = get_ds_test_from_df(df_test, device, f_mean, f_std)
    cuda_loader = FastFillTensorDataLoader(dataset, batch_size=len(df), shuffle=True,
                                           pin_memory=False)  # rename to train_loader
    exp_col.train_dataloader = cuda_loader

    # initialize subspace model
    #      - Train two independent models
    #      - Connect them via a Bézier-Curve
    #      - Span subspace from the different weight vectors
    curve_model, curve_wandb_id, _, _ = initialize_subspace_model(
        exp_col=exp_col,
        plot_predictive_f=plot_predictive_regression,
        plot_curve_solution_f=plot_curve_solutions_regression)
    print("*" * 27)
    print("* Initialisation finished *")
    print("*" * 27)
    subspace_model, wandb_logger = load_subspace_model(exp_col,
                                                       curve_model,
                                                       curve_wandb_id)

    # %% Run the ESS on the subspace
    cuda_loader_no_shuffle = FastFillTensorDataLoader(dataset, batch_size=len(dataset), shuffle=False, pin_memory=False)
    ess = SemiSubEllipticalSliceSampling(subspace_model,
                                         prior_scale=1.,
                                         prior_scale_subspace=5.,
                                         temperature=1.,
                                         temperature_only_on_nn=False,
                                         num_integration_samples=2000,
                                         num_chains=2,
                                         num_warmup=20,
                                         integration_range=20.,
                                         device=device,
                                         criterion=gaussian_nll,
                                         num_samples=10,
                                         seed=exp_col.seed)
    wandb_logger.experiment.config.update({'ess_param': ess.hparams_initial})
    logprobs = ess.fit(dataset=cuda_loader_no_shuffle)

    # compute predictive performance
    xt, mt, yt = dataset_test[:]
    preds = ess.predict(ess.all_samples.T, xt, mt).squeeze()
    lppd = (torch.logsumexp(-gaussian_nll(preds, yt), dim=0) - np.log(preds.shape[0])).sum()
    wandb_logger.experiment.summary["valid_lppd"] = lppd
    # Plot results
    plot_subspace_solution_regression(exp_col,
                                      ess=ess,
                                      logprobs=logprobs,
                                      curve_wandb_id=curve_wandb_id,
                                      wandb_logger=wandb_logger)
    wandb.finish()
