import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
from matplotlib import colors, ticker
import seaborn as sns
# from pydicom import dcmread
import pytorch_lightning as pl
from torchmetrics import MetricCollection, AUROC, CalibrationError, SumMetric
import argparse
import wandb
from src.ess import SemiSubEllipticalSliceSampling
from utils_datamodel.pl_utils import MelanomDataModuleFromSplit
from utils_datamodel.utils import FastFillTensorDataLoader
from utils_datamodel.wandb_utils import load_model
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from subspace import curves
from subspace_inference.posteriors.proj_model import SubspaceModel, ProjectedModel
from src.plot import plot_subspace, plot_subspace_solution_regression_pyro
from matplotlib import bezier
from src.model import cudaIter, getModel, NllLoss, CurveArchitectureFromSequential, CurveModel, CurveLogger, SimpleCnn
from src.plot import exclude_project_code_dirs
from argparse import ArgumentParser
from tqdm import tqdm
from torchmetrics import SumMetric
from src.semi_sub_utils import get_curve_space_torch, log_pointwise_predictive_likelihood, span_space_from_curve_model
import pyro
import arviz as az
import xarray as xr
from pyro.infer.autoguide import init_to_sample

# %load_ext tensorboard
sns.set_style("darkgrid")
if torch.cuda.is_available() and 1:
    print("Use cuda. Device: ", torch.cuda.get_device_name(torch.cuda.current_device()))
    device = torch.device('cuda', torch.cuda.current_device())
else:
    device = torch.device('cpu')
#torch.set_default_device(device)
print("Device: ", device)

os.environ['WANDB_BASE_URL'] = 'http://141.37.176.203:8080'


# %% md
### Train at minimum two models with different seed

# %%
def main(batch_size, reuse_artifact, split, max_epochs, weight_decay, lr, seed, num_bends, num_samples, num_warmup, num_chains, gpu_dev):
    wandb_project= 'semi_subspace_split_v2'
    device=f"cuda:{gpu_dev}"
    # load start and end point
    wandb_logger = WandbLogger(project=wandb_project, log_model=False, name='model_curve',
                               group="SimpleCnnCurve", resume='never')
    curve_wandb_id = wandb_logger.experiment.id
    
    # define a metric we are interested in the best of
    wandb_logger.experiment.define_metric("valid/loss", summary="min")
    wandb_logger.experiment.define_metric("valid/BinaryCalibrationError", summary="min")
    wandb_logger.experiment.define_metric("valid/BinaryAUROC", summary="max")
    wandb_logger.experiment.define_metric("valid/BinaryAccuracy", summary="max")
    wandb_logger.experiment.define_metric("valid/BinaryAveragePrecision", summary="max")
    wandb_logger.experiment.define_metric("valid/BinaryF1Score", summary="max")

    # log code
    wandb_logger.experiment.log_code("./", name=f"project_code_{wandb_logger.experiment.id}",
                                     exclude_fn=exclude_project_code_dirs)  # exclude drbayes, wandb, dnn-mode-connectivity

    transform = nn.Sequential(torchvision.transforms.ConvertImageDtype(torch.float32),
                              torchvision.transforms.Normalize((0.8061, 0.6210, 0.5914), (0.1484, 0.1748, 0.1999)))
    transform = torch.jit.trace(transform, torch.randint(0, 255, (3, 128, 128), dtype=torch.uint8))
    dm = MelanomDataModuleFromSplit(wandb_logger=wandb_logger,
                                    batch_size=batch_size,
                                    reuse_artifact=reuse_artifact,
                                    split=split,
                                    meta_features=['age_approx'],
                                    transform_test=transform,
                                    transform_train=transform)
    outcome_dist = pyro.distributions.Bernoulli  # outcome distributions
    loss_fn = NllLoss(outcome_dist,
                      constrains={'probs': torch.nn.Sigmoid()},
                      reduction='mean')
    net_kwargs = dict(loss_fn=loss_fn,
                      num_structure=1,
                      weight_decay=weight_decay,
                      lr=lr,
                      output_dim=1,
                      seed=seed)
    
    model_structure = getModel(SimpleCnn, **net_kwargs).to(
        device=device)
    curve_model = getModel(CurveModel, model_start=None, model_end=None, fix_start=False, fix_end=False,
                       architecture_kwargs=dict(base_sequential_model=model_structure.dnn),
                       num_bends=num_bends,
                       classification=True,
                       **net_kwargs)
    ckp_dir = os.path.join(wandb_logger.experiment.dir, "checkpoints")
    callbacks = [ModelCheckpoint(dirpath=ckp_dir, save_top_k=1, monitor="valid/loss"),
                 CurveLogger(t_space=np.linspace(0, 1, 20))]
    trainer = pl.Trainer(devices=[gpu_dev],
                         max_epochs=max_epochs,
                         logger=wandb_logger,
                        #  limit_train_batches=12,
                        #  limit_val_batches=1,
                        #  num_sanity_val_steps=0,
                         callbacks=callbacks)
    
    dm.prepare_data()
    dm.setup()
    cuda_loader_val = cudaIter(device=device, bs=551, data_set=dm.ds_val)

    u, x, y = [], [], []
    for (uu, xx), yy in dm.train_dataloader():
        u.append(uu)
        x.append(xx)
        y.append(yy)
    u = torch.vstack(u).to(device=device)
    x = torch.vstack(x).to(device=device)
    y = torch.hstack(y).to(device=device)
    ds = torch.utils.data.TensorDataset(u,x,y)
    train_loader = FastFillTensorDataLoader(ds, batch_size=256, shuffle=True, pin_memory=False)

    trainer.fit(curve_model, train_dataloaders=train_loader, val_dataloaders=cuda_loader_val)
    best_curve_model = type(curve_model).load_from_checkpoint(checkpoint_path=callbacks[0].best_model_path,
                                                 model_start=None, model_end=None,
                                                 architecture_kwargs=dict(base_sequential_model=model_structure.dnn),
                                                 num_bends=num_bends,
                                                 classification=True,
                                                 **net_kwargs)
    trainer.test(best_curve_model, datamodule=dm)

    # save model as artifact
    art = wandb.Artifact(f"model_state-{wandb_logger.experiment.id}", type=curve_model.__class__.__name__,
                         description="Simple CNN Model")
    art.add_dir(ckp_dir)
    wandb_logger.experiment.log_artifact(art)
    wandb.finish()


    #%% -> create subspace
    wandb_logger = WandbLogger(project=wandb_project, log_model=False,
                               name=f'ESS_subspace_from_{curve_wandb_id}',
                               group="SemiSub", resume='never')
    # log code
    wandb_logger.experiment.log_code("./", name=f"project_code_{wandb_logger.experiment.id}",
                                     exclude_fn=exclude_project_code_dirs)
    # reload such that wandb is aware of the dependency
    best_curve_model, config = load_model(wandb_logger.experiment,
                                          f'ddold/{wandb_logger.experiment.project}/{curve_wandb_id}',
                                          strict=True, file_name=None,
                                          model_cls=type(curve_model),
                                          model_start=None, model_end=None,
                                          metric_collection=MetricCollection([]),
                                          architecture=CurveArchitectureFromSequential,
                                          architecture_kwargs=dict(
                                              base_sequential_model=model_structure.dnn))  # to check architecture_kwargs=dict(base_sequential_model=models[0].lin))
    mean, cov = get_curve_space_torch(best_curve_model)
    subspace_model = getModel(SimpleCnn, mean=mean, cov_factor=cov, **net_kwargs).to(device=device)
    # svd not possible
    # mean, cov, S = span_space_from_curve_model(best_curve_model.model, num_bends)
    # S = S.detach().cpu().numpy()[None,:]
    # wandb_logger.experiment.log({"singular_values": wandb.Table(data=S, columns=list(range(S.shape[1])))})

    # %% Run the ESS on the subspace
    torch.set_default_dtype(torch.float32)
    cuda_loader_no_shuffle = cudaIter(device=device, bs=422, data_set=dm.ds_train)
    loss_fn = NllLoss(outcome_dist,
                      constrains={'probs': torch.nn.Sigmoid()},
                      reduction=None)
    ess = SemiSubEllipticalSliceSampling(subspace_model,
                                         prior_scale=1.,
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
                                         seed=seed+3)
    wandb_logger.experiment.config.update({'ess_param': ess.hparams_initial})
    logprobs = ess.fit(dataset=cuda_loader_no_shuffle)

    # compute predictive performance
    preds = []
    obs = []
    for (u_valid, x_valid), y_valid in cuda_loader_val:
        pred = ess.predict(ess.all_samples.T, u_valid, x_valid)
        preds.append(pred)
        obs.append(y_valid)
    preds = torch.concatenate(preds, dim=1) # shape: (#samples,#data,#params)
    obs = torch.concatenate(obs, dim=-1)
    lppd = (torch.logsumexp(-loss_fn(preds, obs), dim=0) - np.log(preds.shape[0])).sum()
    wandb_logger.experiment.summary["valid_lppd"] = lppd
    del cuda_loader_val

    preds = []
    obs = []
    for (u_test, x_test), y_test in dm.test_dataloader():
        pred = ess.predict(ess.all_samples.T, u_test.to(device=device), x_test.to(device=device))
        preds.append(pred)
        obs.append(y_test.to(device=device))
    preds = torch.concatenate(preds, dim=1)
    obs = torch.concatenate(obs, dim=-1)
    lppd = (torch.logsumexp(-loss_fn(preds, obs), dim=0) - np.log(preds.shape[0])).sum()
    wandb_logger.experiment.summary["test_lppd"] = lppd
    
    # create arviz inference object
    labeller = az.labels.MapLabeller(var_name_map={"theta": r"$\theta$", "varphi": r"$\varphi$"})
    samples = ess.all_samples.T.reshape(ess.num_chains_, ess.num_samples, -1)
    theta_shape = ess.base_model.structure_lin.weight.shape
    varphi_dim = num_bends - 1
    coords = {'theta_dim_0': np.arange(theta_shape[0]), 'theta_dim_1': np.arange(theta_shape[1]),
              'varphi_dim': np.arange(varphi_dim)}
    dims = {"theta": ['theta_dim_0', 'theta_dim_1'], "varphi": ['varphi_dim']}
    data = {'theta': samples[..., varphi_dim:].reshape(*samples.shape[:-1], *theta_shape),
            'varphi': samples[..., :varphi_dim]}
    az_inference_data = az.convert_to_inference_data(data, coords=coords, dims=dims)
    # y_obs = xr.Dataset({'y_obs': ({'obs': ['obs']}, dataset[:][2].detach().cpu().numpy())},
    #                    coords={'obs': np.arange(num_data)})
    # az_inference_data.add_groups(observed_data=y_obs)
    az_inference_data.add_groups(log_likelihood={'obs': logprobs.reshape(num_chains, num_samples)})

    # save samples with wandb
    az_inference_data.to_netcdf(f"az_subspace_posterior_{wandb_logger.experiment.id}.nc")
    art = wandb.Artifact(f"data_{wandb_logger.experiment.id}", type="xarray",
                            description="posterior from subspace model")
    art.add_file(f"az_subspace_posterior_{wandb_logger.experiment.id}.nc")
    wandb_logger.experiment.log_artifact(art)
    # save pyro model state dict
    torch.save(subspace_model.state_dict(), f"model_state_dict_{wandb_logger.experiment.id}.pt")
    art = wandb.Artifact(f"model_state_{wandb_logger.experiment.id}", type="pyroSemiSubspace",
                            description="subspace model state dict")
    art.add_file(f"model_state_dict_{wandb_logger.experiment.id}.pt")
    wandb_logger.experiment.log_artifact(art)

    # Description
    summary_subspace = az.summary(az_inference_data)
    wandb_logger.experiment.log({"description": wandb.Table(dataframe=summary_subspace.reset_index())})
    max_r_hat = az.rhat(az_inference_data, var_names='theta')['theta'].to_numpy().max()
    wandb_logger.experiment.summary["r_hat_max_structure"] = max_r_hat
    r_hat_smaller_1_1 = max_r_hat < 1.1
    wandb_logger.experiment.summary["r_hat<1.1"] = r_hat_smaller_1_1


    if num_bends == 3:
        # Subspace plot
        # Compute grid
        x = np.linspace(-10, 10, 40, dtype=np.float32)
        y = np.linspace(-8, 8, 40, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        grid = np.vstack([xx.flatten(), yy.flatten()]).T
        prior_dist = torch.distributions.Normal(0.,
                                                torch.tensor([5., 5., 1.],
                                                             device=device))
        log_prob_joint = []
        # cond_theta = torch.from_numpy(true_thetas.flatten()).to(device=device)
        cond_theta = torch.from_numpy(
            az_inference_data.posterior['theta'].mean(['chain', 'draw']).to_numpy().flatten()).to(
            device=device, dtype=torch.float32)
        with torch.no_grad():
            for p in tqdm(grid):
                p_t = torch.from_numpy(p).to(device=device)
                p_t = torch.concat([p_t, cond_theta])
                subspace_model.set_parameter_vector(p_t)
                ll = 0.
                for (u_train, x_train), y_train in cuda_loader_no_shuffle:
                    eta = subspace_model(img=u_train, meta=x_train)
                    ll += loss_fn(eta, y_train).sum().cpu().item()
                ll += prior_dist.log_prob(p_t).sum().cpu().item()
                # ll = dist.Normal(mu, 1.).log_prob(y_train)
                log_prob_joint.append(-ll)
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
                            vmin=np.quantile(log_prob_joint, 0.9))
        post_varphi = az_inference_data['posterior']['varphi'].to_numpy().reshape(-1, 2)
        sns.scatterplot(x=post_varphi[:, 0], y=post_varphi[:, 1], alpha=0.2, linewidth=0., s=1.)
        ax = plt.gca()
        ax.get_legend().remove()
        ax.set_xlabel(r"$\varphi_1$")
        ax.set_ylabel(r"$\varphi_2$")
        wandb_logger.experiment.log({'Subspace plot': wandb.Image(plt.gcf())})
        wandb.finish()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = CurveModel.add_model_specific_args(parser)
    parser = MelanomDataModuleFromSplit.add_data_module_specific_args(parser)
    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser.add_argument("--max_epochs", default=3, type=int)
    parser.add_argument("--num_samples", default=3, type=int)
    parser.add_argument("--num_warmup", default=3, type=int)
    parser.add_argument("--num_chains", default=3, type=int)
    parser.add_argument("--gpu_dev", default=1, type=int)
    args = parser.parse_args()
    main(**vars(args))
