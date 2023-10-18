import typing

import torch
from tqdm import tqdm, tnrange
from drbayes.subspace_inference.posteriors import elliptical_slice, slice_sample
import numpy as np
from subspace_inference.posteriors.ess import EllipticalSliceSampling
from subspace_inference.posteriors.proj_model import ProjectedModel
from src.semiSub_model import Subspace, SemiAdditive
from pytorch_lightning.core.mixins import HyperparametersMixin
import argparse


class SemiSubEllipticalSliceSampling(HyperparametersMixin, torch.nn.Module):
    def __init__(self, model: typing.Union[Subspace, SemiAdditive], prior_scale: float, prior_scale_subspace: float,
                 criterion=torch.nn.BCEWithLogitsLoss(reduction='none'), num_samples=20, num_integration_samples=50000,
                 integration_range=50, temperature=1, num_chains=1, num_warmup=0,
                 device='cpu', method='elliptical', temperature_only_on_nn=False, seed=None, *args, **kwargs):
        super().__init__()
        if method == 'elliptical':
            self.slice_method = elliptical_slice
        if method == 'slice':
            self.slice_method = slice_sample
        if (seed != None) and (type(seed) == int):
            torch.random.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)

        assert isinstance(model, Subspace), f"Expected Model of type Subspace. Got '{type(model)}'"
        self.base_model = model
        if model.device != device:
            self.base_model = self.base_model.to(device=device)

        self.criterion = criterion
        self.num_samples = num_samples
        self.num_integration_samples_ = num_integration_samples
        self.integration_range_ = integration_range
        self.temperature_ = temperature
        self.num_chains_ = num_chains
        self.num_warmup_ = num_warmup

        self.all_samples = None
        self.prior_scale_ = prior_scale
        self.prior_scale_subspace_ = prior_scale_subspace
        self.device_ = device
        self.split_temperature_ = temperature_only_on_nn
        self.save_hyperparameters("prior_scale", "seed", "prior_scale_subspace", "criterion", "num_samples",
                                  "num_integration_samples",
                                  "integration_range", "temperature", "method", "temperature_only_on_nn", "num_chains",
                                  "num_warmup")

    @classmethod
    def add_model_specific_args(cls, parent_parser):
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument('--prior_scale', type=float, default=argparse.SUPPRESS, required=True)
        parser.add_argument('--num_samples', type=int, default=argparse.SUPPRESS)
        parser.add_argument('--num_integration_samples', type=int, default=argparse.SUPPRESS)
        parser.add_argument('--integration_range', type=float, default=argparse.SUPPRESS)
        parser.add_argument('--temperature', type=float, default=argparse.SUPPRESS)
        parser.add_argument('--temperature_only_on_nn', default=argparse.SUPPRESS, action='store_true')
        return parent_parser

    def forward(self, *args, **kwargs):
        return self.base_model.forward(*args, **kwargs)

    def prior_sample(self):
        cov_mat = np.eye(self.base_model.rank)
        # elif prior=='schur':
        #     trans_cov_mat = self.subspace.cov_factor.matmul(self.subspace.cov_factor.subspace.t()).numpy()
        #     trans_cov_mat /= (self.swag_model.n_models.item() - 1)
        #     cov_mat = np.eye(self.subspace.cov_factor.size(0)) + trans_cov_mat
        split = self.base_model.rank - self.base_model.num_structure_
        cov_mat[:split, :split] *= self.prior_scale_subspace_
        cov_mat[split:, split:] *= self.prior_scale_
        sample = np.random.multivariate_normal(np.zeros(self.base_model.rank), cov_mat.astype(np.float64), 1)[0, :]
        return sample

    def predict(self, samples, input) -> torch.Tensor:  # shape (#sample x #ds x out)
        with torch.no_grad():
            samples = torch.tensor(samples, dtype=torch.float32, device=self.device_)
            preds = None
            for sample in tqdm(samples):
                self.base_model.set_parameter_vector(sample)
                pred = self(input)[None, ...]
                preds = pred if preds is None else torch.concat([preds, pred], dim=0)
        return preds

    def log_pdf(self, params, dataset, minibatch=False):
        params_tensor = torch.tensor(params, dtype=torch.float32, device=self.device_)
        params_tensor = params_tensor.view(-1)
        with torch.no_grad():
            self.base_model.set_parameter_vector(params_tensor)
            nll = 0
            if self.split_temperature_:
                batch_log_like_theta = torch.zeros(len(dataset),
                                                   self.num_integration_samples_ ** self.base_model.num_structure_,
                                                   device=params_tensor.device,
                                                   dtype=params_tensor.dtype)
                structure_param_space_linspace = torch.linspace(-self.integration_range_, self.integration_range_,
                                                                self.num_integration_samples_,
                                                                device=params_tensor.device,
                                                                dtype=params_tensor.dtype)
                # get multidimensional grid (equidistant linspace in each dimension)
                structure_param_space = torch.cartesian_prod(
                    *[structure_param_space_linspace] * self.base_model.num_structure_).T  # shape (#S) or (#D x #S)
                if len(structure_param_space.shape) == 1:
                    structure_param_space = structure_param_space.unsqueeze(0)  # shape (1 x #S)
            for batch_num, (bdata, target) in enumerate(dataset):
                if minibatch and batch_num > 0:
                    break
                # copy data to device
                if target.device != self.device_:
                    target = target.to(device=self.device_)
                if isinstance(self.base_model, SemiAdditive):
                    img, meta = bdata
                    if meta.device != self.device_:
                        meta = meta.to(device=self.device_)
                else:
                    img = bdata
                if img.device != self.device_:
                    img = img.to(device=self.device_)

                # compute -p(D| \theta_1 ,\theta_2)
                if isinstance(self.base_model, SemiAdditive):
                    # for semi structure model
                    out_structure = self.base_model.forward_structure(meta)  # computes QR from meta if necessary
                    out_nn = self.base_model.forward_nn(img)  # Shape (Bx1)
                    out = out_nn + out_structure
                    nll += self.criterion(out, target).sum()

                    if self.split_temperature_:
                        # for computation of p(D| \theta_1)
                        out_structure = torch.mm(meta,
                                                 structure_param_space)  # torch.mm(meta (Bx(1...D)), structure_param_space ((1...D)xS))
                        out = out_nn + out_structure  # self.base_model_(img) (Bx1)  ;   torch.mm() (BxS)
                        batch_item_log_like_theta = -self.criterion(out.unsqueeze(-1), torch.tile(target.unsqueeze(-1), (
                            1, out.size(1))))  # (BxS) # criterion is negative log likelihood
                        batch_log_like_theta[batch_num] = batch_item_log_like_theta.sum(0)
                else:
                    # normal way
                    out = self.base_model(img)  # Shape (Bx1)
                    nll += self.criterion(out, target).sum()

            # compute of p(D| \theta_1) = \int p(D| \theta_1, \theta_2) p(\theta_2)
            # only for semi structure model and if we want to split temperature effect
            if self.split_temperature_ and isinstance(self.base_model, SemiAdditive):
                prior_dist = torch.distributions.Normal(
                    torch.zeros(self.base_model.num_structure_, 1, device=img.device, dtype=img.dtype),
                    torch.full((self.base_model.num_structure_, 1), self.prior_scale_, device=img.device))  # (#D_s x 1)
                log_post_unnorm_theta_space = batch_log_like_theta.sum(0) + prior_dist.log_prob(
                    structure_param_space).sum(0)  # (#B x #S).sum(0) + (#D_s x #S).sum(0) = (#S)
                # do numerical integration
                dx = structure_param_space_linspace[1:] - structure_param_space_linspace[:-1]
                dx = torch.cat([dx, dx[-1:]])
                dx[0] /= 2  # because of trapz rule f_0/2 + f_1-n-1 + f_n/2
                dx[-1] /= 2  # because of trapz rule f_0/2 + f_1-n-1 + f_n/2
                # compute delta volume from linspace (Tensorprodukt-Trapezsumme)
                dxs = dx
                for _ in range(self.base_model.num_structure_ - 1):
                    dxs = torch.kron(dxs, dx)
                # log_post_unnorm_theta = [torch.logsumexp(log_post_unnorm_theta_space + torch.log(dx))]
                log_post_unnorm_theta = torch.logsumexp(log_post_unnorm_theta_space + torch.log(dxs), dim=0)
                # compute partial temperatured likelihood p(D| \theta_1, \theta_2) / p(D| \theta_1) * p(D| \theta_1)^1/T
                logpdf = -nll + ((1 - self.temperature_) / self.temperature_) * log_post_unnorm_theta
            else:
                # print("normal")
                logpdf = -nll / self.temperature_
            logpdf = logpdf.cpu().numpy()
            # print("logpdf, ", logpdf)
            return logpdf

    def fit(self, **kwargs):
        all_samples = np.zeros((self.base_model.rank, self.num_samples * self.num_chains_))
        logprobs = np.zeros(self.num_samples * self.num_chains_)
        # run multiple chains in sequential mode
        for j in tnrange(self.num_chains_, position=1, desc="Chains"):
            # print("Run chain ", j)
            # initialize at prior mean = 0
            # current_sample = np.zeros(self.base_model.rank)
            current_sample = np.random.randn(self.base_model.rank)
            logprob = None
            # run warmup
            if self.num_warmup_:
                for i in tnrange(self.num_warmup_, position=2,leave=False, postfix="Warmup"):
                    prior_sample = self.prior_sample()
                    current_sample, logprob = self.slice_method(initial_theta=current_sample, prior=prior_sample,
                                                                lnpdf=self.log_pdf, **kwargs)
            # run sampling
            for i in tnrange(self.num_samples, position=2,leave=False, postfix="Sampling"):
                prior_sample = self.prior_sample()
                current_sample, logprob = self.slice_method(initial_theta=current_sample, prior=prior_sample,
                                                            lnpdf=self.log_pdf, **kwargs)
                logprobs[i + j * self.num_samples] = logprob
                all_samples[:, i + j * self.num_samples] = current_sample

        self.all_samples = all_samples
        return logprobs


class SemiSubEllipticalSliceSamplingMV(SemiSubEllipticalSliceSampling):
    def log_pdf(self, params, dataset, minibatch=False):
        params_tensor = torch.tensor(params, dtype=torch.float32, device=self.device_)
        params_tensor = params_tensor.view(-1)
        with torch.no_grad():
            self.base_model.set_parameter_vector(params_tensor)
            nll = 0
            if self.split_temperature_:
                # compute log likelihood for each category, batch and integration_sample. Compute ll for each
                # category of dummy encoding
                batch_log_like_theta = torch.zeros(1 + self.base_model.num_structure_, len(dataset),
                                                   self.num_integration_samples_,
                                                   device=params_tensor.device,
                                                   dtype=params_tensor.dtype)
                structure_param_space_linspace = torch.linspace(-self.integration_range_, self.integration_range_,
                                                                self.num_integration_samples_,
                                                                device=params_tensor.device,
                                                                dtype=torch.float64)
                # get multidimensional grid (equidistant linspace in each dimension)
                structure_param_space = torch.tile(structure_param_space_linspace.unsqueeze(0),
                                                   (self.base_model.num_structure_, 1))
                # structure_param_space = torch.cartesian_prod(
                #     *[structure_param_space_linspace] * self.base_model.num_structure_).T  # shape (#S) or (#D x #S)
                # if len(structure_param_space.shape) == 1:
                #     structure_param_space = structure_param_space.unsqueeze(0)  # shape (1 x #S)
            for batch_num, (bdata, target) in enumerate(dataset):
                if minibatch and batch_num > 0:
                    break
                # copy data to device
                if target.device != self.device_:
                    target = target.to(device=self.device_)
                if isinstance(self.base_model, SemiAdditive):
                    img, meta = bdata
                    if meta.device != self.device_:
                        meta = meta.to(device=self.device_)
                else:
                    img = bdata
                if img.device != self.device_:
                    img = img.to(device=self.device_)

                # compute -p(D| \theta_nn ,\theta_structure)
                if isinstance(self.base_model, SemiAdditive):
                    # for semi structure model
                    out_structure = self.base_model.forward_structure(meta)  # computes QR from meta if necessary
                    # first compute structure output such that we can compute Q to orthogonalize NN output
                    out_nn = self.base_model.forward_nn(img)  # Shape (Bx1)
                    out = out_nn + out_structure
                    nll += self.criterion(out.squeeze(), target).sum()

                    if self.split_temperature_:
                        # for computation of p(D| \theta_nn)
                        out_structure = torch.mm(meta.to(dtype=torch.float64),
                                                 structure_param_space)  # torch.mm(meta (Bx(1...D)), structure_param_space ((1...D)xS))
                        out = out_nn + out_structure  # self.base_model_(img) (Bx1)  ;   torch.mm() (BxS)
                        batch_item_log_like_theta = -self.criterion(out, torch.tile(target.unsqueeze(-1), (
                            1, out.size(1))))  # (BxS) # criterion is negative log likelihood
                        # batch_log_like_theta[batch_num] = batch_item_log_like_theta.sum(0)

                        num_struc = self.base_model.num_structure_
                        index_null = torch.eq(meta, torch.zeros(1, num_struc, dtype=out.dtype, device=out.device))
                        index_all_null = index_null[:, 0]
                        for i in np.arange(1, num_struc):
                            index_all_null = torch.bitwise_and(index_all_null, index_null[:, i])
                            batch_log_like_theta[i, batch_num] = batch_item_log_like_theta[~index_null[:, i - 1]].sum(0)
                        batch_log_like_theta[0, batch_num] = batch_item_log_like_theta[index_all_null].sum(0)
                        batch_log_like_theta[-1, batch_num] = batch_item_log_like_theta[~index_null[:, -1]].sum(0)
                else:
                    # normal way
                    out = self.base_model(img)  # Shape (Bx1)
                    nll += self.criterion(out.squeeze(), target).sum()

            # compute of p(D| \theta_1) = \int p(D| \theta_1, \theta_2) p(\theta_2)
            # only for semi structure model and if we want to split temperature effect
            if self.split_temperature_ and isinstance(self.base_model, SemiAdditive):
                prior_dist = torch.distributions.Normal(
                    torch.zeros(self.base_model.num_structure_, 1, device=img.device, dtype=img.dtype),
                    torch.full((self.base_model.num_structure_, 1), self.prior_scale_, device=img.device))  # (#D_s x 1)
                batch_log_like_theta = batch_log_like_theta.sum(1)  # sum over every batch
                log_post_unnorm_theta_space = batch_log_like_theta[1:] + prior_dist.log_prob(
                    structure_param_space)  # (#D_s+1 x #B x #S).sum(1) + (#D_s x #S).sum(0) = (#D_s+1 x #S)
                # do numerical integration
                dx = structure_param_space_linspace[1:] - structure_param_space_linspace[:-1]
                dx = torch.cat([dx, dx[-1:]])
                dx[0] /= 2  # because of trapz rule f_0/2 + f_1-n-1 + f_n/2
                dx[-1] /= 2  # because of trapz rule f_0/2 + f_1-n-1 + f_n/2

                # first integration over data where meta was zero in every structure dimension
                int_log_post_unnorm_theta = batch_log_like_theta[0].mean()
                assert torch.isclose(batch_log_like_theta[
                                         0, 0],
                                     int_log_post_unnorm_theta), f"Expected to be the same got:\n{batch_log_like_theta[0, 0]} \n{int_log_post_unnorm_theta}"
                # integrate over other every structure dimensions
                for i in range(self.base_model.num_structure_):
                    int_log_post_unnorm_theta += torch.logsumexp(log_post_unnorm_theta_space[i] + torch.log(dx),
                                                                 dim=0)

                # compute delta volume from linspace
                # dxs = dx
                # for _ in range(self.base_model.num_structure_ -1):
                #     dxs = torch.kron(dxs, dx)
                # log_post_unnorm_theta = [torch.logsumexp(log_post_unnorm_theta_space + torch.log(dx))]
                # log_post_unnorm_theta = torch.logsumexp(log_post_unnorm_theta_space + torch.log(dx), dim=0)
                # compute partial temperatured likelihood p(D| \theta_nn, \theta_st) / p(D| \theta_nn) * p(D| \theta_nn)^1/T
                logpdf = -nll + ((1 - self.temperature_) / self.temperature_) * int_log_post_unnorm_theta
            else:
                # print("normal")
                logpdf = -nll / self.temperature_
            logpdf = logpdf.cpu().numpy()
            # print("logpdf, ", logpdf)
            return logpdf


def test_double_integral():
    from scipy.integrate import dblquad
    from functools import partial

    def func1log(x, y, z):
        return (z - (.5 * x - 1. * y)) ** 2

    def func1(x, y, z):
        return np.exp((z - (.5 * x - 1. * y)) ** 2)

    # dblquad integration
    int_dblquad = dblquad(func1, -2, 1., -2., 1., args=(1.,))

    # trapz integration
    xx = torch.linspace(-2, 1, 600)
    yy = torch.linspace(-2, 1, 600)
    space = torch.cartesian_prod(*[xx] * 2).T
    mse = partial(func1log, z=torch.ones(space.size(1)))
    ff = mse(*space).exp()
    ff = ff.reshape(len(xx), -1)
    int_trapz = torch.trapz(torch.trapz(ff, x=xx, dim=0), x=yy).item()

    # Integration in logspace
    dx = xx[1:] - xx[:-1]
    dx = torch.cat([dx, dx[-1:]])
    dx[0] /= 2  # because of trapz rule f_0/2 + f_1-n-1 + f_n/2
    dx[-1] /= 2  # because of trapz rule f_0/2 + f_1-n-1 + f_n/2
    # compute delta volume from linspace
    dxs = dx
    for _ in range(2 - 1):
        dxs = torch.kron(dxs, dx)
    mse = partial(func1log, z=torch.ones(space.size(1)))
    ff = mse(*space)
    # log_post_unnorm_theta = [torch.logsumexp(log_post_unnorm_theta_space + torch.log(dx))]
    int_logsumexp = torch.logsumexp(ff + torch.log(dxs), dim=0).exp().item()
    print("Integrate gaus-2,1")
    print("dblquad: ", int_dblquad)
    print("trapz: ", int_trapz)
    print("logsumexp: ", int_logsumexp)
