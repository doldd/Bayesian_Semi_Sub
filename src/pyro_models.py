import torch
import torch.nn as nn
from functools import partial
import pyro
from pyro.nn import PyroSample, PyroModule, PyroParam
from pyro import distributions as dist

import jax.numpy as jnp
import numpyro
numpyro.set_platform('cpu')
numpyro.set_host_device_count(40)
from numpyro import distributions as np_dist
import flax.linen as flax_nn
from numpyro.contrib.module import flax_module, _update_params
from copy import deepcopy
from typing import Final

class pyroSemiSubspace(PyroModule):
    sequential_dnn: Final[nn.Sequential]
    def __init__(self,
                 mean,  # shape (#p)
                 cov_factor,  # shape (#dim_subspace, #p)
                 sequential_dnn,
                 module_structure,  # self.subspace_model_.structure_lin
                 outcome_dist=None,
                 prior_scale = 1., 
                 prior_scale_subspace=5.):
        super().__init__()

        self.mean = PyroParam(mean, event_dim=-1)
        self.cov = PyroParam(cov_factor, event_dim=-2)
        self.out_dist = outcome_dist
        self.sequential_dnn = sequential_dnn
        self.structure_nn = module_structure
        pyro.nn.module.to_pyro_module_(self.structure_nn)
        self.device = module_structure.weight.device
        dtype = module_structure.weight.dtype

        varphi_dim = cov_factor.size(0)
        self.varphi = PyroSample(prior=dist.Normal(
            torch.tensor(0., device=self.device, dtype=dtype),
            torch.tensor(prior_scale_subspace, device=self.device)).expand(
            (varphi_dim,)).to_event(1))

        self.structure_nn.weight = PyroSample(prior=dist.Normal(
            torch.tensor(0., device=self.device, dtype=dtype),
            torch.tensor(prior_scale, device=self.device, dtype=dtype)).expand(
            module_structure.weight.shape).to_event(module_structure.weight.dim()))

        # x_example = torch.randn((10, sequential_dnn[0].in_features))
        # self.functional_module = torch.jit.trace(self.functional_module_, (x_example, mean.shape[-1]))

    def functional_module(self, x:torch.Tensor, param_vec:torch.Tensor):
        module = self.sequential_dnn
        p_idx = 0
        for m in module.modules():
            if type(m) is nn.Linear:
                weight = param_vec[p_idx:p_idx + m.weight.numel()].view(m.weight.shape)
                p_idx += m.weight.numel()
                bias = None
                if m.bias is not None:
                    bias = param_vec[p_idx:p_idx + m.bias.numel()].view(m.bias.shape)
                    p_idx += m.bias.numel()
                x = nn.functional.linear(x, weight, bias)
            elif type(m) is nn.Conv2d:
                weight = param_vec[p_idx:p_idx + m.weight.numel()].view(m.weight.shape)
                p_idx += m.weight.numel()
                bias = None
                if m.bias is not None:
                    bias = param_vec[p_idx:p_idx + m.bias.numel()].view(m.bias.shape)
                    p_idx += m.bias.numel()
                x = nn.functional.conv2d(x,
                                         weight=weight, bias=bias,
                                         stride=m.stride, padding=m.padding, dilation=m.dilation, groups=m.groups)
            elif type(m) is nn.ReLU:
                x = m(x)
            elif type(m) is nn.Sequential:
                pass
            else:
                raise Exception(f"{type(m)} functional API is not implemented yet (Name: {m.__class__.__name__})")
        return x

    def forward(self, u, x, y=None):
        # forward pass of structural full bayes model
        out_structure = self.structure_nn(x)
        # forward pass of dnn with controlled parameters from subspace
        param_vec = self.mean + self.varphi @ self.cov
        if param_vec.dim() > 1:
            out_dnn = torch.vmap(self.functional_module, in_dims=0, out_dims=0)(u.repeat(param_vec.size(0), 1, 1),
                                                                                param_vec)
        else:
            out_dnn = self.functional_module(u, param_vec)

        dist_params = out_structure + out_dnn

        with pyro.plate("data"):
            outcome_dist = self.out_dist(dist_params)  # condition on u and x
            # outcome_dist = torch.distributions.Normal(loc=dist_params[:,0], scale=1.)
            likely = pyro.sample("obs", outcome_dist, obs=y)
        return likely
    
class pyroSubspaceUCI(PyroModule):
    sequential_dnn: Final[nn.Sequential]
    def __init__(self,
                 mean,  # shape (#p)
                 cov_factor,  # shape (#dim_subspace, #p)
                 sequential_dnn,
                 outcome_dist=None, 
                 prior_scale_subspace=5.):
        super().__init__()

        self.mean = PyroParam(mean, event_dim=-1)
        self.cov = PyroParam(cov_factor, event_dim=-2)
        self.out_dist = outcome_dist
        self.sequential_dnn = sequential_dnn
        param_ = next(iter(sequential_dnn.parameters()))
        self.device = param_.device
        dtype = param_.dtype

        varphi_dim = cov_factor.size(0)
        self.varphi = PyroSample(prior=dist.Normal(
            torch.tensor(0., device=self.device, dtype=dtype),
            torch.tensor(prior_scale_subspace, device=self.device)).expand(
            (varphi_dim,)).to_event(1))
        self.sigma = PyroSample(prior=dist.LogNormal(
            torch.tensor(0., device=self.device, dtype=dtype),
            torch.tensor(1., device=self.device)))

    def functional_module(self, x:torch.Tensor, param_vec:torch.Tensor):
        module = self.sequential_dnn
        p_idx = 0
        for m in module.modules():
            if type(m) is nn.Linear:
                weight = param_vec[p_idx:p_idx + m.weight.numel()].view(m.weight.shape)
                p_idx += m.weight.numel()
                bias = None
                if m.bias is not None:
                    bias = param_vec[p_idx:p_idx + m.bias.numel()].view(m.bias.shape)
                    p_idx += m.bias.numel()
                x = nn.functional.linear(x, weight, bias)
            elif type(m) is nn.Conv2d:
                weight = param_vec[p_idx:p_idx + m.weight.numel()].view(m.weight.shape)
                p_idx += m.weight.numel()
                bias = None
                if m.bias is not None:
                    bias = param_vec[p_idx:p_idx + m.bias.numel()].view(m.bias.shape)
                    p_idx += m.bias.numel()
                x = nn.functional.conv2d(x,
                                         weight=weight, bias=bias,
                                         stride=m.stride, padding=m.padding, dilation=m.dilation, groups=m.groups)
            elif type(m) is nn.ReLU or type(m) is nn.Tanh:
                x = m(x)
            elif type(m) is nn.Sequential:
                pass
            else:
                raise Exception(f"{type(m)} functional API is not implemented yet (Name: {m.__class__.__name__})")
        return x

    def forward(self, u, y=None):
        # forward pass of dnn with controlled parameters from subspace
        param_vec = self.mean + self.varphi @ self.cov
        if param_vec.dim() > 1:
            dist_params = torch.vmap(self.functional_module, in_dims=0, out_dims=0)(u.repeat(param_vec.size(0), 1, 1),
                                                                                param_vec)
        else:
            dist_params = self.functional_module(u, param_vec)
        self.out_dist.dist_kwargs = {'scale':self.sigma}

        with pyro.plate("data"):
            outcome_dist = self.out_dist(dist_params)  # condition on u and x
            # outcome_dist = torch.distributions.Normal(loc=dist_params[:,0], scale=1.)
            likely = pyro.sample("obs", outcome_dist, obs=y)
        return likely

class pyroSubspaceUCI_Zt(PyroModule):
    sequential_dnn: Final[nn.Sequential]
    def __init__(self,
                 mean,  # shape (#p)
                 cov_factor,  # shape (#dim_subspace, #p)
                 sequential_dnn,
                 prior_scale_subspace=5., 
                 z_scale=1,
                 z_mean=0):
        super().__init__()
        self.z_scale = z_scale
        self.z_mean = z_mean
        self.mean = PyroParam(mean, event_dim=-1)
        self.cov = PyroParam(cov_factor, event_dim=-2)
        self.sequential_dnn = sequential_dnn
        param_ = next(iter(sequential_dnn.parameters()))
        self.device = param_.device
        dtype = param_.dtype

        varphi_dim = cov_factor.size(0)
        self.varphi = PyroSample(prior=dist.Normal(
            torch.tensor(0., device=self.device, dtype=dtype),
            torch.tensor(prior_scale_subspace, device=self.device)).expand(
            (varphi_dim,)).to_event(1))
        self.sigma = PyroSample(prior=dist.LogNormal(
            torch.tensor(0., device=self.device, dtype=dtype),
            torch.tensor(1., device=self.device)))

    def functional_module(self, x:torch.Tensor, param_vec:torch.Tensor):
        module = self.sequential_dnn
        p_idx = 0
        for m in module.modules():
            if type(m) is nn.Linear:
                weight = param_vec[p_idx:p_idx + m.weight.numel()].view(m.weight.shape)
                p_idx += m.weight.numel()
                bias = None
                if m.bias is not None:
                    bias = param_vec[p_idx:p_idx + m.bias.numel()].view(m.bias.shape)
                    p_idx += m.bias.numel()
                x = nn.functional.linear(x, weight, bias)
            elif type(m) is nn.Conv2d:
                weight = param_vec[p_idx:p_idx + m.weight.numel()].view(m.weight.shape)
                p_idx += m.weight.numel()
                bias = None
                if m.bias is not None:
                    bias = param_vec[p_idx:p_idx + m.bias.numel()].view(m.bias.shape)
                    p_idx += m.bias.numel()
                x = nn.functional.conv2d(x,
                                         weight=weight, bias=bias,
                                         stride=m.stride, padding=m.padding, dilation=m.dilation, groups=m.groups)
            elif type(m) is nn.ReLU or type(m) is nn.Tanh:
                x = m(x)
            elif type(m) is nn.Sequential:
                pass
            else:
                raise Exception(f"{type(m)} functional API is not implemented yet (Name: {m.__class__.__name__})")
        return x

    def forward(self, u, y=None):
        # forward pass of dnn with controlled parameters from subspace
        param_vec = self.mean + self.varphi @ self.cov
        if param_vec.dim() > 1:
            dist_params = torch.vmap(self.functional_module, in_dims=0, out_dims=0)(u.repeat(param_vec.size(0), 1, 1),
                                                                                param_vec)
        else:
            dist_params = self.functional_module(u, param_vec)
        # std_u = std_n * z_scale

        scale = torch.sqrt(self.sigma**2*self.z_scale**2)
        # scale = self.sigma*self.z_scale

        with pyro.plate("data"):
            outcome_dist = torch.distributions.Normal(loc=dist_params[:,0]*self.z_scale+self.z_mean, scale=scale)
            likely = pyro.sample("obs", outcome_dist, obs=y)
        return likely

class pyroSemiSubspaceBlackbox(PyroModule):
    sequential_dnn: Final[nn.Sequential]
    structure_nn: Final[nn.Linear]
    
    def __init__(self,
                 mean,  # shape (#p)
                 cov_factor,  # shape (#dim_subspace, #p)
                 sequential_dnn,
                 module_structure,  # self.subspace_model_.structure_lin
                 num_structure, 
                 outcome_dist=None):
        super().__init__()

        self.mean = PyroParam(mean, event_dim=-1)
        self.cov = PyroParam(cov_factor, event_dim=-2)
        self.out_dist = outcome_dist
        self.sequential_dnn = sequential_dnn
        self.structure_nn = module_structure
        self.device = module_structure.weight.device
        self.num_structure = num_structure
        dtype = module_structure.weight.dtype

        varphi_dim = cov_factor.size(0)
        self.varphi = PyroSample(prior=dist.Normal(
            torch.tensor(0., device=self.device, dtype=dtype),
            torch.tensor(5., device=self.device)).expand(
            (varphi_dim,)).to_event(1))

        # x_example = torch.randn((10, sequential_dnn[0].in_features))
        # self.functional_module = torch.jit.trace(self.functional_module_, (x_example, mean.shape[-1]))

    def functional_module(self, x:torch.Tensor, param_vec:torch.Tensor):
        module = self.sequential_dnn
        p_idx = 0
        for m in module.modules():
            if type(m) is nn.Linear:
                weight = param_vec[p_idx:p_idx + m.weight.numel()].view(m.weight.shape)
                p_idx += m.weight.numel()
                bias = None
                if m.bias is not None:
                    bias = param_vec[p_idx:p_idx + m.bias.numel()].view(m.bias.shape)
                    p_idx += m.bias.numel()
                x = nn.functional.linear(x, weight, bias)
            if type(m) is nn.ReLU:
                x = m(x)
        return x

    def functional_module_linear(self, x:torch.Tensor, param_vec:torch.Tensor):
        m = self.structure_nn
        p_idx = 0
        weight = param_vec[p_idx:p_idx + m.weight.numel()].view(m.weight.shape)
        p_idx += m.weight.numel()
        bias = None
        if m.bias is not None:
            bias = param_vec[p_idx:p_idx + m.bias.numel()].view(m.bias.shape)
            p_idx += m.bias.numel()
        x = nn.functional.linear(x, weight, bias)
        return x


    def forward(self, u, x, y=None):
        # forward pass of dnn with controlled parameters from subspace
        param_vec = self.mean + self.varphi @ self.cov
        param_vec_nn = param_vec[..., :-self.num_structure]
        param_vec_struct = param_vec[..., -self.num_structure:]
        if param_vec.dim() > 1:
            out_dnn = torch.vmap(self.functional_module, in_dims=0, out_dims=0)(u.repeat(param_vec_nn.size(0), 1, 1),
                                                                                param_vec_nn)
            out_structure = torch.vmap(self.functional_module_linear, in_dims=0, out_dims=0)(x.repeat(param_vec_struct.size(0), 1, 1),
                                                                                             param_vec_struct)
        else:
            out_dnn = self.functional_module(u, param_vec_nn)
            out_structure = self.functional_module_linear(x, param_vec_struct)

        dist_params = out_structure + out_dnn

        with pyro.plate("data"):
            outcome_dist = self.out_dist(dist_params)  # condition on u and x
            # outcome_dist = torch.distributions.Normal(loc=dist_params[:,0], scale=1.)
            likely = pyro.sample("obs", outcome_dist, obs=y)
        return likely
    
def random_flax_module(
        name,
        nn_module,
        prior,
        *args,
        input_shape=None,
        apply_rng=None,
        mutable=None,
        **kwargs
):
    """
    Function copied from numpyro.contrib.module import random_flax_module (changed parameter nameing convetntion for xarry to netcdf save (doesnt allow / in name))
    A primitive to place a prior over the parameters of the Flax module `nn_module`.
    .. note::
        Parameters of a Flax module are stored in a nested dict. For example,
        the module `B` defined as follows::
            class A(flax.linen.Module):
                @flax.linen.compact
                def __call__(self, x):
                    return nn.Dense(1, use_bias=False, name='dense')(x)
            class B(flax.linen.Module):
                @flax.linen.compact
                def __call__(self, x):
                    return A(name='inner')(x)
        has parameters `{'inner': {'dense': {'kernel': param_value}}}`. In the argument
        `prior`, to specify `kernel` parameter, we join the path to it using dots:
        `prior={"inner.dense.kernel": param_prior}`.
    :param str name: name of NumPyro module
    :param flax.linen.Module: the module to be registered with NumPyro
    :param prior: a NumPyro distribution or a Python dict with parameter names as keys and
        respective distributions as values. For example::
            net = random_flax_module("net",
                                     flax.linen.Dense(features=1),
                                     prior={"bias": dist.Cauchy(), "kernel": dist.Normal()},
                                     input_shape=(4,))
        Alternatively, we can use a callable. For example the following are equivalent::
            prior=(lambda name, shape: dist.Cauchy() if name == "bias" else dist.Normal())
            prior={"bias": dist.Cauchy(), "kernel": dist.Normal()}
    :type prior: dict, ~numpyro.distributions.Distribution or callable
    :param args: optional arguments to initialize flax neural network
        as an alternative to `input_shape`
    :param tuple input_shape: shape of the input taken by the neural network.
    :param list apply_rng: A list to indicate which extra rng _kinds_ are needed for
        ``nn_module``. For example, when ``nn_module`` includes dropout layers, we
        need to set ``apply_rng=["dropout"]``. Defaults to None, which means no extra
        rng key is needed. Please see
        `Flax Linen Intro <https://flax.readthedocs.io/en/latest/notebooks/linen_intro.html#Invoking-Modules>`_
        for more information in how Flax deals with stochastic layers like dropout.
    :param list mutable: A list to indicate mutable states of ``nn_module``. For example,
        if your module has BatchNorm layer, we will need to define ``mutable=["batch_stats"]``.
        See the above `Flax Linen Intro` tutorial for more information.
    :param kwargs: optional keyword arguments to initialize flax neural network
        as an alternative to `input_shape`
    :returns: a sampled module
    """
    nn = flax_module(
        name,
        nn_module,
        *args,
        input_shape=input_shape,
        apply_rng=apply_rng,
        mutable=mutable,
        **kwargs
    )
    params = nn.args[0]
    new_params = deepcopy(params)
    with numpyro.handlers.scope(prefix=name,
                                divider='.'):  # changed divider. Slash is an invaalid varname to save a xarray into netcdf format
        _update_params(params, new_params, prior)
    nn_new = partial(nn.func, new_params, *nn.args[1:], **nn.keywords)
    return nn_new


class jaxNet(flax_nn.Module):
    dimensions: list
    output_dim: int
    input_dim: int

    @flax_nn.compact
    def __call__(self, x):
        for d in self.dimensions:
            x = flax_nn.Dense(d)(x)
            x = flax_nn.relu(x)
        x = flax_nn.Dense(self.output_dim)(x)
        return x


class NumpyroModel:
    def __init__(self, dist, base_net_kwargs, constrains={'loc': lambda x: x, 'scale': jnp.exp}, **dist_kwargs):
        self.dist = partial(dist, **dist_kwargs)
        self.module = jaxNet(**base_net_kwargs)
        self.constrains = constrains

    def __call__(self, u, x, y=None):
        net = random_flax_module("nn", self.module, np_dist.Normal(0, 1.), input_shape=u.shape)
        theta = numpyro.sample("theta", np_dist.Normal(0, 1.).expand((self.module.output_dim, x.shape[1])).to_event(2))
        with numpyro.plate("fullBatch", x.shape[0]):
            rohs = net(u)
            rohs += x @ theta.T
            params = {k: constrain(rohs[..., i]) for i, (k, constrain) in enumerate(self.constrains.items())}
            numpyro.sample('y', self.dist(**params), obs=y)
