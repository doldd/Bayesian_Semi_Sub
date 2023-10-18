import numpy as np
import torch
from torch import nn
from torchmetrics import MetricCollection

base_net_kwargs = {"dimensions": [16, 16],
                   "output_dim": 1,
                   "input_dim": 2}


def features(x):
    return np.hstack([x[:, None] / 2.0, (x[:, None] / 2.0) ** 2, ])


# dataloader
def get_ds_from_df(df, device, apply_znorm=True):
    f = features(df.loc[:, ('data', 'x')].to_numpy()).astype(np.float32)
    f_mean, f_std = None, None
    if apply_znorm:
        f_mean = f.mean(axis=0)
        f_std = f.std(axis=0)
        f = (f - f_mean) / f_std
    print(f"f_mean = {f_mean}")
    print(f"f_std = {f_std}")
    y = df.loc[:,('data', 'y')].to_numpy().astype(np.float32)
    meta_ds = df.meta.astype(np.float32).to_numpy()
    # meta_ds = pd.get_dummies(df['category'], drop_first=True).astype(np.float32).to_numpy()
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(f).to(device=device),
                                             torch.from_numpy(meta_ds).to(device=device),
                                             torch.from_numpy(y).to(device=device))
    return dataset, (f_mean, f_std)


def get_ds_test_from_df(df, device, f_mean=None, f_std=None):
    f = features(df.loc[:, ('data', 'x')].to_numpy()).astype(np.float32)
    if (type(f_mean) != type(None)) and (type(f_std) != type(None)):
        f = (f - f_mean) / f_std
    y = df.loc[:, ('data', 'y')].to_numpy().astype(np.float32)
    meta_ds = df.meta.astype(np.float32).to_numpy()
    # meta_ds = pd.get_dummies(df['category'], drop_first=True).astype(np.float32).to_numpy()
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(f).to(device=device),
                                             torch.from_numpy(meta_ds).to(device=device),
                                             torch.from_numpy(y).to(device=device))
    return dataset


def get_curve_space_torch(model_curve_):
    """
    Computes the subspace from the curve parameters.
    The subspace is constructed similar to Izmailov. => 3 Control points spanned the subspace using dot products.

    :param model_curve: curve model which includes the control points symbolized with _# framing
    :return: mean (#p), cov(2,#p)
    """
    w0 = torch.tensor([])
    w12 = torch.tensor([])
    w2 = torch.tensor([])
    for n, p in model_curve_.named_parameters():
        if '_0' in n.split('.')[-1]:
            w0 = torch.hstack([w0, p.detach().clone().flatten()])
        elif '_1' in n.split('.')[-1]:
            w12 = torch.hstack([w12, p.detach().clone().flatten()])
        elif '_2' in n.split('.')[-1]:
            w2 = torch.hstack([w2, p.detach().clone().flatten()])
    mean = torch.mean(torch.vstack([w0, w12, w2]), dim=0)
    u = w2 - w0
    u /= torch.norm(u)
    v = w12 - w0
    v -= u * torch.dot(u, v)
    v /= torch.norm(v)
    return mean, torch.vstack([u[None, :], v[None, :]])


def span_space_from_curve_model(model_curve, num_bends=3):
    """
    Computes the subspace from the curve parameters.
    The subspace is spanned by the singular vectors of the control points. => dim = num_bends-1

    :param model_curve: curve model which includes the control points symbolized with _# framing
    :param num_bends: number of control points
    :return: mean (#p), cov(dim,#p)
    """
    all_curve_params = [np.array([])] * num_bends
    for n, p in model_curve.named_parameters():
        control_point_i = n.split('.')[-1]
        if '_' in control_point_i:
            control_point_i = int(control_point_i.split('_')[1])
            all_curve_params[control_point_i] = np.hstack(
                [all_curve_params[control_point_i], p.detach().clone().flatten().numpy()])
    all_curve_params = np.array(all_curve_params)
    all_curve_params = torch.from_numpy(all_curve_params)
    mean = all_curve_params.mean(0, keepdim=True)
    U, S, Vh = torch.linalg.svd(all_curve_params - mean)
    cov = Vh[:(num_bends - 1)]
    return mean.squeeze(), cov, S

class expCollector():
    def __init__(self, wandb_project, use_ortho, seed, base_net_kwargs, net_kwargs, nll_fn, num_bends=3,
                 max_epochs=1000, metric_collection:MetricCollection=None):
        self.wandb_project = wandb_project
        self.use_ortho = use_ortho
        self.seed = seed
        self.base_net_kwargs = base_net_kwargs
        self.net_kwargs = net_kwargs
        self.hue_order = ['cat1', 'cat2', 'cat3']
        self.df = None
        self.df_test = None
        self.f_mean = None
        self.f_std = None
        self.train_dataloader = None
        self.valid_dataloader = None
        self.test_dataloader = None
        self.num_bends = num_bends
        self.nll_fn = nll_fn
        self.max_epochs = max_epochs
        self.metric_collection = metric_collection

    def set_df(self, df, df_test):
        self.df = df
        self.df_test = df_test

    def __repr__(self):
        str = (f"{key}:{value}" if type(value) != dict else f"{key}:\n\t" + "\n\t".join(
            (f"{k2}:{v2}" for k2, v2 in value.items())) for key, value in vars(self).items())
        str = "\n".join(str)
        return str
    
def log_pointwise_predictive_likelihood(model, posterior_samples, log_like_node='obs', **model_kwargs):
    from pyro import poutine
    num_samples = list(posterior_samples.values())[0].shape[0]
    log_probs = []
    samples = [
        {k: v[i] for k, v in posterior_samples.items()} for i in range(num_samples)
    ]

    for i in range(num_samples):
        trace = poutine.trace(poutine.condition(model, samples[i])).get_trace(**model_kwargs)
        trace.compute_log_prob()
        log_probs.append(trace.nodes[log_like_node]["log_prob"])
    return torch.stack(log_probs)