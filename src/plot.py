from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import bezier
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm
from src.semi_sub_utils import expCollector, get_ds_from_df, get_ds_test_from_df, span_space_from_curve_model, features
from src.semiSub_model import getModel
from src.base_models import RegNet
import wandb
from torchmetrics import MetricCollection, MeanSquaredError, MeanAbsoluteError


# https://jwalton.info/Embed-Publication-Matplotlib-Latex/
tex_fonts = {
    # Use LaTeX to write all text
    "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{amsfonts}\usepackage{amsmath}\usepackage{amssymb}",  #  for the align enivironment
    "text.usetex": True,  # use inline math for ticks
    "font.family": "serif",
    "pgf.rcfonts": False,  # don't setup fonts from rc parameters
    "savefig.transparent": True,
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 9,
    "font.size": 9,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
}
plt.rcParams.update(tex_fonts)
textwidth = 430
# columnwidth = 252

# https://jwalton.info/Embed-Publication-Matplotlib-Latex/
def set_size(width=textwidth, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)
# height = set_size(textwidth, fraction=1 / 2)[0]


def plot_subspace(df, df_name, t0=None, t_bend=None, t2=None, aspect_equal=True, linear_color=False, df_samples=None,
                  interpolate=False, vmax=None, vmin=None):
    """
    :param df: pandas dataframe (must contain xx and yy columns for axis)
    :param df_name: column name of data
    :param t0: two dimensional point P1 of Bézier curve
    :param t_bend: two dimensional point Bend point of Bézier curve
    :param t2: two dimensional point P2 of Bézier curve
    :param aspect_equal: Draw x and y axis in same scale
    :param linear_color: if True use linear coloring else linear color steps according their quantiles
    :param df_samples: pandas dataframe of samples (must contain xx and yy columns for axis) or tuple with (dataframe, xname, yname)
    :param interpolate: if True use counterf to plot else pcolormesh
    :param vmax: max value of coloring
    :param vmin: min value of coloring
    """
    plt.rcParams['axes.grid'] = False
    fig = plt.figure(figsize=set_size(), dpi=150)
    ax = fig.gca()
    cmap = plt.colormaps['viridis']
    if vmax is None:
        vmax = df[df_name].max()
    if vmin is None:
        vmin = df[df_name].min()
    if linear_color:
        norm = colors.Normalize(clip=True, vmax=vmax, vmin=vmin)
    else:
        # compute boundaries according data quantile (step_size=1/255)
        levels = [(df[(df[df_name] <= vmax) & (df[df_name] >= vmin)][df_name]).quantile(1 / cmap.N * i) for i in range(cmap.N)]
        levels.append(vmax)
        # df_min = df[df_name].min()
        # shrink = (vmax - vmin) / (df[df_name].max() - df_min)
        # levels = shrink * (np.array(levels) - df_min) + vmin
        norm = colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    shape = np.sqrt(len(df['xx'].to_numpy())).astype(np.int32)
    shape = (shape, shape)
    xx = df['xx'].to_numpy().reshape(shape)
    yy = df['yy'].to_numpy().reshape(shape)
    zz = df[df_name].to_numpy().reshape(shape)
    if interpolate:
        if linear_color:
            pcm = ax.contourf(xx, yy, zz, levels=cmap.N, cmap=cmap, norm=norm)
        else:
            pcm = ax.contourf(xx, yy, zz, levels=levels, cmap=cmap, norm=norm)
    else:
        pcm = ax.pcolormesh(xx, yy, zz, cmap=cmap, norm=norm, shading='nearest')
    if aspect_equal:
        ax.set_aspect('equal')
    if linear_color:
        fig.colorbar(pcm, ax=ax, label=df_name, shrink=0.5, aspect=20*0.5, format="%1.2f")
    else:
        cb = fig.colorbar(pcm, ax=ax, label=df_name, shrink=0.5, aspect=20*0.5, format="%1.2f", ticks=levels[::50])
        cb.ax.minorticks_off()
    if t0 is not None and t2 is not None:
        ax.scatter(*t0, c='red')
        ax.scatter(*t2, c='red')
        if t_bend is not None:
            # ax.scatter(*t_bend, c='red')
            bz = bezier.BezierSegment(np.vstack([t0, t_bend, t2]))
            curve_t = np.array([bz.point_at_t(i) for i in np.linspace(0, 1, 100)]).T
            ax.plot(*curve_t, c='red', label="Bezier curve", linewidth=1.)

    # plot samples
    if (df_samples is not None) and isinstance(df_samples, pd.DataFrame):
        sns.scatterplot(data=df_samples, x='theta1', y='theta2', color='Orange', alpha=0.4, linewidth=.0,
                        label="Samples", s=10.)
    if (df_samples is not None) and isinstance(df_samples, tuple):
        df_samples, xname, yname = df_samples
        sns.scatterplot(data=df_samples, x=xname, y=yname, color='Orange', alpha=0.4, linewidth=.0,
                        label="Samples", s=10.)
        plt.xlabel(xname)
        plt.ylabel(yname)
    plt.legend()
    plt.title(df_name)
    plt.tight_layout()
    return fig


def exclude_project_code_dirs(path):
    if "wandb/" in path:
        return True
    elif "drbayes/" in path:
        return True
    elif "dnn-mode-connectivity/" in path:
        return True
    else:
        return False
    return False


def run_metrics_on_samples(ess, samples, loader, device, metric_collection):
    prob_s_b = torch.empty((len(samples), len(loader.dataset)), device=device)  # shape (#s, #ds)
    bs = 0
    with torch.no_grad():
        for data, y in tqdm(loader):
            y = y.to(device=device) if y.device is not device else y
            bl = len(y)
            out_s = torch.empty((len(samples), len(y)), device=device)  # shape (#s, #bs)
            for s, sample in enumerate(samples):
                sample = torch.from_numpy(sample).to(device=device)
                ess.base_model.set_parameter_vector(sample)
                out = ess(data).squeeze()
                # out = torch.sigmoid(out)
                # likeli = y*out + (1-y)*(1-out)  # Bernoulli pdf
                prob_s_b[s, bs:bs + bl] = (
                    -torch.nn.functional.binary_cross_entropy_with_logits(out, y, reduction='none')).exp()
                # prob_s_b[s, bs:bs+bl] = likeli
                out_s[s, :] = torch.sigmoid(out)
            metric_collection.update(out_s.mean(0), y.to(dtype=torch.int32))
            bs += bl
    return prob_s_b

def plot_predictive_regression(exp_col: expCollector, model, wandb_logger):
    # plot regression result
    fig = plt.figure(figsize=(9., 7.))
    plot_data(exp_col)
    x_test = np.linspace(-7, 7, 1000)
    f_test = (features(x_test) - exp_col.f_mean) / exp_col.f_std
    with torch.no_grad():
        meta_cat1 = torch.tile(torch.tensor([[0., 0.]]), (len(x_test), 1))
        meta_cat2 = torch.tile(torch.tensor([[1., 0.]]), (len(x_test), 1))
        meta_cat3 = torch.tile(torch.tensor([[0., 1.]]), (len(x_test), 1))
        x_test_t = torch.from_numpy(f_test.astype(np.float32))

        pred_test = model(x_test_t, meta_cat1).squeeze().numpy()
        plt.scatter(x_test, pred_test, label="prediction_cat1", c=sns.color_palette()[0], s=1)

        pred_test = model(x_test_t, meta_cat2).squeeze().numpy()
        plt.scatter(x_test, pred_test, label="prediction_cat2", c=sns.color_palette()[1], s=1)

        pred_test = model(x_test_t, meta_cat3).squeeze().numpy()
        plt.scatter(x_test, pred_test, label="prediction_cat3", c=sns.color_palette()[2], s=1)
    # plt.scatter(data[:,0], data[:,1])
    plt.legend()
    wandb_logger.experiment.log({"prediction": wandb.Image(fig)})


def plot_curve_solutions_regression(exp_col: expCollector, best_curve_model, wandb_logger):
    plt.figure(figsize=(9., 7.))
    device = best_curve_model.device
    # %% plot performance along curve
    dataset_test = get_ds_test_from_df(exp_col.df, device, exp_col.f_mean, exp_col.f_std)
    x_train, meta_train, y_train = dataset_test[:]
    t_space = torch.linspace(0, 1., 101, device=device)
    nll_vs_t = []
    for t in t_space:
        out = best_curve_model.model(x_train, t) + best_curve_model.structure_lin(meta_train)
        nll = exp_col.nll_fn(out, y_train).mean()
        nll_vs_t.append(nll.detach().cpu().item())
    plt.plot(t_space.cpu().numpy(), nll_vs_t)
    plt.xlabel("Bézier curve t-space")
    plt.ylabel(r'nll ~ $N(y|\mu=DNN(),\sigma=0.005)$')
    wandb_logger.experiment.log({"Bézier Curve NLL": wandb.Image(plt.gcf())})

    # %% plot predictive performance
    z = np.linspace(-10, 10, 100)
    feature = (features(z) - exp_col.f_mean) / exp_col.f_std
    inp = torch.from_numpy(feature.astype(np.float32)).to(device=device)
    trajectories = []
    for t in t_space:
        out = best_curve_model.model(inp, t)
        trajectories.append(out.detach().cpu().numpy().ravel())
    trajectories = np.vstack(trajectories)

    def plot_samples(x_axis, preds, ax, color='blue'):
        mu = preds.mean(0)
        sigma = preds.std(0)

        ax.plot(x_axis, mu, "-", lw=2., color=color)
        ax.plot(x_axis, mu - 3 * sigma, "-", lw=0.75, color=color)
        ax.plot(x_axis, mu + 3 * sigma, "-", lw=0.75, color=color)

        np.random.shuffle(preds)
        for traj in preds[:10]:
            ax.plot(x_axis, traj, "-", alpha=.25, color=color, lw=1.)

        ax.fill_between(x_axis, mu - 3 * sigma, mu + 3 * sigma, alpha=0.35, color=color)

    fig = plt.figure(figsize=(9., 7.))
    plot_data(exp_col)
    ax = fig.gca()
    plot_samples(z, trajectories, ax=ax, color=sns.color_palette()[0])
    plt.title("Curve Solution")
    wandb_logger.experiment.log({"prediction": wandb.Image(plt.gcf())})

    # %% plot subspace
    mean, cov, S = span_space_from_curve_model(best_curve_model.model.cpu(), exp_col.num_bends)
    subspace_model = getModel(RegNet, mean=mean, cov_factor=cov, **exp_col.base_net_kwargs, **exp_col.net_kwargs).to(
        device=device)
    metric_collection_base = MetricCollection([MeanSquaredError(),
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
            p_t = torch.concat([p_t, torch.zeros(2, device=device)])
            subspace_model.set_parameter_vector(p_t)
            nll = 0.
            for data, y in exp_col.train_dataloader:
                # y = y.cuda() if y.device is not device else y
                p_pred = subspace_model(*tuple(data))
                # nll += subspace_model.loss_fn(p_pred, y).item()
                nll += exp_col.nll_fn(p_pred, y).mean().item()
                metric_collection.update(p_pred.squeeze(), y)
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
                        interpolate=False, vmax=.4)
    wandb_logger.experiment.log({"train_grid_MSE": wandb.Image(fig)})
    fig = plot_subspace(df_grid, "MeanAbsoluteError", cp[0, :2], cp[1:-1, :2], cp[-1, :2], linear_color=True,
                        interpolate=False, vmax=.4)
    wandb_logger.experiment.log({"train_grid_MAE": wandb.Image(fig)})
    w_table = wandb.Table(dataframe=df_grid)
    wandb_logger.experiment.log({"train_grid": w_table})


def plot_subspace_solution_regression(exp_col, ess, logprobs, curve_wandb_id, wandb_logger):
    # visualize samples
    subspace_labels = ['$\\varphi_{NN_1}$', '$\\varphi_{NN_2}$']
    num_structure_params = exp_col.net_kwargs['num_structure'] * exp_col.base_net_kwargs['output_dim']
    structure_subspace_labels = ["$\\theta_{" + str(ss_p) + "}$" for ss_p in range(num_structure_params)]
    subspace_labels.extend(structure_subspace_labels)
    df_samples = pd.DataFrame(ess.all_samples.T, columns=subspace_labels)
    df_samples['logpdf'] = logprobs
    df_samples_w = df_samples.copy()
    df_samples_w.columns = [s.replace("\\", '#') for s in df_samples_w.columns]
    w_table = wandb.Table(dataframe=df_samples_w)
    wandb_logger.experiment.log({"ess_samples": w_table})

    # load subspace grid
    run = wandb_logger.experiment
    runapi = wandb.Api().run(f"{run.entity}/{run.project}/{curve_wandb_id}")
    grid_artifact_path = ""
    for la in runapi.logged_artifacts():
        if (la.type in ['run_table']) and ("train_grid" in la.name):
            grid_artifact_path = f"{run.entity}/{run.project}/{la.name}"
    print("Grid artifact_path to load:", grid_artifact_path)
    df_grid = wandb_table_to_dataframe(run, artifact_path=grid_artifact_path, table_name="train_grid")
    fig = plot_subspace(df_grid, "nll", df_samples=(df_samples, *(subspace_labels[:2])), interpolate=True,
                        linear_color=False)
    wandb_logger.experiment.log({'GridWithSamples': wandb.Image(fig)})

    g = sns.PairGrid(df_samples[subspace_labels], diag_sharey=False, aspect=1, height=5)
    g.map_upper(sns.scatterplot, s=15, linewidth=0., alpha=np.min([(1 / len(df_samples)) * 100., 0.9]))
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot, lw=2)
    plt.tight_layout()
    wandb_logger.experiment.log({'Correlation': wandb.Image(plt.gcf())})

    # Description
    array = df_samples[subspace_labels].to_numpy().reshape(-1, ess.num_samples, len(subspace_labels)).transpose(2, 0, 1)
    samples_dict = {k: v for k, v in zip(subspace_labels, array)}
    description = pd.DataFrame(summary(samples_dict))
    description_w = description.reset_index()
    description_w.columns = [s.replace("\\", '#') for s in description_w.columns]
    w_table2 = wandb.Table(dataframe=description_w)
    wandb_logger.experiment.log({"description": w_table2})


def plot_subspace_solution_regression_pyro(az_post_hmc, pyro_model, dataset, curve_model, mean, cov):
    import pyro
    from pyro.poutine.indep_messenger import IndepMessenger
    # compute grid
    device = pyro_model.device
    u_train, x_train, y_train = dataset[:]
    x = np.linspace(-10, 10, 40, dtype=np.float32)
    y = np.linspace(-8, 8, 40, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    grid = np.vstack([xx.flatten(), yy.flatten()]).T
    cond_theta = torch.from_numpy(az_post_hmc['posterior']['theta'].mean(dim=['chain', 'draw']).to_numpy()).to(
        device=device, dtype=torch.float32)
    with IndepMessenger("grid", size=grid.shape[0], dim=-2):
        cond_model = pyro.condition(pyro_model, data={"structure_nn.weight": cond_theta,
                                                        "varphi": torch.from_numpy(grid).to(device=device)})
        trace = pyro.poutine.trace(cond_model).get_trace(u_train, x_train, y_train)
        trace.compute_log_prob()
    log_like = trace.nodes['obs']['log_prob'].sum(1).detach().cpu().numpy()
    log_prob_joint = log_like.copy()
    log_prob_joint += trace.nodes['structure_nn.weight']['log_prob'].item()  # wasn't broadcasted => single value
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
    p_inv = cov.cpu().numpy().T
    t0 = (w0.cpu().numpy() - mean.cpu().numpy()) @ p_inv
    t12 = (w12.cpu().numpy() - mean.cpu().numpy()) @ p_inv
    t2 = (w2.cpu().numpy() - mean.cpu().numpy()) @ p_inv
    fig = plot_subspace(df, "log_prob_joint", t0, t12, t2, linear_color=False, interpolate=False,
                        vmin=np.quantile(log_prob_joint, 0.8))
    post_varphi = az_post_hmc['posterior']['varphi'].to_numpy().reshape(-1, 2)
    sns.scatterplot(x=post_varphi[:, 0], y=post_varphi[:, 1], alpha=np.min((0.75, 100./post_varphi.shape[0])), linewidth=0., s=3)
    ax = plt.gca()
    ax.get_legend().remove()
    ax.set_xlabel(r"$\varphi_1$")
    ax.set_ylabel(r"$\varphi_2$")


def plot_data(exp_col: expCollector):
    df_combine = pd.concat([exp_col.df, exp_col.df_test], axis=0, keys=['train', 'test'], names=["ds"]).reset_index(0)
    df_combine_plot = df_combine['data'].rename(columns={'x': 'u', 'category': 'category (x)'})
    sns.scatterplot(data=df_combine_plot, x='u', y='y', hue='category (x)', hue_order=exp_col.hue_order,
                    style=df_combine['ds'],
                    markers=['D', '.'], sizes=(5, 20), size=df_combine['ds'])
