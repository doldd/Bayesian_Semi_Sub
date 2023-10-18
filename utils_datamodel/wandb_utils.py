import copy

import wandb
import torch
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd
import os

sns.set_style("darkgrid")


def parse_runs(entity, project, my_runs, art_types=["due", 'wideResNet']):
    api = wandb.Api()
    runs = api.runs(entity + "/" + project)
    parsed_runs = dict()
    for run in runs:
        if run.id in my_runs:
            config = {}
            config['config'] = {k: v for k, v in run.config.items() if not k.startswith('_')}
            config['id'] = run.id
            config['name'] = run.name
            config['summary'] = run.summary

            for a in run.logged_artifacts():
                if a.type in art_types:
                    # art_dir = a.download()
                    config['art_name'] = a.name
                    config['art_type'] = a.type
                    # config['art_dir'] = art_dir
            parsed_runs[run.name + '_' + run.id] = config
    return parsed_runs


def load_model(wandb_run, wandb_path, strict=True, model_cls=None, file_name='last.ckpt', **model_kwargs):
    """

    :param wandb_run:
    :param wandb_path:
    :param strict:
    :param model_cls: Model class
    :param file_name: filename of artifact if None uses the last file in directory
    :param model_kwargs: overrides config parameters stored in wandb
    :return: tuple(model, config)
    """
    api = wandb.Api(timeout=10)
    run = api.run(wandb_path)
    print(f"Name: {run.name}, ID: {run.id}")
    config = {}
    config['config'] = {k: v for k, v in run.config.items() if not k.startswith('_')}
    config['id'] = run.id
    config['name'] = run.name
    config['summary'] = run.summary
    for a in run.logged_artifacts():
        if a.type in ["due", 'wideResNet', model_cls.__name__]:
            config['art_name'] = a.name
            config['art_type'] = a.type
    print("This run used following artifacts:")
    for a in run.used_artifacts():
        print(a.name)
        try:
            print(a.logged_by())
        except Exception as e:
            print("Exeption: ", e)
    valid_acc = config['summary'].get('valid/acc_epoch', None)
    if valid_acc is None:
        valid_acc = config['summary'].get('valid/acc', None)
    if valid_acc is None:
        valid_acc = config['summary'].get('valid/Accuracy', None)
    print("Last Acc", valid_acc)
    print("loaded artifact: ")
    print(config['art_type'])
    print(config['art_name'])

    artifact = wandb_run.use_artifact(config['art_name'], type=config['art_type'])
    art_dir = artifact.download()
    if model_cls is None:
        raise DeprecationWarning("you must define a model_cls type. It is not any more inherit automatically")
        # if config['art_type'] == 'due':
        #     model = DueDkl.load_from_checkpoint(os.path.join(art_dir, 'last.ckpt'), **config['config'], strict=strict)
        # else:
        #     model = DueVanilla.load_from_checkpoint(os.path.join(art_dir, 'last.ckpt'), **config['config'], strict=strict)
    else:
        if file_name is None:
            files = os.listdir(art_dir)
            file_name = files[-1]
        kwargs = copy.deepcopy(config['config'])
        for k, i in model_kwargs.items():
            kwargs[k] = i
        model = model_cls.load_from_checkpoint(os.path.join(art_dir, file_name), **kwargs,
                                               strict=strict)
    return model, config


def log_in_dist_performance(trainer, model, wandb_run, dataloader_train, dataloader_test):
    pred_out = trainer.predict(model=model, dataloaders=dataloader_train)
    prob = torch.vstack([p[0] for p in pred_out])
    y = torch.hstack([p[1] for p in pred_out]).cpu().numpy()
    entropy_train = -(prob * torch.log(prob.clamp(min=1e-8))).sum(-1).cpu().numpy()
    table = wandb.Table(data=np.stack([entropy_train, y], axis=-1).tolist(), columns=["entropy", "label"])
    histogram = wandb.plot_table(vega_spec_name="ddold/histogramm_v1",
                                 data_table=table,
                                 fields=dict(groupKeys="name", value="entropy"),
                                 string_fields=dict(title="Entropy on CIFAR10 train data"))
    wandb_run.log({"entropy/train": histogram}, commit=True)

    pred_out = trainer.predict(model=model, dataloaders=dataloader_test)
    prob = torch.vstack([p[0] for p in pred_out])
    y = torch.hstack([p[1] for p in pred_out]).cpu().numpy()
    entropy_test = -(prob * torch.log(prob.clamp(min=1e-8))).sum(-1).cpu().numpy()
    table = wandb.Table(data=np.stack([entropy_test, y], axis=-1).tolist(), columns=["entropy", "label"])
    histogram = wandb.plot_table(vega_spec_name="ddold/histogramm_v1",
                                 data_table=table,
                                 fields=dict(groupKeys="name", value="entropy"),
                                 string_fields=dict(title="Entropy on CIFAR10 test data"))
    wandb_run.log({"entropy/test": histogram}, commit=True)
    return entropy_train, entropy_test


def log_ood_performance(trainer, model, wandb_run, dataloader_ood, entropy_in, name="CIFAR10test_SVHN"):
    pred_out = trainer.predict(model=model, dataloaders=dataloader_ood)
    prob = torch.vstack([p[0] for p in pred_out])
    y = torch.hstack([p[1] for p in pred_out]).cpu().numpy()
    entropy_ood = -(prob * torch.log(prob.clamp(min=1e-8))).sum(-1).cpu().numpy()
    table = wandb.Table(data=np.stack([entropy_ood, y], axis=-1).tolist(), columns=["entropy", "label"])
    histogram = wandb.plot_table(vega_spec_name="ddold/histogramm_v1",
                                 data_table=table,
                                 fields=dict(groupKeys="name", value="entropy"),
                                 string_fields=dict(title="Entropy on SVHN train data"))
    wandb_run.log({"entropy/ood": histogram}, commit=True)

    fig = plt.figure()
    ents = np.hstack([entropy_in,
                      entropy_ood])
    mark_in_dist = np.hstack([np.zeros_like(entropy_in),
                              np.ones_like(entropy_ood)])
    roc_auc = roc_auc_score(mark_in_dist, ents)
    fpr, tpr, th = roc_curve(mark_in_dist, ents)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f}')
    plt.title('Receiver operating characteristic example')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    wandb_run.log({f"{name}/OOD_ROC": fig})
    wandb_run.log({f"{name}/OOD_ROC_image": wandb.Image(fig)})
    fig = plt.figure()
    data_names = np.hstack([np.full(entropy_in.shape, "CIFAR10:test"),
                            np.full(entropy_ood.shape, name)])
    entropy_df = pd.DataFrame.from_dict({'entropy': ents,
                                         'dataset': data_names})
    p = np.full(10, 1. / 10)
    entropy_max = -np.sum(p * np.log(p))
    sns.histplot(data=entropy_df, x="entropy", hue='dataset', stat='probability', binrange=(0., entropy_max))
    wandb_run.log({f"{name}/Entropy_compare": wandb.Image(fig)})
    wandb_run.summary.update({f"{name}/OOD_ROC_AUC_TEST": roc_auc})
    return entropy_ood


def wandb_table_to_dataframe(wandb_run, artifact_path: str, table_name='ess_samples'):
    if wandb_run is None:
        art = wandb.Api().artifact(artifact_path)
        table = art.get(table_name)
        return pd.DataFrame(table.data, columns=table.columns)
    else:
        artifact = wandb_run.use_artifact(artifact_path, type='run_table')
        artifact_dir = artifact.download()
        table = artifact.get(table_name)
        return pd.DataFrame(table.data, columns=table.columns)
