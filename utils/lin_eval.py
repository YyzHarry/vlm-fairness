import torch
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from utils.eval import prob_metrics


def get_representations(model, loader, device):
    atts, zs = [], []

    model.eval()
    with torch.no_grad():
        for _, x, a in loader:
            z = model.encode_image(x.to(device)).detach().cpu().numpy()
            zs.append(z)
            atts.append(a)

    return np.concatenate(zs, axis=0), np.concatenate(atts, axis=0)


def fit_model(train_X, train_Y, val_X, val_Y, test_X, test_Y, model_type='lr'):
    if model_type == 'lr':
        pipe = Pipeline(steps=[
            ('model', LogisticRegression(random_state=42, n_jobs=-1))
        ])
        param_grid = {
            'model__C': 10**np.linspace(-5, 1, 10)
        }
    elif model_type == 'rf':
        pipe = Pipeline(steps=[
            ('model', RandomForestClassifier(random_state=42, n_jobs=-1))
        ])
        param_grid = {
            'model__max_depth': list(range(1, 7))
        }
    else:
        raise NotImplementedError

    pds = PredefinedSplit(test_fold=np.concatenate([np.ones((len(train_X),))*-1, np.zeros((len(val_X),))]))

    cv_lr = (GridSearchCV(pipe, param_grid, refit=False, cv=pds, scoring='roc_auc_ovr', verbose=10, n_jobs=-1).fit(
        np.concatenate((train_X, val_X)), np.concatenate((train_Y, val_Y))))

    pipe = clone(
        clone(pipe).set_params(**cv_lr.best_params_)
    )
    pipe = pipe.fit(train_X, train_Y)

    label_set = np.sort(np.unique(train_Y))
    res = {}
    for sset, X, Y in zip(['va', 'te'], [val_X, test_X], [val_Y, test_Y]):
        preds = pipe.predict_proba(X)
        if len(label_set) == 2:
            preds = preds[:, 1]

        res[sset] = prob_metrics(Y, preds, label_set=label_set, return_arrays=True)
        res[sset]['pred_probs'] = res[sset]['preds']
        del res[sset]['targets']

        # per class AUROC
        if preds.squeeze().ndim == 1:  # 2 classes
            res[sset][f'class_1_AUROC'] = roc_auc_score(Y, preds, labels=[0, 1])
            res[sset][f'class_0_AUROC'] = res[sset][f'class_1_AUROC']
        else:
            for y in np.unique(Y):
                new_label = Y == y
                new_preds = preds[:, int(y)]
                res[sset][f'class_{y}_AUROC'] = roc_auc_score(new_label, new_preds, labels=[0, 1])

    return res
