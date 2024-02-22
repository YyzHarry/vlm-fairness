import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netcal.metrics
import statsmodels.stats.weightstats as st
import seaborn as sns
from tqdm import tqdm
from scipy import stats
from sklearn.utils import resample
from sklearn.metrics import (
    auc, roc_curve, precision_recall_curve, accuracy_score, confusion_matrix, roc_auc_score, average_precision_score,
    balanced_accuracy_score, recall_score, brier_score_loss, classification_report
)


def sigmoid(x):
    z = 1/(1 + np.exp(-x))
    return z


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    expand = target.expand(-1, max(topk))
    correct = pred.eq(expand)
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def compute_mean(stats, is_df=True):
    spec_labels = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
    if is_df:
        spec_df = stats[spec_labels]
        res = np.mean(spec_df.iloc[0])
    else:
        # cis is df, within bootstrap
        vals = [stats[spec_label][0] for spec_label in spec_labels]
        res = np.mean(vals)
    return res


def choose_operating_point(fpr, tpr):
    # J = sens + spec - 1 = tpr - fpr
    sens = 0
    spec = 0
    J = 0
    for _fpr, _tpr in zip(fpr, tpr):
        if _tpr - _fpr > J:
            sens = _tpr
            spec = 1 - _fpr
            J = _tpr - _fpr
    return sens, spec


def plot_roc(y_pred, y_true, roc_name, plot=False):
    # given the test_ground_truth, and test_predictions
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    if plot:
        plt.figure(dpi=100)
        plt.title(roc_name)
        plt.plot(fpr, tpr, 'b', label=f"AUC = {roc_auc:.2f}")
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
    return fpr, tpr, thresholds, roc_auc


def plot_pr(y_pred, y_true, pr_name, plot=False):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    # plot the precision-recall curves
    baseline = len(y_true[y_true == 1]) / len(y_true)

    if plot:
        plt.figure(dpi=20)
        plt.title(pr_name)
        plt.plot(recall, precision, 'b', label=f"AUC = {pr_auc:.2f}")
        plt.legend(loc='lower right')
        plt.plot([0, 1], [baseline, baseline], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.show()
    return precision, recall, thresholds


def hypo_test_two_lists(list1, list2, show_plots=False):
    print(f"List1: {np.mean(list1):.4f} ({np.std(list1):.4f})")
    print(f"List2: {np.mean(list2):.4f} ({np.std(list2):.4f})")
    print("======================================")
    # hypothesis test: unpaired two-sample t-test
    alpha, confidence = 0.05, 0.95
    # Step 1: estimate distribution to be (approximately) normal from samples
    if show_plots:
        plt.figure()
        sns.kdeplot(list1, shade=True, color='r', label='List1')
        sns.kdeplot(list2, shade=True, color='b', label='List2')
        plt.title(f'Data distribution from [List1: {len(list1)}], [List2: {len(list2)}] samples')
        plt.show()
    # Step 2: F-test
    _, p_val_f = stats.bartlett(list1, list2)
    equal_var = False if p_val_f < alpha else True
    usevar = 'unequal' if p_val_f < alpha else 'pooled'
    print(f'F-test P value: [{p_val_f}]')
    print("Variances for two population are same") if equal_var else print("Variances for two population are different")
    print("======================================")
    # Step 3: t-test
    _, p_val, dof = st.ttest_ind(list1, list2, usevar=usevar)
    # _, p_val = stats.ttest_ind(list1, list2, equal_var=equal_var)
    sem = np.sqrt(np.var(list1)/np.sqrt(len(list1)) + np.var(list2)/np.sqrt(len(list2)))
    ci = stats.t.interval(confidence, dof, np.mean(list1) - np.mean(list2), sem)
    print(f't-test: P value [{p_val}]; Confidence Interval [{ci}]')
    return p_val, ci


def hypo_test_two_lists_nonparametric(list1, list2):
    print(f"List1: {np.mean(list1):.4f} ({np.std(list1):.4f})")
    print(f"List2: {np.mean(list2):.4f} ({np.std(list2):.4f})")
    print("======================================")
    # hypothesis test: Wilcoxon rank-sum statistic
    _, p_val = stats.ranksums(list1, list2)
    print(f'Wilcoxon rank-sum test: P value [{p_val}]')
    return p_val


def paired_test_two_lists_nonparametric(list1, list2):
    print(f"List1: {np.mean(list1):.4f} ({np.std(list1):.4f})")
    print(f"List2: {np.mean(list2):.4f} ({np.std(list2):.4f})")
    print("======================================")
    # hypothesis test: Wilcoxon signed-rank test
    _, p_val = stats.wilcoxon(list1, list2)
    print(f'Wilcoxon signed-rank test: P value [{p_val}]')
    return p_val


def evaluate(y_pred, y_true, cxr_labels, label_idx_map=None, verbose=False):
    """
    We expect `y_pred` and `y_true` to be numpy arrays, both of shape (num_samples, num_classes)

    `y_pred` is a numpy array consisting of probability scores with all values in range 0-1.
    `y_true` is a numpy array consisting of binary values representing if a class is present in the cxr.

    This function provides all relevant evaluation information, ROC, AUROC, Sensitivity, Specificity,
    PR-Curve, Precision, Recall for each class.
    """
    import warnings
    warnings.filterwarnings('ignore')
    # number of total labels
    num_classes = y_pred.shape[-1]
    dataframes = []
    for i in range(num_classes):
        if label_idx_map is None:
            y_pred_i = y_pred[:, i]  # (num_samples,)
            y_true_i = y_true[:, i]  # (num_samples,)
        else:
            y_pred_i = y_pred[:, i]  # (num_samples,)
            true_index = label_idx_map[cxr_labels[i]]
            y_true_i = y_true[:, true_index]  # (num_samples,)

        cxr_label = cxr_labels[i]

        # ROC Curve
        fpr, tpr, thresholds, roc_auc = plot_roc(y_pred_i, y_true_i, roc_name=f"{cxr_label} ROC Curve")
        sens, spec = choose_operating_point(fpr, tpr)

        results = [[roc_auc]]
        df = pd.DataFrame(results, columns=[cxr_label+'_auc'])
        dataframes.append(df)
        if verbose:
            print(f"[{cxr_label}]\n\tAUC: {roc_auc:.4f}\n\tSensitivity: {sens:.4f}\n\tSpecificity: {spec:.4f}")

        # Precision-Recall Curve
        precision, recall, thresholds = plot_pr(y_pred_i, y_true_i, pr_name=f"{cxr_label} Precision-Recall Curve")

    dfs = pd.concat(dataframes, axis=1)
    return dfs


def compute_cis(data, confidence_level=0.05):
    """
    Given a Pandas dataframe of (n, labels), return another Pandas dataframe that is (3, labels).
    Each row is lower bound, mean, upper bound of a confidence interval with `confidence`.

    Args:
        * data: Pandas Dataframe, of shape (num_bootstrap_samples, num_labels)
        * confidence_level (optional): confidence level of interval

    Returns:
        * Pandas Dataframe, of shape (3, labels), representing mean, lower, upper
    """
    data_columns = list(data)
    intervals = []
    for i in data_columns:
        series = data[i]
        sorted_perfs = series.sort_values()
        lower_index = int(confidence_level/2 * len(sorted_perfs)) - 1
        upper_index = int((1 - confidence_level/2) * len(sorted_perfs)) - 1
        lower = sorted_perfs.iloc[lower_index].round(4)
        upper = sorted_perfs.iloc[upper_index].round(4)
        mean = round(sorted_perfs.mean(), 4)
        interval = pd.DataFrame({i: [mean, lower, upper]})
        intervals.append(interval)
    intervals_df = pd.concat(intervals, axis=1)
    intervals_df.index = ['mean', 'lower', 'upper']
    return intervals_df


def bootstrap(y_pred, y_true, cxr_labels, n_samples=1000, label_idx_map=None):
    """
    Randomly sample with replacement from y_pred and y_true then evaluate `n` times and obtain AUROC scores for each.
    You can specify the number of samples that should be used with the `n_samples` parameter.
    Confidence intervals will be generated from each of the samples.

    Note:
    * n_total_labels >= n_cxr_labels
        `n_total_labels` is greater iff alternative labels are being tested
    """
    np.random.seed(97)
    idx = np.arange(len(y_true))

    boot_stats = []
    for i in tqdm(range(n_samples)):
        sample = resample(idx, replace=True, random_state=i)
        y_pred_sample = y_pred[sample]
        y_true_sample = y_true[sample]

        sample_stats = evaluate(y_pred_sample, y_true_sample, cxr_labels, label_idx_map=label_idx_map)
        boot_stats.append(sample_stats)

    # pandas array of evaluations for each sample
    boot_stats = pd.concat(boot_stats)
    return boot_stats, compute_cis(boot_stats)


def bootstrap_auc(preds, targets, metric='AUROC', num_iters=1000):
    np.random.seed(97)
    label_set = np.unique(targets)
    bootstrap_means = []

    for _ in range(num_iters):
        indices = np.random.choice(len(preds), size=len(preds), replace=True)
        res = prob_metrics(targets[indices], preds[indices], label_set)
        bootstrap_means.append(res[metric])

    mean_diff = np.mean(bootstrap_means)
    lower = np.percentile(bootstrap_means, 2.5)
    upper = np.percentile(bootstrap_means, 97.5)
    print(f"\tBootstrap {metric}: {mean_diff:.2f} ({lower:.2f}, {upper:.2f})")
    return mean_diff, (lower, upper), bootstrap_means


def bootstrap_fairness(preds, targets, thres=0.5, metric='FNR', num_iters=1000):
    np.random.seed(97)
    preds_rounded = preds >= thres if preds.squeeze().ndim == 1 else preds.argmax(1)
    bootstrap_means = []

    for _ in range(num_iters):
        indices = np.random.choice(len(preds), size=len(preds), replace=True)
        res = binary_metrics(targets[indices], preds_rounded[indices])
        bootstrap_means.append(res[metric])

    mean_diff = np.mean(bootstrap_means)
    lower = np.percentile(bootstrap_means, 2.5)
    upper = np.percentile(bootstrap_means, 97.5)
    print(f"\tBootstrap {metric}: {mean_diff:.2f} ({lower:.2f}, {upper:.2f})")
    return mean_diff, (lower, upper), bootstrap_means


def bootstrap_fairness_diff(preds, targets, attrs, attr1=None, attr2=None, thres=0.5,
                            mode='max', metric='FNR', num_iters=1000):
    assert mode in ['max', 'select']
    np.random.seed(97)
    bootstrap_means = []

    if mode == 'select':
        # select two attrs
        preds1, preds2 = preds[attrs == attr1], preds[attrs == attr2]
        targets1, targets2 = targets[attrs == attr1], targets[attrs == attr2]
        for _ in range(num_iters):
            indices1 = np.random.choice(len(preds1), size=len(preds1), replace=True)
            res1 = binary_metrics(targets1[indices1], preds1[indices1])
            indices2 = np.random.choice(len(preds2), size=len(preds2), replace=True)
            res2 = binary_metrics(targets2[indices2], preds2[indices2])
            bootstrap_means.append(np.abs(res1[metric] - res2[metric]))
    else:
        for _ in range(num_iters):
            indices = np.random.choice(len(preds), size=len(preds), replace=True)
            res = eval_metrics(preds[indices], targets[indices], attrs[indices], thres)
            all_vals = [res['per_attribute'][a][metric] for a in res['per_attribute']]
            bootstrap_means.append(np.max(all_vals) - np.min(all_vals))

    mean_diff = np.mean(bootstrap_means)
    lower = np.percentile(bootstrap_means, 2.5)
    upper = np.percentile(bootstrap_means, 97.5)
    print(f"\tBootstrap fairness diff: {mean_diff:.2f} ({lower:.2f}, {upper:.2f})")
    return mean_diff, (lower, upper), bootstrap_means


def eval_metrics(preds, targets, attributes, thres, add_arrays=False):
    label_set = np.unique(targets)
    preds_rounded = preds >= thres if preds.squeeze().ndim == 1 else preds.argmax(1)

    res = {}
    res['overall'] = {
        **binary_metrics(targets, preds_rounded, label_set),
        **prob_metrics(targets, preds, label_set)
    }
    res['per_attribute'] = {}
    res['per_class'] = {}

    # per attribute results
    for a in np.unique(attributes):
        mask = attributes == a
        res['per_attribute'][int(a)] = {
            **binary_metrics(targets[mask], preds_rounded[mask], label_set),
            **prob_metrics(targets[mask], preds[mask], label_set)
        }

    # per class binary results
    classes_report = classification_report(targets, preds_rounded, output_dict=True, zero_division=0.)
    res['overall'][f'macro_avg'] = classes_report['macro avg']
    res['overall'][f'weighted_avg'] = classes_report['weighted avg']
    for y in np.unique(targets):
        res['per_class'][int(y)] = classes_report[str(y)]

    # per class AUROC
    if preds.squeeze().ndim == 1:  # 2 classes
        res['per_class'][1]['AUROC'] = roc_auc_score(targets, preds, labels=[0, 1])
        res['per_class'][0]['AUROC'] = res['per_class'][1]['AUROC']
    else:
        for y in np.unique(targets):
            new_label = targets == y
            new_preds = preds[:, int(y)]
            res['per_class'][int(y)]['AUROC'] = roc_auc_score(new_label, new_preds, labels=[0, 1])

    res['min_attr'] = pd.DataFrame(res['per_attribute']).min(axis=1).to_dict()
    res['max_attr'] = pd.DataFrame(res['per_attribute']).max(axis=1).to_dict()
    res['max_gap'] = (pd.DataFrame(res['per_attribute']).max(axis=1) -
                      pd.DataFrame(res['per_attribute']).min(axis=1)).to_dict()

    if add_arrays:
        res['y'] = targets
        res['a'] = attributes
        res['preds'] = preds

    return res


def evaluate_attr(preds, targets):
    label_set = np.unique(targets)
    res = {
        'overall': prob_metrics(targets, preds, label_set),
        'per_class_AUROC': {}
    }

    # per class AUROC
    if preds.squeeze().ndim == 1:  # 2 classes
        res['per_class_AUROC'][1] = roc_auc_score(targets, preds, labels=[0, 1])
        res['per_class_AUROC'][0] = res['per_class_AUROC'][1]
    else:
        for y in np.unique(targets):
            new_label = targets == y
            new_preds = preds[:, int(y)]
            res['per_class_AUROC'][int(y)] = roc_auc_score(new_label, new_preds, labels=[0, 1])

    return res


def binary_metrics(targets, preds, label_set=[0, 1], return_arrays=False):
    if len(targets) == 0:
        return {}

    res = {
        'accuracy': accuracy_score(targets, preds),
        'n_samples': len(targets)
    }

    if len(label_set) == 2:
        CM = confusion_matrix(targets, preds, labels=label_set)

        res['TN'] = CM[0][0].item()
        res['FN'] = CM[1][0].item()
        res['TP'] = CM[1][1].item()
        res['FP'] = CM[0][1].item()

        res['error'] = res['FN'] + res['FP']

        if res['TP'] + res['FN'] == 0:
            res['TPR'] = 0
            res['FNR'] = 1
        else:
            res['TPR'] = res['TP']/(res['TP']+res['FN'])
            res['FNR'] = res['FN']/(res['TP']+res['FN'])

        if res['FP'] + res['TN'] == 0:
            res['FPR'] = 1
            res['TNR'] = 0
        else:
            res['FPR'] = res['FP']/(res['FP']+res['TN'])
            res['TNR'] = res['TN']/(res['FP']+res['TN'])

        res['pred_prevalence'] = (res['TP'] + res['FP']) / res['n_samples']
        res['prevalence'] = (res['TP'] + res['FN']) / res['n_samples']
    else:
        res['TPR'] = recall_score(targets, preds, labels=label_set, average='macro', zero_division=0.)

    if len(np.unique(targets)) > 1:
        res['balanced_acc'] = balanced_accuracy_score(targets, preds)

    if return_arrays:
        res['targets'] = targets
        res['preds'] = preds

    return res


def prob_metrics(targets, preds, label_set, return_arrays=False):
    if len(targets) == 0:
        return {}

    res = {
        'ECE': netcal.metrics.ECE().measure(preds, targets)
    }

    if len(set(targets)) > 2:
        try:
            res['AUROC'] = roc_auc_score(targets, preds, multi_class='ovr', labels=label_set)
        except:
            res['AUROC'] = roc_auc_score(targets, preds, multi_class='ovo', labels=label_set)
    elif len(set(targets)) == 2:
        res['AUROC'] = roc_auc_score(targets, preds, labels=label_set)
    elif len(set(targets)) == 1:
        res['AUROC'] = None

    if len(set(targets)) == 2:
        res['AUPRC'] = average_precision_score(targets, preds, average='macro')
        res['brier'] = brier_score_loss(targets, preds)
        res['mean_pred_1'] = preds[targets == 1].mean()
        res['mean_pred_0'] = preds[targets == 0].mean()

    if return_arrays:
        res['targets'] = targets
        res['preds'] = preds

    return res
