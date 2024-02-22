from sklearn.metrics import matthews_corrcoef
from utils.eval import *
from zero_shot import *


def evaluate_model(x_dir, y_dir, model_path, cxr_labels, alt_labels_dict=None):
    context_length = 77

    # templates list of positive and negative template pairs
    cxr_pair_templates = [("{}", "no {}")]

    cxr_results, y_pred = run_zero_shot(
        cxr_labels, cxr_pair_templates, model_path, cxr_filepath=x_dir, final_label_path=y_dir,
        alt_labels_dict=alt_labels_dict, softmax_eval=True, context_length=context_length,
        pretrained=True, use_bootstrap=True, cutlabels=True
    )
    return cxr_results, y_pred


def f1_mcc_bootstrap(y_pred, y_true, cxr_labels, best_p_vals, eval_func, n_samples=5000, label_idx_map=None):
    """
    This function will randomly sample with replacement
    from y_pred and y_true then evaluate `n` times and obtain AUROC scores for each.

    You can specify the number of samples that should be used with the `n_samples` parameter.

    Confidence intervals will be generated from each of the samples.
    """
    idx = np.arange(len(y_true))

    boot_stats = []
    for i in tqdm(range(n_samples)):
        sample = resample(idx, replace=True)
        y_pred_sample = y_pred[sample]
        y_true_sample = y_true[sample]

        sample_stats = eval_func(y_pred_sample, y_true_sample, best_p_vals, cxr_labels=cxr_labels, label_idx_map=label_idx_map)
        boot_stats.append(sample_stats)

    # pandas array of evaluations for each sample
    boot_stats = pd.concat(boot_stats)
    return boot_stats, compute_cis(boot_stats)


def get_best_alt_labels(res_df, cxr_labels):
    best_alt_labels_dict = dict()
    best_alt_labels_vals = dict()
    res_cols = list(res_df)

    curr_path_name = None
    for col in res_cols:  # for each col
        path_name = col.split("_")[0]  # pathology name
        mean_auc = res_df[col][0]  # mean auc

        if path_name in cxr_labels: 
            # reset the vars
            curr_path_name = path_name
            best_alt_labels_dict[path_name] = [path_name]
            best_alt_labels_vals[path_name] = mean_auc

        if best_alt_labels_vals[curr_path_name] < mean_auc:
            best_alt_labels_vals[curr_path_name] = mean_auc
            best_alt_labels_dict[curr_path_name] = [path_name]

    return best_alt_labels_dict


def y_true_csv_to_np(df_path, cxr_labels):
    groundtruth = pd.read_csv(df_path)
    groundtruth = groundtruth[cxr_labels]
    groundtruth = groundtruth.to_numpy()[:, :].astype(int)
    return groundtruth


def get_best_p_vals(pred, groundtruth, cxr_labels, metric_func=matthews_corrcoef, spline_k: int = None, verbose: bool = False):
    """
    WARNING: CXR_LABELS must 
    Params:
    * pred: np arr
        probabilities output by model

    * plot_graphs: bool
        if True, will save plots for metric vs. threshold for each pathology

    Note:
    * `probabilities` value is a linspace of possible probabilities
    """
    probabilities = [val for val in np.arange(0.4, 0.64, 0.0001)]
    best_p_vals = dict()
    for idx, cxr_label in enumerate(cxr_labels):
        y_true = groundtruth[:, idx]
        _, _, probabilities = roc_curve(y_true, pred[:, idx])
        probabilities = probabilities[1:]
        probabilities.sort()
        
        metrics_list = []
        for p in probabilities:
            y_pred = np.where(pred[:, idx] < p, 0, 1)
            metric = metric_func(y_true, y_pred)
            metrics_list.append(metric)

        if spline_k is not None: 
            try:
                from scipy.interpolate import UnivariateSpline
                spl = UnivariateSpline(probabilities, metrics_list, k=spline_k)
                spl_y = spl(probabilities)
                # get optimal thresholds on the spline and on the val_metric_list
                best_index = np.argmax(spl_y)
            except: 
                best_index = np.argmax(metrics_list)
        else:
            best_index = np.argmax(metrics_list)
        
        best_p = probabilities[best_index]
        best_metric = metrics_list[best_index]
        if verbose: 
            print(f"Best metric for {cxr_label} is {best_metric}. threshold = {best_p}.")

        best_p_vals[cxr_label] = best_p
    return best_p_vals


def compute_f1(y_pred, y_true, cxr_labels, thresholds, label_idx_map=None):
    def get_f1_clip_bootstrap(y_pred, y_true, best_p_vals, cxr_labels=cxr_labels, label_idx_map=None):
        stats = {}
        probs = np.copy(y_pred)
        for idx, cxr_label in enumerate(cxr_labels):
            p = best_p_vals[cxr_label]
            probs[:, idx] = np.where(probs[:, idx] < p, 0, 1)
        clip_preds = np.copy(probs)
        for idx, cxr_label in enumerate(cxr_labels):
            if label_idx_map is None: 
                curr_y_true = y_true[:, idx]
            else: 
                curr_y_true = y_true[:, label_idx_map[cxr_label]]
            curr_y_pred = clip_preds[:, idx]

            m = confusion_matrix(curr_y_true, curr_y_pred)
            if len(m.ravel()) == 1:
                tn = 500
                fp = 0
                fn = 0
                tp = 0
            else:
                tn, fp, fn, tp = m.ravel()

            if (2 * tp + fp + fn) == 0:
                stats[cxr_label] = 1
                continue

            stats[cxr_label] = [(2 * tp) / (2*tp + fp + fn)]
        # compute mean over five major pathologies
        stats["Mean"] = compute_mean(stats, is_df=False)
        return pd.DataFrame.from_dict(stats)

    boot_stats, f1_cis = f1_mcc_bootstrap(
        y_pred, y_true, cxr_labels, thresholds, get_f1_clip_bootstrap, n_samples=1000, label_idx_map=label_idx_map)
    return f1_cis


def compute_mcc(y_pred: np.array, y_true: np.array, cxr_labels: List, thresholds: dict, label_idx_map: dict = None):
    def get_mcc_bootstrap(y_pred, y_true, best_p_vals, cxr_labels=cxr_labels, label_idx_map=None):
        stats = {}
        probs = np.copy(y_pred)

        for idx, cxr_label in enumerate(cxr_labels):
            p = best_p_vals[cxr_label]
            probs[:, idx] = np.where(probs[:, idx] < p, 0, 1)

        clip_preds = np.copy(probs)

        for idx, cxr_label in enumerate(cxr_labels):
            if label_idx_map is None:
                curr_y_true = y_true[:, idx]
            else:
                curr_y_true = y_true[:, label_idx_map[cxr_label]]

            curr_y_pred = clip_preds[:, idx]
            stats[cxr_label] = [matthews_corrcoef(curr_y_true, curr_y_pred)]
        # compute mean over five major pathologies
        stats["Mean"] = compute_mean(stats, is_df=False)
        return pd.DataFrame.from_dict(stats)

    boot_stats, mcc_cis = f1_mcc_bootstrap(
        y_pred, y_true, cxr_labels, thresholds, get_mcc_bootstrap, n_samples=1000, label_idx_map=label_idx_map)
    return mcc_cis
