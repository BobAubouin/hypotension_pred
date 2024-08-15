from pathlib import Path
import multiprocessing as mp

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import auc, roc_curve, average_precision_score
from hp_pred.databuilder import DataBuilder
from optuna import Trial
from tqdm import tqdm


NUMBER_CV_FOLD = 3
N_INTERPOLATION = 1000


def objective_xgboost(
    trial: Trial,
    data: pd.DataFrame,
    feature_name: list[str],
) -> float:
    """
    Calculate the mean AUC score for XGBoost model using Optuna hyperparameter optimization.

    Args:
        trial (Trial): Optuna trial object for hyperparameter optimization.
        data (pd.DataFrame): Input data for training and testing.
        feature_name (list[str]): List of feature names.
        cv_split (list[np.ndarray]): List of arrays containing indices for cross-validation splits.

    Returns:
        float: Mean AUC score of the XGBoost model.

    """
    params = {
        "max_depth": trial.suggest_int("max_depth", 1, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.01, 1.0, log=True
        ),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "eval_metric": trial.suggest_categorical("eval_metric", ["auc", "aucpr", "logloss", "map"]),
        "objective": "binary:logistic",
        "nthread": 8,
        "scale_pos_weight": data.label.value_counts()[0] / data.label.value_counts()[1],
    }
    number_cv_fold = len(data.cv_split.unique())
    fold_number = 0
    # separate training in 3 folds
    auc_scores = np.zeros(number_cv_fold)
    assert (data.cv_split != 'test').all(), "cv_split should be 'cv_i' with i an integer, not test"
    for i, validate_data in data.groupby("cv_split", observed=True):
        # split the data
        train_data = data[~data.cv_split.isin([i])]

        X_train = train_data[feature_name]
        y_train = train_data.label
        X_validate = validate_data[feature_name]
        y_validate = validate_data.label
        y_label_event = validate_data.label_id

        optuna_model = XGBClassifier(**params)
        optuna_model.fit(X_train, y_train)
        # Make predictions
        y_pred = optuna_model.predict_proba(X_validate)[:, 1]

        # Evaluate predictions
        precision, recall, thr_pr = precision_event_recall(y_validate.values, y_pred, y_label_event.values)
        auprc = auc(recall, precision)

        auc_scores[fold_number] = auprc
        fold_number += 1

    return auc_scores.mean()


def precision_event_recall(y_true: np.ndarray, y_pred: np.ndarray, label_id: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate precision, recall and thresholds for precision-recall curve.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted probabilities.
        label_id (np.ndarray): Label IDs.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the following arrays:
            - precision (np.ndarray): Precision values.
            - recall (np.ndarray): Recall values.
            - thresholds (np.ndarray): Thresholds for precision.
    """
    desc_score_indices = np.argsort(y_pred, kind="mergesort")[::-1]
    y_pred = y_pred[desc_score_indices]
    y_true = y_true[desc_score_indices]
    label_id = label_id[desc_score_indices]

    # Find distinct value indices
    distinct_value_indices = np.where(np.diff(y_pred))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # Compute true positive counts at each threshold
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = np.cumsum(1 - y_true)[threshold_idxs]
    ps = tps + fps
    # Compute precision at each threshold
    precision = np.divide(tps, ps, where=(ps != 0))

    # Reduce the number of thresholds by removing those that do not change precision
    # Compute changes in precision
    precision_changes = np.abs(np.diff(precision))
    # Extend changes array to match threshold_idxs size
    precision_changes = np.r_[precision_changes, precision_changes[-1]]

    # Define the number of thresholds to select
    num_selected_thresholds = 1000

    if len(precision_changes) > num_selected_thresholds:

        # Normalize changes to use as probabilities
        weights = precision_changes / np.sum(precision_changes)

        # Randomly select indices based on weights, ensuring some spread
        selected_threshold_idxs = np.random.choice(
            threshold_idxs, size=num_selected_thresholds, replace=False, p=weights)

        # Ensure the first and last thresholds are included
        selected_threshold_idxs = np.unique(np.r_[selected_threshold_idxs, threshold_idxs[0], threshold_idxs[-1]])

    else:
        selected_threshold_idxs = threshold_idxs

    # Recompute precision at selected thresholds
    precision_label = np.cumsum(y_true)[selected_threshold_idxs] / \
        np.cumsum(np.ones_like(y_true))[selected_threshold_idxs]

    # Compute recall at selected thresholds without explicit for loop
    nb_unique_label = len(np.unique(label_id[np.where(y_true == 1)[0]]))

    # Converting to a single-line computation using list comprehension
    recall_label = np.fromiter((len(np.unique(label_id[np.where(y_true[:threshold_idx + 1] == 1)[0]]))
                               for threshold_idx in selected_threshold_idxs), dtype=float) / nb_unique_label

    precision_label = np.hstack([1, precision_label])
    recall_label = np.hstack([0, recall_label])

    return precision_label, recall_label, y_pred[selected_threshold_idxs]


def get_all_stats(
    y_true: np.ndarray, y_pred: np.ndarray, label_id: np.ndarray, strategy: str = "max_precision", target: float = None
) -> tuple[
    float, float, float, float, float, float, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Calculate various statistics for binary classification.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted probabilities.
        label_id (np.ndarray): Label IDs.

    Returns:
        tuple: A tuple containing the following statistics:
            - auc_ (float): Area under the ROC curve.
            - sensitivity (float): Sensitivity.
            - specificity (float): Specificity.
            - ppv (float): Positive predictive value.
            - npv (float): Negative predictive value.
            - fpr (np.ndarray): False positive rates.
            - tpr (np.ndarray): True positive rates.
            - thr (np.ndarray): Thresholds.
            - precision (np.ndarray): Precision values.
            - recall (np.ndarray): Recall values.
            - thr_pr (np.ndarray): Thresholds for precision.
            - ap (float): Average precision score.
            - auprc (float): Area under the precision-recall curve.
            - precision_threshold (float): Precision threshold.
            - recall_threshold (float): Recall threshold.
    """
    precision, recall, thr_pr = precision_event_recall(y_true, y_pred, label_id)
    ap = average_precision_score(y_true, y_pred)

    fpr, tpr, thr = roc_curve(y_true, y_pred)
    auc_ = float(auc(fpr, tpr))
    auprc = auc(recall, precision)

    if strategy == 'max_precision':
        # find the threshold that optimize the precision after a recall of 0.1
        max_precision = np.max(precision[recall > 0.1])
        id_thresh_opt_prc = int(np.argmin(np.abs(precision - max_precision)))
    elif strategy == 'targeted_precision':
        id_thresh_opt_prc = int(np.argmin(np.abs(precision - target)))
    elif strategy == 'targeted_recall':
        id_thresh_opt_prc = int(np.argmin(np.abs(recall - target)))

    threshold_opt = thr_pr[id_thresh_opt_prc]

    precision_threshold = precision[id_thresh_opt_prc]
    recall_threshold = recall[id_thresh_opt_prc]

    nb_unique_label = len(np.unique(label_id[np.where(y_true == 1)[0]]))
    prevalence = nb_unique_label / (nb_unique_label + np.sum(1-y_true))

    term1 = prevalence / recall_threshold
    term2 = (1 - precision_threshold) / precision_threshold

    # Calculate specificity
    specificity = (1 - prevalence) / (term1 * term2 + (1 - prevalence))

    npv = (specificity * (1 - prevalence)) / (specificity * (1 - prevalence) + (1 - recall_threshold) * prevalence)

    f1 = 2 * (precision_threshold * recall_threshold) / (precision_threshold + recall_threshold)

    return auc_, specificity, npv, fpr, tpr, thr, precision, recall, thr_pr, ap, auprc, precision_threshold, recall_threshold, f1, threshold_opt


def bootstrap_worker(args):
    y_true, y_pred, y_label_id, xs_interpolation, recall_levels, seed, len_y_pred, strategy, target = args
    rng = np.random.default_rng(seed)
    while True:
        indices = rng.integers(0, len_y_pred, len_y_pred)
        if len(np.unique(y_true[indices])) < 2:
            continue

        (
            auc,
            specificity,
            npv,
            fpr,
            tpr,
            thr,
            precision,
            recall,
            thr_pr,
            ap,
            auprc,
            precision_threshold,
            recall_threshold,
            f1,
            threshold_opt,
        ) = get_all_stats(y_true[indices], y_pred[indices], y_label_id[indices], strategy, target)

        tpr_interpolated = np.interp(xs_interpolation, fpr, tpr)
        thr_interpolated = np.interp(xs_interpolation, fpr, np.clip(thr, -100, 100))
        precision_interpolated = np.interp(recall_levels, recall, precision)
        precision_interpolated[0] = 1
        thr_pr_interpolated = np.interp(recall_levels, recall[1:], thr_pr)

        return (
            auc,
            specificity,
            npv,
            ap,
            tpr_interpolated,
            thr_interpolated,
            precision_interpolated,
            thr_pr_interpolated,
            auprc,
            precision_threshold,
            recall_threshold,
            f1,
            threshold_opt,
        )


def bootstrap_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_label_id: np.ndarray,
    n_bootstraps: int = 200,
    rng_seed: int = 42,
    strategy: str = "max_precision",
    target: float = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Perform bootstrap testing for evaluating a regression model's performance.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        y_label_id (np.ndarray): Label IDs.
        n_bootstraps (int, optional): Number of bootstrap iterations. Defaults to 200.
        rng_seed (int, optional): Seed for the random number generator. Defaults to 42.

    Returns:
        tuple[pd.DataFrame, np.ndarray]: A tuple containing the DataFrame with evaluation metrics and the interpolated TPR values.
    """
    np.random.seed(rng_seed)
    seeds = np.random.randint(0, np.iinfo(np.int32).max, size=n_bootstraps)

    # for each subgoup of data, create a regressor and evaluate it
    n_bootstraps = 200
    xs_interpolation = np.linspace(0, 1, N_INTERPOLATION)
    recall_levels = np.linspace(0, 1, N_INTERPOLATION)

    results = []
    args = [(y_true, y_pred, y_label_id, xs_interpolation, recall_levels, seeds[i], len(y_pred), strategy, target)
            for i in range(n_bootstraps)]

    with mp.Pool() as pool:
        for result in tqdm(pool.imap(bootstrap_worker, args), total=n_bootstraps):
            results.append(result)

    # Initialize arrays to store the results
    aucs = np.zeros(n_bootstraps, dtype=float)
    tprs_interpolated = np.zeros((n_bootstraps, N_INTERPOLATION), dtype=float)
    thrs_interpolated = np.zeros((n_bootstraps, N_INTERPOLATION), dtype=float)
    precision_interpolated = np.zeros((n_bootstraps, N_INTERPOLATION), dtype=float)
    thrs_interpolated_precision = np.zeros((n_bootstraps, N_INTERPOLATION), dtype=float)

    specificities = np.zeros(n_bootstraps, dtype=float)
    npvs = np.zeros(n_bootstraps, dtype=float)
    aps = np.zeros(n_bootstraps, dtype=float)
    auprcs = np.zeros(n_bootstraps, dtype=float)
    precision_thresholds = np.zeros(n_bootstraps, dtype=float)
    recall_thresholds = np.zeros(n_bootstraps, dtype=float)
    f1s = np.zeros(n_bootstraps, dtype=float)
    threshold_opts = np.zeros(n_bootstraps, dtype=float)

    for i, result in enumerate(results):
        (
            auc,
            specificity,
            npv,
            ap,
            tpr_interpolated,
            thr_interpolated,
            precision_interpolated_,
            thr_pr_interpolated,
            auprc,
            precision_threshold,
            recall_threshold,
            f1,
            threshold_opt,
        ) = result

        aucs[i] = auc
        specificities[i] = specificity
        npvs[i] = npv
        aps[i] = ap
        auprcs[i] = auprc
        precision_thresholds[i] = precision_threshold
        recall_thresholds[i] = recall_threshold
        f1s[i] = f1
        threshold_opts[i] = threshold_opt

        tprs_interpolated[i] = tpr_interpolated
        thrs_interpolated[i] = thr_interpolated
        precision_interpolated[i] = precision_interpolated_
        thrs_interpolated_precision[i] = thr_pr_interpolated

    tpr_mean, tpr_std = tprs_interpolated.mean(0), tprs_interpolated.std(0)
    precision_mean, precision_std = precision_interpolated.mean(0), precision_interpolated.std(0)
    thr_recall_mean = thrs_interpolated.mean(0)
    thr_mean = thrs_interpolated.mean(0)
    npv_mean, npv_std = npvs.mean(), npvs.std()
    auc_mean, auc_std = aucs.mean(), aucs.std()
    ap_mean, ap_std = aps.mean(), aps.std()
    specificity_mean, specificity_std = specificities.mean(), specificities.std()
    auprc_mean, auprc_std = auprcs.mean(), auprcs.std()
    precision_threshold_mean, precision_threshold_std = precision_thresholds.mean(), precision_thresholds.std()
    recall_threshold_mean, recall_threshold_std = recall_thresholds.mean(), recall_thresholds.std()
    f1_mean, f1_std = f1s.mean(), f1s.std()
    threshold_opt_mean, threshold_opt_std = threshold_opts.mean(), threshold_opts.std()

    df = pd.DataFrame(
        {
            "fpr": xs_interpolation,
            "tpr": tpr_mean,
            "tpr_std": tpr_std,
            "threshold": thr_mean,
            "auc": auc_mean,
            "auc_std": auc_std,
            "specificity": specificity_mean,
            "specificity_std": specificity_std,
            "npv": npv_mean,
            "npv_std": npv_std,
            "recall": recall_levels,
            "precision": precision_mean,
            "precision_std": precision_std,
            "thr_precision": thr_recall_mean,
            "ap": ap_mean,
            "ap_std": ap_std,
            "auprc": auprc_mean,
            "auprc_std": auprc_std,
            "precision_threshold": precision_threshold_mean,
            "precision_threshold_std": precision_threshold_std,
            "recall_threshold": recall_threshold_mean,
            "recall_threshold_std": recall_threshold_std,
            "f1": f1_mean,
            "f1_std": f1_std,
            "threshold_opt": threshold_opt_mean,
            "threshold_opt_std": threshold_opt_std,
        }
    )
    return df, tprs_interpolated, precision_interpolated


def load_labelized_cases(
    dataset_path: Path,
    caseid: int,
):
    """
    Labelize the IOH event of a single case.
    Args:
        databuilder (Path): Path of the dataset.
        caseid (int): Case ID.
    Returns:
        pd.DataFrame: Labeled case data.
    """
    databuilder = DataBuilder.from_json(dataset_path)
    databuilder.raw_data_folder = databuilder.raw_data_folder / f"case-{caseid:04d}.parquet"
    case_data, _ = databuilder._import_raw()
    case_data = case_data.reset_index("caseid", drop=True)
    preprocess_case = databuilder._preprocess(case_data)
    label, _ = databuilder._labelize(preprocess_case)
    preprocess_case["label"] = label

    return preprocess_case


def print_statistics(df: pd.DataFrame) -> None:
    """
    Print the evaluation statistics of the model.
    Args:
        df (pd.DataFrame): DataFrame containing the evaluation metrics.
    """
    print(f"AUC: {df.auc.iloc[-1]:.1%} ± {df.auc_std.iloc[-1]:.1%}")

    print(f"AP: {df.ap.iloc[-1]:.1%} ± {df.ap_std.iloc[-1]:.1%}")
    print(f"AUPRC: {df.auprc.iloc[-1]:.1%} ± {df.auprc_std.iloc[-1]:.1%}")
    print(f"Threshold: {df.threshold_opt.iloc[-1]:.2f} ± {df.threshold_opt_std.iloc[-1]:.2f}")
    print(f"Recall: {df.recall_threshold.iloc[-1]:.1%} ± {df.recall_threshold_std.iloc[-1]:.1%}")
    print(f"Precision: {df.precision_threshold.iloc[-1]:.1%} ± {df.precision_threshold_std.iloc[-1]:.1%}")
    print(f"Specificity: {df.specificity.iloc[-1]:.1%} ± {df.specificity_std.iloc[-1]:.1%}")
    print(f"NPV: {df.npv.iloc[-1]:.1%} ± {df.npv_std.iloc[-1]:.1%}")
    print(f"F1-score: {df.f1.iloc[-1]:.2f} ± {df.f1_std.iloc[-1]:.2f}")
    return
