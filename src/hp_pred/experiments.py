from pathlib import Path
import multiprocessing as mp
import os
from typing import Dict, List

import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import auc, roc_curve, average_precision_score, root_mean_squared_error, precision_recall_curve
from hp_pred.databuilder import DataBuilder
from optuna import Trial
from tqdm import tqdm
from scipy.stats import shapiro

N_INTERPOLATION = 1000


def objective_xgboost_regression(
    trial: Trial,
    data_train: list[pd.DataFrame],
    data_test: list[pd.DataFrame],
    feature_name: List[str],
) -> float:
    """
    Calculate the mean AUC score for XGBoost model using Optuna hyperparameter optimization.

    Args:
        trial (Trial): Optuna trial object for hyperparameter optimization.
        data_train (list[pd.DataFrame]): List of training data sets.
        data_test (list[pd.DataFrame]): List of testing data sets corresponding to the training data sets.
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
        "eval_metric": "rmse",
        "objective": "reg:squarederror",
        "nthread": os.cpu_count(),
        'multi_strategy': 'multi_output_tree',
        # "scale_pos_weight": data.label.value_counts()[0] / data.label.value_counts()[1],
    }
    number_cv_fold = len(data_train)
    assert number_cv_fold == len(data_test), "The number of training and testing data set should be the same."

    labels_name = [col for col in data_train[0].columns if "future" in col]

    params['num_target'] = len(labels_name)

    # separate training in 3 folds
    rmse_scores = np.zeros(number_cv_fold)
    for i in range(number_cv_fold):

        X_train = data_train[i][feature_name]
        y_train = data_train[i][labels_name]

        X_validate = data_test[i][feature_name]
        y_validate = data_test[i][labels_name]

        optuna_model = XGBRegressor(**params)
        optuna_model.fit(X_train, y_train, eval_set=[(X_validate, y_validate)], verbose=0)
        # Make predictions
        y_pred = optuna_model.predict(X_validate)

        # Evaluate predictions with RMSE score
        rmse_scores[i] = root_mean_squared_error(y_validate, y_pred)

    return rmse_scores.mean()


def objective_xgboost(
    trial: Trial,
    data_train: list[pd.DataFrame],
    data_test: list[pd.DataFrame],
    feature_name: List[str],
) -> float:
    """
    Calculate the mean AUC score for XGBoost model using Optuna hyperparameter optimization.

    Args:
        trial (Trial): Optuna trial object for hyperparameter optimization.
        data_train (list[pd.DataFrame]): List of training data sets.
        data_test (list[pd.DataFrame]): List of testing data sets corresponding to the training data sets.
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
        "nthread": os.cpu_count(),
        # "scale_pos_weight": data.label.value_counts()[0] / data.label.value_counts()[1],
    }
    number_cv_fold = len(data_train)
    assert number_cv_fold == len(data_test), "The number of training and testing data set should be the same."

    fold_number = 0
    # separate training in 3 folds
    ap_scores = np.zeros(number_cv_fold)
    for i in range(number_cv_fold):

        X_train = data_train[i][feature_name]
        y_train = data_train[i].label

        X_validate = data_test[i][feature_name]
        y_validate = data_test[i].label

        optuna_model = XGBClassifier(**params)
        optuna_model.fit(X_train, y_train)
        # Make predictions
        y_pred = optuna_model.predict_proba(X_validate)[:, 1]

        # Evaluate predictions with AP score
        ap_scores[fold_number] = average_precision_score(y_validate, y_pred)
        fold_number += 1

    return ap_scores.mean()


def operationnal_precision_recall(y_true: np.ndarray, y_pred: np.ndarray, label_id: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate precision, recall, and thresholds for precision-recall curve,
    considering groups of successive positive predictions as single events.

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
    # Compute distinct threshold indices
    thresholds = np.unique(y_pred)  # Unique thresholds sorted automatically

    # get a mximized number of thresholds
    thresholds = np.linspace(np.min(thresholds), np.max(thresholds), np.min([100, len(thresholds)]))

    precisions = []
    recalls = []

    nb_unique_label = len(np.unique(label_id[np.where(y_true == 1)[0]]))

    for threshold in thresholds:
        # Identify events dynamically for the current threshold
        above_threshold = (y_pred >= threshold).astype(int)

        alert_id = np.where(np.diff(above_threshold) == 1)[0]+1
        if above_threshold[0]:
            alert_id = np.hstack([0, alert_id])

        # Include events where label_id changes but threshold remains above
        label_change_idx = np.where(label_id[1:] != label_id[:-1])[0] + 1
        alert_id = np.unique(np.concatenate((alert_id, label_change_idx[above_threshold[label_change_idx] == 1])))

        true_alert = np.sum(y_true[alert_id])
        false_alert = len(alert_id) - true_alert

        precisions.append(true_alert / (true_alert + false_alert) if true_alert + false_alert > 0 else 0)
        # number of unique labels in the alert
        alert_label = label_id[alert_id]
        counted_labels = np.unique(alert_label[np.where(y_true[alert_id] == 1)[0]])

        recalls.append(len(counted_labels) / nb_unique_label if nb_unique_label > 0 else 0)

    # Append initial values for PR curve
    precisions = np.hstack([1, precisions])
    recalls = np.hstack([0, recalls])

    # sort the values by recall
    idx = np.argsort(recalls)
    idx_thres = np.argsort(recalls[1:])
    precisions = precisions[idx]
    recalls = recalls[idx]
    thresholds = thresholds[idx_thres]

    return precisions, recalls, thresholds


def precision_event_recall_old(y_true: np.ndarray, y_pred: np.ndarray, label_id: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    op_precision, op_recall, thr_ope = operationnal_precision_recall(y_true, y_pred, label_id)
    ap = average_precision_score(y_true, y_pred)
    precision, recall, thr_std = precision_recall_curve(y_true, y_pred)

    fpr, tpr, threshold_auc = roc_curve(y_true, y_pred)
    auc_ = float(auc(fpr, tpr))
    au_op_prc = auc(op_recall, op_precision)
    auprc = auc(recall, precision)

    if strategy == 'max_precision':
        # find the threshold that optimize the precision after a recall of 0.1
        max_precision = np.max(op_precision[op_recall > 0.02])
        id_thresh_opt_prc = int(np.argmin(np.abs(op_precision - max_precision)))
    elif strategy == 'targeted_precision':
        id_thresh_opt_prc = int(np.argmin(np.abs(op_precision - target)))
    elif strategy == 'targeted_recall':
        id_thresh_opt_prc = int(np.argmin(np.abs(op_recall - target)))
    elif strategy == 'fixed_threshold':
        id_thresh_opt_prc = int(np.argmin(np.abs(thr_ope - target)))

    threshold_opt = thr_ope[id_thresh_opt_prc]

    op_precision_threshold = op_precision[id_thresh_opt_prc]
    op_recall_threshold = op_recall[id_thresh_opt_prc]

    nb_unique_label = len(np.unique(label_id[np.where(y_true == 1)[0]]))
    prevalence = nb_unique_label / (nb_unique_label + np.sum(1-y_true))

    term1 = prevalence / op_recall_threshold
    term2 = (1 - op_precision_threshold) / op_precision_threshold

    # Calculate specificity
    specificity = (1 - prevalence) / (term1 * term2 + (1 - prevalence))

    npv = (specificity * (1 - prevalence)) / (specificity * (1 - prevalence) + (1 - op_recall_threshold) * prevalence)

    f1 = 2 * (op_precision_threshold * op_recall_threshold) / (op_precision_threshold + op_recall_threshold)

    return auc_, specificity, npv, fpr, tpr, threshold_auc, op_precision, op_recall, thr_ope, ap, au_op_prc, op_precision_threshold, op_recall_threshold, f1, threshold_opt, precision, recall, thr_std, auprc


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
            threshold_auc,
            op_precision,
            op_recall,
            thr_ope,
            ap,
            au_op_prc,
            precision_threshold,
            recall_threshold,
            f1,
            threshold_opt,
            precision,
            recall,
            thr_std,
            auprc,

        ) = get_all_stats(y_true[indices], y_pred[indices], y_label_id[indices], strategy, target)

        tpr_interpolated = np.interp(xs_interpolation, fpr, tpr)
        threshold_auc = np.interp(xs_interpolation, fpr, np.clip(threshold_auc, -100, 100))
        op_precision_interpolated = np.interp(recall_levels, op_recall, op_precision)
        op_precision_interpolated[0] = 1
        threshold_precision_opertional = np.interp(recall_levels, op_recall[1:], thr_ope)
        precision_interpolated = np.interp(recall_levels, recall[::-1], precision[::-1])
        threshold_precision_standard = np.interp(recall_levels, recall[-2::-1], thr_std[::-1])

        return (
            auc,
            specificity,
            npv,
            ap,
            tpr_interpolated,
            threshold_auc,
            op_precision_interpolated,
            threshold_precision_opertional,
            au_op_prc,
            precision_threshold,
            recall_threshold,
            f1,
            threshold_opt,
            precision_interpolated,
            threshold_precision_standard,
            auprc,
        )


def bootstrap_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_label_id: np.ndarray,
    n_bootstraps: int = 200,
    rng_seed: int = 42,
    strategy: str = "max_precision",
    target: float = None,
) -> tuple[Dict[str, pd.DataFrame], np.ndarray]:
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
    auprcs = np.zeros(n_bootstraps, dtype=float)
    tprs_interpolated = np.zeros((n_bootstraps, N_INTERPOLATION), dtype=float)
    op_precision_interpolated = np.zeros((n_bootstraps, N_INTERPOLATION), dtype=float)
    precision_interpolated = np.zeros((n_bootstraps, N_INTERPOLATION), dtype=float)
    threshold_auc = np.zeros((n_bootstraps, N_INTERPOLATION), dtype=float)
    threshold_operational = np.zeros((n_bootstraps, N_INTERPOLATION), dtype=float)
    threshold_standard_prc = np.zeros((n_bootstraps, N_INTERPOLATION), dtype=float)

    specificities = np.zeros(n_bootstraps, dtype=float)
    npvs = np.zeros(n_bootstraps, dtype=float)
    aps = np.zeros(n_bootstraps, dtype=float)
    au_op_prcs = np.zeros(n_bootstraps, dtype=float)
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
            threshold_auc_,
            op_precision_interpolated_,
            threshold_precision_opertional,
            au_op_prc,
            precision_threshold,
            recall_threshold,
            f1,
            threshold_opt,
            precision_interpolated_,
            threshold_precision_standard,
            auprc,
        ) = result

        aucs[i] = auc
        specificities[i] = specificity
        npvs[i] = npv
        aps[i] = ap
        auprcs[i] = auprc
        au_op_prcs[i] = au_op_prc
        precision_thresholds[i] = precision_threshold
        recall_thresholds[i] = recall_threshold
        f1s[i] = f1
        threshold_opts[i] = threshold_opt

        tprs_interpolated[i] = tpr_interpolated
        precision_interpolated[i] = precision_interpolated_
        op_precision_interpolated[i] = op_precision_interpolated_
        threshold_auc[i] = threshold_auc_
        threshold_operational[i] = threshold_precision_opertional
        threshold_standard_prc[i] = threshold_precision_standard

    dict = {
        "fprs": xs_interpolation,
        "tpr": tprs_interpolated,
        "threshold_auc": threshold_auc,
        "aucs": aucs,
        "specificity": specificities,
        "npvs": npvs,
        "op_recall": recall_levels,
        "precision": precision_interpolated,
        "threshold_precision_standard": threshold_standard_prc,
        "op_precision": op_precision_interpolated,
        "op_threshold": threshold_operational,
        "aps": aps,
        "auprcs": auprcs,
        "au_op_prcs": au_op_prcs,
        "op_precision_threshold": precision_thresholds,
        "recall_threshold": recall_thresholds,
        "f1": f1s,
        "threshold_opt": threshold_opts,
    }
    return dict, precision_interpolated, threshold_operational


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
    if 'chu' in str(databuilder.raw_data_folder):
        databuilder.raw_data_folder = databuilder.raw_data_folder / f"chu_case_{caseid:03d}.parquet"
    elif 'feature' in str(databuilder.raw_data_folder):
        databuilder.raw_data_folder = databuilder.raw_data_folder / f"case_{caseid:04d}.parquet"
    else:
        databuilder.raw_data_folder = databuilder.raw_data_folder / f"case-{caseid:04d}.parquet"

    case_data, _ = databuilder._import_raw()
    case_data = case_data.reset_index("caseid", drop=True)
    preprocess_case = databuilder._preprocess(case_data)
    label, _ = databuilder._labelize(preprocess_case)
    preprocess_case["label"] = label
    mbp_column = databuilder.mbp_column

    return preprocess_case, mbp_column


def print_one_stat(series: pd.Series, percent: bool = False) -> bool:
    """ Test if a series is normally distributed using the Shapiro-Wilk test.

    If it is return mean (sd), otherwise return median (Q1, Q3).
    I percent is True, the values are returned as percentages.

    Args:
        series (pd.Series): Series to test.

    Returns:
        bool: True if the series is normally distributed, False otherwise.
    """
    stat, p = shapiro(series)
    if p > 0.05:
        if percent:
            return f"{series.mean():.1%} ({series.std():.1%})"
        else:
            return f"{series.mean():.2f} ({series.std():.2f})"
    else:
        if percent:
            return f"{series.median():.1%} [{series.quantile(0.25):.1%}, {series.quantile(0.75):.1%}]"
        else:
            return f"{series.median():.2f} [{series.quantile(0.25):.2f}, {series.quantile(0.75):.2f}]"


def print_statistics(dict: Dict[str, pd.DataFrame]) -> None:
    """
    Print the evaluation statistics of the model.
    Args:
        df (pd.DataFrame): DataFrame containing the evaluation metrics.
    """
    df = pd.DataFrame()
    for key in ["aucs", "aps", "auprcs", "threshold_opt", "recall_threshold", "specificity", "npvs", "f1", "au_op_prcs", "op_precision_threshold"]:
        df[key] = dict[key]
    print('----- General stats -----')
    print(f"AUCROC: {print_one_stat(df.aucs, False)}")
    print(f"AUPRC: {print_one_stat(df.auprcs, False)}")
    print(f"Operational AUPRC: {print_one_stat(df.au_op_prcs, False)}")

    print('----- At threshold stats -----')
    print(f"Threshold: {print_one_stat(df.threshold_opt, False)}")
    print(f"Operational Recall: {print_one_stat(df.recall_threshold, True)}")
    print(f"Operational Precision: {print_one_stat(df.op_precision_threshold, True)}")
    print(f"Operational Specificity: {print_one_stat(df.specificity, True)}")
    print(f"Operational NPV: {print_one_stat(df.npvs, True)}")
    print(f"Operational F1-score: {print_one_stat(df.f1, True)}")
    return
