from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.metrics import auc, roc_curve
from hp_pred.databuilder import build_databuilder
from optuna import Trial

NUMBER_CV_FOLD = 3
N_INTERPOLATION = 100


def objective_xgboost(
    trial: Trial,
    data: pd.DataFrame,
    feature_name: list[str],
    cv_split: list[np.ndarray],
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
        "eval_metric": "auc",
        "objective": "binary:logistic",
        "nthread": 8,
        "scale_pos_weight": 15,
    }
    number_cv_fold = data.cv_split.unique().shape[0]
    # separate training in 3 folds
    auc_scores = np.zeros(number_cv_fold)
    for i, validate_data in data.groupby("cv_split"):
        # split the data
        train_data = data[~data.cv_split.isin([i])]

        X_train = train_data[feature_name]
        y_train = train_data.label
        X_validate = validate_data[feature_name]
        y_validate = validate_data.label

        optuna_model = XGBClassifier(**params)
        optuna_model.fit(X_train, y_train)

        # Make predictions
        y_pred = optuna_model.predict_proba(X_validate)[:, 1]

        # Evaluate predictions
        accuracy = roc_auc_score(y_validate, y_pred)
        auc_scores[i] = accuracy

    return auc_scores.mean()


def stats_for_one_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    label_id: np.ndarray,
) -> float:
    """
    Calculate sensitivity for a given threshold.

    Args:
        y_true (np.ndarray): Array of true labels.
        y_pred (np.ndarray): Array of predicted probabilities.
        threshold (float): Threshold value for classification.
        label_id (np.ndarray): Array of label IDs.

    Returns:
        float: Sensitivity value.
    """
    y_pred_thresholded = (y_pred > threshold).astype(int)
    df = pd.DataFrame(
        {"y_true": y_true, "y_pred": y_pred_thresholded, "label_id": label_id}
    )

    true_positive = 0
    false_negative = 0
    for label, df_label in df.groupby("label_id"):
        if pd.isna(label):
            continue

        true_positive += df_label.y_pred.max()
        false_negative += 1 - df_label.y_pred.max()

    sensitivity = true_positive / (true_positive + false_negative)
    return sensitivity


def get_all_stats(
    y_true: np.ndarray, y_pred: np.ndarray, label_id: np.ndarray
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
            - sensitivity_ioh (float): Sensitivity for one threshold.
            - fpr (np.ndarray): False positive rates.
            - tpr (np.ndarray): True positive rates.
            - thr (np.ndarray): Thresholds.
    """
    fpr, tpr, thr = roc_curve(y_true, y_pred)
    auc_ = float(auc(fpr, tpr))
    gmean = np.sqrt(tpr * (1 - fpr))
    id_thresh_opt = int(gmean.argmax())

    sensitivity: float = tpr[id_thresh_opt] * 100
    specificity: float = (1 - fpr[id_thresh_opt]) * 100

    prevalence = y_true.mean()
    ppv: float = (
        100
        * tpr[id_thresh_opt]
        * prevalence
        / (tpr[id_thresh_opt] * prevalence + fpr[id_thresh_opt] * (1 - prevalence))
    )
    npv: float = (
        100
        * (1 - fpr[id_thresh_opt])
        * (1 - prevalence)
        / (
            (1 - tpr[id_thresh_opt]) * prevalence
            + (1 - fpr[id_thresh_opt]) * (1 - prevalence)
        )
    )

    sensitivity_ioh = stats_for_one_threshold(
        y_true, y_pred, thr[id_thresh_opt], label_id
    )

    return auc_, sensitivity, specificity, ppv, npv, sensitivity_ioh, fpr, tpr, thr


def bootstrap_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_label_id: np.ndarray,
    n_bootstraps: int = 200,
    rng_seed: int = 42,
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

    rng = np.random.RandomState(rng_seed)

    # for each subgoup of data, create a regressor and evaluate it
    n_bootstraps = 200

    aucs = np.zeros(n_bootstraps, dtype=float)
    tprs_interpolated = np.zeros((n_bootstraps, N_INTERPOLATION), dtype=float)
    thrs_interpolated = np.zeros((n_bootstraps, N_INTERPOLATION), dtype=float)
    xs_interpolation = np.linspace(0, 1, N_INTERPOLATION)

    sensitivities = np.zeros(n_bootstraps, dtype=float)
    specificities = np.zeros(n_bootstraps, dtype=float)
    ioh_sensitivites = np.zeros(n_bootstraps, dtype=float)

    ppvs = np.zeros(n_bootstraps, dtype=float)
    npvs = np.zeros(n_bootstraps, dtype=float)

    i_bootstrap = 0
    while i_bootstrap < n_bootstraps:
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        (
            auc,
            sensitivity,
            specificity,
            ppv,
            npv,
            ioh_sensitivity,
            fpr,
            tpr,
            thr,
        ) = get_all_stats(y_true[indices], y_pred[indices], y_label_id[indices])

        aucs[i_bootstrap] = auc
        sensitivities[i_bootstrap] = sensitivity
        specificities[i_bootstrap] = specificity
        ppvs[i_bootstrap] = ppv
        npvs[i_bootstrap] = npv
        ioh_sensitivites[i_bootstrap] = ioh_sensitivity

        tprs_interpolated[i_bootstrap] = np.interp(xs_interpolation, fpr, tpr)
        thrs_interpolated[i_bootstrap] = np.interp(xs_interpolation, fpr, np.clip(thr, -100, 100))

        i_bootstrap += 1

    tpr_mean, tpr_std = tprs_interpolated.mean(0), tprs_interpolated.std(0)
    thr_mean = thrs_interpolated.mean(0)
    ppv_mean, ppv_std = ppvs.mean(), ppvs.std()
    npv_mean, npv_std = npvs.mean(), npvs.std()
    auc_mean, auc_std = aucs.mean(), aucs.std()
    sensitivity_mean, sensitivity_std = sensitivities.mean(), sensitivities.std()
    specificity_mean, specificity_std = specificities.mean(), specificities.std()
    ioh_sensitivity_mean, ioh_sensitivity_std = (
        ioh_sensitivites.mean(),
        ioh_sensitivites.std(),
    )

    df = pd.DataFrame(
        {
            "fpr": xs_interpolation,
            "tpr": tpr_mean,
            "tpr_std": tpr_std,
            "threshold": thr_mean,
            "auc": auc_mean,
            "auc_std": auc_std,
            "sensitivity": sensitivity_mean,
            "sensitivity_std": sensitivity_std,
            "specificity": specificity_mean,
            "specificity_std": specificity_std,
            "ioh_sensitivity": ioh_sensitivity_mean,
            "ioh_sensitivity_std": ioh_sensitivity_std,
            "ppv": ppv_mean,
            "ppv_std": ppv_std,
            "npv": npv_mean,
            "npv_std": npv_std,
        }
    )
    return df, tprs_interpolated


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
    databuilder = build_databuilder(dataset_path)
    databuilder.raw_data_folder = databuilder.raw_data_folder / f"case-{caseid:04d}.parquet"
    case_data, _ = databuilder._import_raw()
    case_data = case_data.reset_index("caseid", drop=True)
    preprocess_case = databuilder._preprocess(case_data)
    label, _ = databuilder._labelize(preprocess_case)
    preprocess_case["label"] = label

    return preprocess_case
