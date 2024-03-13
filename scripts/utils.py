import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.metrics import auc, roc_curve

NUMBER_CV_FOLD = 3


def objective_xgboost(trial, data, feature_name):
    params = {
        'max_depth': trial.suggest_int('max_depth', 1, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'eval_metric': 'auc',
        'objective': 'binary:logistic',
        'nthread': 8,
        'scale_pos_weight': 15,
    }
    # separate training in 3 folds
    caseid_list = np.array_split(data.caseid.unique(), NUMBER_CV_FOLD)
    auc_list = []
    for i in range(NUMBER_CV_FOLD):
        # split the data
        test_caseid = caseid_list[i]
        train_caseid = np.concatenate([caseid_list[j] for j in range(NUMBER_CV_FOLD) if j != i])
        df_train = data[data.caseid.isin(train_caseid)]
        df_test = data[data.caseid.isin(test_caseid)]

        X_train = df_train[feature_name]
        y_train = df_train.label
        X_test = df_test[feature_name]
        y_test = df_test.label

        optuna_model = XGBClassifier(**params)
        optuna_model.fit(X_train, y_train)

        # Make predictions
        y_pred = optuna_model.predict_proba(X_test)[:, 1]

        # Evaluate predictions
        accuracy = roc_auc_score(y_test, y_pred)
        auc_list.append(accuracy)

    return np.mean(auc_list)


def stats_for_one_threshold(y_true, y_pred, threshold, label_id):
    y_pred = (y_pred > threshold).astype(int)
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'label_id': label_id})

    true_positive = 0
    false_negative = 0
    for label, df_label in df.groupby('label_id'):
        if np.isnan(label):
            continue

        true_positive += df_label.y_pred.max()
        false_negative += 1 - df_label.y_pred.max()

    sensitivity = true_positive / (true_positive + false_negative)
    return sensitivity


def get_all_stats(y_true, y_pred, label_id):

    fpr, tpr, thr = roc_curve(y_true, y_pred)
    auc_ = auc(fpr, tpr)
    gmean = np.sqrt(tpr * (1 - fpr))
    id_thresh_opt = np.argmax(gmean)
    sensitivity = tpr[id_thresh_opt] * 100
    specificity = (1 - fpr[id_thresh_opt]) * 100
    prevalence = np.mean(y_true)
    ppv = 100 * tpr[id_thresh_opt] * prevalence / \
        (tpr[id_thresh_opt] * prevalence + fpr[id_thresh_opt] * (1 - prevalence))
    npv = 100 * (1 - fpr[id_thresh_opt]) * (1 - prevalence) / ((1 - tpr[id_thresh_opt])
                                                               * prevalence + (1 - fpr[id_thresh_opt]) * (1 - prevalence))

    sensitivity_ioh = stats_for_one_threshold(y_true, y_pred, thr[id_thresh_opt], label_id)

    return auc_, sensitivity, specificity, ppv, npv, sensitivity_ioh, fpr, tpr, thr


def bootstrap_test(y_true, y_pred, y_label_id, n_bootstraps=200, rng_seed=42):

    rng = np.random.RandomState(rng_seed)
    fpr = np.linspace(0, 1, 100)

    tpr_list, auc_list, thr_list = [], [], []
    sensitivity_list, specificity_list, ppv_list, npv_list, sensi_ioh = [], [], [], [], []

    # for each subgoup of data, create a regressor and evaluate it
    n_bootstraps = 200
    bootstrapped_scores = []

    i_bootstrap = 0
    while i_bootstrap < n_bootstraps:
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        i_bootstrap += 1

        auc_, sensitivity_, specificity_, ppv_, npv_, sensitivity_ioh_, fpr_, tpr_, thr_ = get_all_stats(
            y_true[indices], y_pred[indices], y_label_id[indices])
        sensitivity_list.append(sensitivity_ * 100)
        specificity_list.append(specificity_ * 100)
        ppv_list.append(ppv_)
        npv_list.append(npv_)

        tpr_list.append(np.interp(np.linspace(0, 1, 100), fpr_, tpr_))
        thr_list.append(np.interp(np.linspace(0, 1, 100), fpr_, thr_))
        auc_list.append(auc_)
        sensi_ioh.append(sensitivity_ioh_)

    tpr_mean = np.mean(tpr_list, axis=0)
    tpr_std = np.std(tpr_list, axis=0)
    thr_mean = np.mean(thr_list, axis=0)
    sensi_mean = np.mean(sensitivity_list)
    speci_mean = np.mean(specificity_list)
    ppv_mean = np.mean(ppv_list)
    npv_mean = np.mean(npv_list)
    sensi_std = np.std(sensitivity_list)
    speci_std = np.std(specificity_list)
    ppv_std = np.std(ppv_list)
    npv_std = np.std(npv_list)
    auc_mean = np.mean(auc_list)
    auc_std = np.std(auc_list)
    sensi_ioh_mean = np.mean(sensi_ioh)
    sensi_ioh_std = np.std(sensi_ioh)

    df = pd.DataFrame(
        {
            "fpr": fpr,
            "tpr": tpr_mean,
            "tpr_std": tpr_std,
            "threshold": thr_mean,
            "auc": auc_mean,
            "auc_std": auc_std,
            "sensitivity": sensi_mean,
            "sensitivity_std": sensi_std,
            "specificity": speci_mean,
            "specificity_std": speci_std,
            "ppv": ppv_mean,
            "ppv_std": ppv_std,
            "npv": npv_mean,
            "npv_std": npv_std,
            "sensitivity_ioh": sensi_ioh_mean,
            "sensitivity_ioh_std": sensi_ioh_std,
        }
    )
    return df, tpr_list
