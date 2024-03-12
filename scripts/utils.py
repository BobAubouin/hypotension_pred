import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

NUMBER_CV_FOLD = 3


def objective(trial, data, feature_name):
    params = {
        'max_depth': trial.suggest_int('max_depth', 1, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0, log=True),
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
        if np.isna(label):
            continue

        true_positive += df_label.y_pred.max()
        false_negative += 1 - df_label.y_pred.max()

    sensitivity = true_positive / (true_positive + false_negative)
    return sensitivity
