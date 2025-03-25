from pathlib import Path
import pickle
from itertools import chain, repeat

import pandas as pd
import numpy as np
import xgboost as xgb
from aeon.classification.deep_learning import IndividualLITEClassifier
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import shap
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

import hp_pred.experiments as expe

BASELINE_FEATURE = "last_map_value"
BASIC_COLORS = list(mcolors.TABLEAU_COLORS.values())


def _revert_dict(d):
    return dict(chain(*[zip(val, repeat(key)) for key, val in d.items()]))


def _grouped_shap(shap_vals, features, groups):
    groupmap = _revert_dict(groups)
    shap_Tdf = pd.DataFrame(shap_vals, columns=pd.Index(features, name='features')).T
    shap_Tdf['group'] = shap_Tdf.reset_index().features.map(groupmap).values
    shap_grouped = shap_Tdf.groupby('group').sum().T
    return shap_grouped


def expected_calibration_error(confidences, true_labels, M=5):
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # get predictions from confidences (positional in this case)
    predicted_label = confidences > 0.5

    confidences = np.abs(confidences - 0.5) + 0.5  # get the confidence of the prediction

    # get a boolean list of correct/false predictions
    accuracies = predicted_label == true_labels

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower &amp; upper)
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return ece


def reformat_time_data(data, signals_names):
    def get_signal_time(signal):
        try:
            return int(signal.split('_')[2])
        # except list index out of range
        except:
            try:
                return int(signal.split('_')[3])
            except:
                return 0

    grouped_columns = {
        signal: sorted([col for col in data.columns if col.startswith(signal)], key=get_signal_time)
        for signal in signals_names
    }

    X = np.stack([data[cols].values for cols in grouped_columns.values()], axis=1)
    return X


class TestModel():
    def __init__(
            self,
            test_data: pd.DataFrame,
            train_data: pd.DataFrame,
            model_filenames: list[str],
            output_name: str,
            plot_name: list[str] = None,
            n_bootstraps: int = 200,
    ):

        self.test_data = test_data
        self.train_data = train_data

        # test if the model path is valid
        self.model = []
        for filename in model_filenames:
            if filename.endswith(".keras"):
                lite_list = []
                for i in range(3):
                    lite = IndividualLITEClassifier()
                    lite.load_model(model_path=Path("data/models") / (filename[:-6]+str(i)+'.keras'),
                                    classes=[0, 1],
                                    )
                    lite_list.append(lite)
                model = VotingClassifier(lite_list, voting='soft')
                self.model += [model]
            elif filename.endswith(".json"):
                self.model += [xgb.XGBClassifier()]
                model_path = Path("data/models") / filename
                try:
                    self.model[-1].load_model(model_path)
                except:
                    raise ValueError("The model path is not valid")
        if plot_name is None:
            plot_name = [f"model {i}" for i in range(len(model_filenames))]
        self.plot_name = plot_name
        self.output_name = output_name
        self.result_folder = Path("data/results")
        if not self.result_folder.exists():
            self.result_folder.exists()
        self.baseline_result_file = self.result_folder / f"baseline_{self.output_name}.pkl"

        self.model_result_file = []
        for model_name in model_filenames:
            self.model_result_file.append(self.result_folder / f"{model_name[:-5]}_{self.output_name}.pkl")

        self.n_bootstraps = n_bootstraps
        self.rng_seed = 42  # control reproducibility

        # get the features names from the model
        self.features_names = self.model[0].get_booster().feature_names

        # ensure that the features names are in the column of the test data
        if not set(self.features_names).issubset(self.test_data.columns):
            raise ValueError("The features names in the model are not in the test data")

        self.test_data.dropna(subset=self.features_names, inplace=True)
        self.test_data.dropna(subset=[BASELINE_FEATURE], inplace=True)
        self.y_test = self.test_data["label"].to_numpy()

        self.train_data.dropna(subset=self.features_names, inplace=True)
        self.train_data.dropna(subset=[BASELINE_FEATURE], inplace=True)
        self.y_train = self.train_data["label"].to_numpy()

        print(f"Number of points in the test data: {len(self.test_data)}")
        print(f"Prevalence of hypotension: {self.test_data['label'].mean():.2%}")

        print("Test data loaded")

    def test_baseline(self):
        model_baseline = CalibratedClassifierCV()
        x_train = self.train_data[BASELINE_FEATURE].to_numpy().reshape(-1, 1)
        model_baseline.fit(x_train, self.y_train)
        x_test = self.test_data[BASELINE_FEATURE].to_numpy()
        self.y_pred_baseline = model_baseline.predict_proba(x_test.reshape(-1, 1))[:, 1]
        y_label_id = self.test_data["label_id"].to_numpy()

        print("Baseline test:")
        self.dict_results_baseline, _, _ = expe.bootstrap_test(
            self.y_test,
            self.y_pred_baseline,
            y_label_id,
            n_bootstraps=self.n_bootstraps,
            rng_seed=self.rng_seed,
            strategy="max_precision",
            target=0.24
        )

        for i in range(len(self.dict_results_baseline["threshold_opt"])):
            self.dict_results_baseline["threshold_opt"][i] = self.test_data[BASELINE_FEATURE].iloc[np.argmin(np.abs(
                self.dict_results_baseline["threshold_opt"][i] - self.y_pred_baseline))]

        for i in range(len(self.dict_results_baseline["threshold_precision_standard"])):
            for j in range(len(self.dict_results_baseline["threshold_precision_standard"][i])):
                self.dict_results_baseline["threshold_precision_standard"][i, j] = self.test_data[BASELINE_FEATURE].iloc[np.argmin(np.abs(
                    self.dict_results_baseline["threshold_precision_standard"][i, j] - self.y_pred_baseline))]

        for i in range(len(self.dict_results_baseline["op_threshold"])):
            for j in range(len(self.dict_results_baseline["op_threshold"][i])):
                self.dict_results_baseline["op_threshold"][i, j] = self.test_data[BASELINE_FEATURE].iloc[np.argmin(np.abs(
                    self.dict_results_baseline["op_threshold"][i, j] - self.y_pred_baseline))]

        self.baseline_recall = np.median(self.dict_results_baseline["recall_threshold"])
        with self.baseline_result_file.open("wb") as f:
            pickle.dump(self.dict_results_baseline, f)

    def test_model(self):
        self.y_pred_model = []
        self.dict_results_model = []
        for i, model in enumerate(self.model):
            self.y_pred_model.append(model.predict_proba(self.test_data[self.features_names])[:, 1])
            y_label_ids = self.test_data["label_id"].to_numpy()

            print(f"{self.plot_name[i]} test:")
            dict_result, _, _ = expe.bootstrap_test(
                self.y_test,
                self.y_pred_model[-1],
                y_label_ids,
                n_bootstraps=self.n_bootstraps,
                rng_seed=self.rng_seed,
                strategy="targeted_recall",
                target=self.baseline_recall
            )

            self.dict_results_model.append(dict_result)

            with self.model_result_file[i].open("wb") as f:
                pickle.dump(self.dict_results_model[-1], f)

    def load_baseline_results(self):
        with self.baseline_result_file.open("rb") as f:
            self.dict_results_baseline = pickle.load(f)
        self.baseline_recall = np.median(self.dict_results_baseline["recall_threshold"])
        model_baseline = CalibratedClassifierCV()
        x_train = self.train_data[BASELINE_FEATURE].to_numpy().reshape(-1, 1)
        model_baseline.fit(x_train, self.y_train)
        x_test = self.test_data[BASELINE_FEATURE].to_numpy()
        self.y_pred_baseline = model_baseline.predict_proba(x_test.reshape(-1, 1))[:, 1]

    def load_model_results(self):
        self.y_pred_model = []
        self.dict_results_model = []
        for i, model in enumerate(self.model):
            with self.model_result_file[i].open("rb") as f:
                self.dict_results_model.append(pickle.load(f))
            self.y_pred_model.append(self.model[i].predict_proba(self.test_data[self.features_names])[:, 1])

    def print_results(self):
        if not hasattr(self, "dict_results_baseline") or not hasattr(self, "dict_results_model"):
            raise ValueError("Results not loaded")
        print('\n')

        print(f"Results for {self.output_name}")
        print('Baseline')
        expe.print_statistics(self.dict_results_baseline)

        print('\n')
        for i, dict_results_model in enumerate(self.dict_results_model):
            print(self.plot_name[i])
            expe.print_statistics(dict_results_model)
            print('\n')
        return

    def update_results_to_min_map_threshold(self, min_map_threshold):
        # find recall associated with the min_map_threshold
        recall = np.linspace(0, 1, 1000)
        threshold = self.dict_results_baseline['op_threshold']
        self.recall_threshold_op = recall[np.argmin(np.abs(threshold.mean(0) - min_map_threshold))]

        for i, precision_op in enumerate(self.dict_results_baseline['op_precision']):
            precision_op[self.recall_threshold_op > recall] = 0
            self.dict_results_baseline['au_op_prcs'][i] = np.trapz(
                precision_op, recall
            ) / (1 - self.recall_threshold_op)

        # update aucs thanks to the new recall_threshold
        for i, dict_results_model in enumerate(self.dict_results_model):
            for j, precision_op in enumerate(dict_results_model['op_precision']):
                precision_op[self.recall_threshold_op > recall] = 0
                self.dict_results_model[i]['au_op_prcs'][j] = np.trapz(
                    precision_op, recall
                ) / (1 - self.recall_threshold_op)

        # same fpr standard precision recall curve
        threshold = self.dict_results_baseline['threshold_precision_standard']
        self.recall_threshold_std = recall[np.argmin(np.abs(threshold.mean(0) - min_map_threshold))]

        for i, precision_std in enumerate(self.dict_results_baseline['precision']):
            precision_std[self.recall_threshold_std > recall] = 0
            self.dict_results_baseline['auprcs'][i] = np.trapz(
                precision_std, recall
            ) / (1 - self.recall_threshold_std)
        for i, dict_results_model in enumerate(self.dict_results_model):
            for j, precision_op in enumerate(dict_results_model['precision']):
                precision_op[self.recall_threshold_std > recall] = 0
                self.dict_results_model[i]['auprcs'][j] = np.trapz(
                    precision_op, recall
                ) / (1 - self.recall_threshold_std)

    def plot_operationnal_precision_recall(self):
        if not hasattr(self, "dict_results_baseline") or not hasattr(self, "dict_results_model"):
            raise ValueError("Results not loaded")

        recall = np.linspace(0, 1, 1000)
        if not hasattr(self, "recall_threshold_op"):
            self.recall_threshold_op = 0

        for i, dict_results_model in enumerate(self.dict_results_model):
            precision_mean, precision_std = dict_results_model['op_precision'].mean(
                0), dict_results_model['op_precision'].std(0)
            plt.fill_between(
                recall, precision_mean - 2 * precision_std, precision_mean + 2 * precision_std, alpha=0.2
            )
            plt.plot(recall, precision_mean, label=f"{self.plot_name[i]} (AUPRC = {
                     expe.print_one_stat(pd.Series(dict_results_model['au_op_prcs']), False)})")

        # add baseline to the plot

        plt.fill_between(
            self.dict_results_baseline['fprs'],
            self.dict_results_baseline['op_precision'].mean(0) - 2 * self.dict_results_baseline['op_precision'].std(0),
            self.dict_results_baseline['op_precision'].mean(0) + 2 * self.dict_results_baseline['op_precision'].std(0),
            alpha=0.2,
        )
        plt.plot(
            self.dict_results_baseline['fprs'],
            self.dict_results_baseline['op_precision'].mean(0),
            label=f"baseline (AUPRC = {expe.print_one_stat(pd.Series(self.dict_results_baseline['au_op_prcs']), False)})",
        )

        plt.plot([0, 1], [self.dict_results_baseline['op_precision'].mean(0)[-1]]*2, "k--")
        plt.xlabel("Operationnal Recall")
        plt.ylabel("Operationnal Precision")
        plt.ylim(-0.05, 1.05)
        plt.xlim(self.recall_threshold_op, 1)
        plt.title("Operationnal PRC")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"output/{self.output_name}_operationnal_prc.pdf", bbox_inches='tight')
        plt.show()

    def plot_roc_curve(self):
        if not hasattr(self, "dict_results_baseline") or not hasattr(self, "dict_results_model"):
            raise ValueError("Results not loaded")

        fpr = np.linspace(0, 1, 1000)
        for i, dict_results_model in enumerate(self.dict_results_model):
            tpr_mean, tpr_std = dict_results_model['tpr'].mean(
                0), dict_results_model['tpr'].std(0)
            plt.fill_between(
                fpr, tpr_mean - 2 * tpr_std, tpr_mean + 2 * tpr_std, alpha=0.2
            )
            plt.plot(fpr, tpr_mean, label=f"{self.plot_name[i]} (AUROC = {
                     expe.print_one_stat(pd.Series(dict_results_model['aucs']), False)})")

        # add baseline to the plot
        plt.fill_between(
            self.dict_results_baseline['fprs'],
            self.dict_results_baseline['tpr'].mean(0) - 2 * self.dict_results_baseline['tpr'].std(0),
            self.dict_results_baseline['tpr'].mean(0) + 2 * self.dict_results_baseline['tpr'].std(0),
            alpha=0.2,
        )
        plt.plot(
            self.dict_results_baseline['fprs'],
            self.dict_results_baseline['tpr'].mean(0),
            label=f"baseline (AUROC = {expe.print_one_stat(pd.Series(self.dict_results_baseline['aucs']), False)})",
        )

        plt.grid()
        plt.xlabel("Sensitivity")
        plt.ylabel("Specificity")
        plt.title("ROC curve")

        plt.legend()
        plt.tight_layout()
        plt.savefig(f"output/{self.output_name}_roc.pdf", bbox_inches='tight')
        plt.show()

    def plot_pr_curve(self):
        if not hasattr(self, "dict_results_baseline") or not hasattr(self, "dict_results_model"):
            raise ValueError("Results not loaded")

        if not hasattr(self, "recall_threshold_std"):
            self.recall_threshold_std = 0
        recall = np.linspace(0, 1, 1000)
        for i, dict_results_model in enumerate(self.dict_results_model):
            precision_mean, precision_std = dict_results_model['precision'].mean(
                0), dict_results_model['precision'].std(0)
            plt.fill_between(
                recall, precision_mean - 2 * precision_std, precision_mean + 2 * precision_std, alpha=0.2
            )
            plt.plot(recall, precision_mean, label=f"{self.plot_name[i]} (AUPRC = {
                     expe.print_one_stat(pd.Series(dict_results_model['auprcs']), False)})")

        # add baseline to the plot

        plt.fill_between(
            self.dict_results_baseline['fprs'],
            self.dict_results_baseline['precision'].mean(0) - 2 * self.dict_results_baseline['precision'].std(0),
            self.dict_results_baseline['precision'].mean(0) + 2 * self.dict_results_baseline['precision'].std(0),
            alpha=0.2,
        )
        plt.plot(
            self.dict_results_baseline['fprs'],
            self.dict_results_baseline['precision'].mean(0),
            label=f"baseline (AUPRC = {expe.print_one_stat(pd.Series(self.dict_results_baseline['au_op_prcs']), False)})",
        )

        plt.plot([0, 1], [self.dict_results_baseline['precision'].mean(0)[-1]]*2, "k--")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.ylim(-0.05, 1.05)
        plt.xlim(self.recall_threshold_std, 1)
        plt.title("PRC")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"output/{self.output_name}_prc.pdf", bbox_inches='tight')
        plt.show()

    def plot_calibration_curve(self, n_bins: float = 6):

        fraction_of_positives_baseline, mean_predicted_value_baseline = calibration_curve(
            self.y_test,
            self.y_pred_baseline,
            n_bins=n_bins,
            strategy='uniform'
        )
        # Compute the number of samples in each bin
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_counts = np.histogram(self.y_pred_baseline, bins=bin_edges)[0]  # Count of samples per bin
        bin_counts = bin_counts[bin_counts > 0]

        # Normalize marker sizes for visualization
        min_size, max_size = 20, 300  # Min and max marker sizes
        marker_sizes_base = np.interp(bin_counts, (bin_counts.min(), bin_counts.max()), (min_size, max_size))

        # Compute ECE
        # ece_baseline = expected_calibration_error(self.y_pred_baseline, self.y_test, M=n_bins)
        brier_baseline = brier_score_loss(self.y_test, self.y_pred_baseline)

        plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='perfectly calibrated')
        for i, y_pred in enumerate(self.y_pred_model):
            fraction_of_positives_model, mean_predicted_value_model = calibration_curve(
                self.y_test,
                y_pred,
                n_bins=n_bins,
                strategy='uniform'
            )
            # Compute ECE
            # ece_model = expected_calibration_error(y_pred, self.y_test, M=n_bins)
            brier_model = brier_score_loss(self.y_test, y_pred)
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_counts = np.histogram(y_pred, bins=bin_edges)[0]  # Count of samples per bin
            bin_counts = bin_counts[bin_counts > 0]

            # Normalize marker sizes for visualization
            min_size, max_size = 20, 300  # Min and max marker sizes
            marker_sizes = np.interp(bin_counts, (bin_counts.min(), bin_counts.max()), (min_size, max_size))

            plt.plot(mean_predicted_value_model, fraction_of_positives_model,
                     label=f'{self.plot_name[i]} (brier={brier_model:.3f})', color=BASIC_COLORS[i], linewidth=0.5)
            plt.scatter(mean_predicted_value_model, fraction_of_positives_model,
                        s=marker_sizes, color=BASIC_COLORS[i], alpha=0.5)

        plt.plot(mean_predicted_value_baseline, fraction_of_positives_baseline,
                 label=f'baseline (brier={brier_baseline:.3f})', linewidth=0.5, color=BASIC_COLORS[i+1])
        plt.scatter(mean_predicted_value_baseline, fraction_of_positives_baseline,
                    s=marker_sizes_base, color=BASIC_COLORS[i+1], alpha=0.5)
        plt.xlabel('Predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title('Calibration curve')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"output/{self.output_name}_calibration.pdf", bbox_inches='tight')
        plt.show()

        fig, ax = plt.subplots(1, len(self.y_pred_model)+1, layout='constrained', sharey=True)
        ax[0].hist(self.y_pred_baseline, bins=20, alpha=0.5, color='orange')
        ax[0].set_title('Baseline')
        ax[0].set_xlim(0, 1)
        for i, y_pred in enumerate(self.y_pred_model):

            ax[i+1].hist(y_pred, bins=20, alpha=0.5)
            ax[i+1].set_title(f'{self.plot_name[i]}')
            ax[i+1].set_xlim(0, 1)
        fig.suptitle('Histogram of predicted probabilities')
        plt.tight_layout()
        plt.savefig(f"output/{self.output_name}_distrib.pdf", bbox_inches='tight')
        plt.show()

    def run(self,
            force_baseline_computation=False,
            force_model_computation=False,):

        if force_baseline_computation or not self.baseline_result_file.exists():
            self.test_baseline()
        else:
            self.load_baseline_results()

        if force_model_computation or not np.all([filename.exists() for filename in self.model_result_file]):
            self.test_model()
        else:
            self.load_model_results()

        self.print_results()
        self.plot_operationnal_precision_recall()
        self.plot_calibration_curve()
        self.plot_roc_curve()
        self.plot_pr_curve()

    def compute_shap_value(self, model_id=0):
        # use SHAP to explain the model
        shap.initjs()
        explainer = shap.TreeExplainer(self.model[model_id])
        self.shap_values = explainer.shap_values(self.test_data[self.features_names])

    def plot_shap_values(self,
                         nb_max_feature=10,
                         ):
        # plot the SHAP value
        test_data = self.test_data[self.features_names]
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        for i in range(nb_max_feature):
            plt.axhline(y=i, color='black', linestyle='--', linewidth=0.5)
        shap.summary_plot(self.shap_values, test_data, feature_names=self.features_names,
                          show=False, plot_type="bar", max_display=nb_max_feature)
        plt.xlabel('mean($|$SHAP value$|$)')
        names = plt.gca().get_yticklabels()
        names = [name.get_text().replace("constant", "intercept") for name in names]
        names = [name.replace("mbp", "MAP") for name in names]
        names = [name.replace("sbp", "SAP") for name in names]
        names = [name.replace("dbp", "DAP") for name in names]
        names = [name.replace("hr", "HR") for name in names]
        names = [name.replace("rf_ct", "RF_CT") for name in names]
        plt.gca().set_yticklabels(names)
        plt.subplot(1, 2, 2)
        shap.summary_plot(self.shap_values, test_data, feature_names=self.features_names,
                          show=False, max_display=nb_max_feature)
        # remove the y thick label
        plt.gca().set_yticklabels([])
        plt.xlabel('SHAP value')
        plt.tight_layout()
        # add horizontal line for each feture
        for i in range(nb_max_feature):
            plt.axhline(y=i, color='black', linestyle='--', linewidth=0.5)

        plt.show()
        return

    def group_shap_values(self):
        # groups = {
        #     'MAP_intercept': [name for name in self.features_names if 'mbp_constant' in name],
        #     'MAP_slope': [name for name in self.features_names if 'mbp_slope' in name],
        #     'MAP_std': [name for name in self.features_names if 'mbp_std' in name],
        #     'DAP_intercept': [name for name in self.features_names if 'dbp_constant' in name],
        #     'DAP_slope': [name for name in self.features_names if 'dbp_slope' in name],
        #     'DAP_std': [name for name in self.features_names if 'dbp_std' in name],
        #     'SAP_intercept': [name for name in self.features_names if 'sbp_constant' in name],
        #     'SAP_slope': [name for name in self.features_names if 'sbp_slope' in name],
        #     'SAP_std': [name for name in self.features_names if 'sbp_std' in name],
        #     'MAC_intercept': [name for name in self.features_names if 'mac_constant' in name],
        #     'MAC_slope': [name for name in self.features_names if 'mac_slope' in name],
        #     'MAC_std': [name for name in self.features_names if 'mac_std' in name],
        #     'HR_intercept': [name for name in self.features_names if 'hr_constant' in name],
        #     'HR_slope': [name for name in self.features_names if 'hr_slope' in name],
        #     'HR_std': [name for name in self.features_names if 'hr_std' in name],
        #     'RR_intercept': [name for name in self.features_names if 'rr_constant' in name],
        #     'RR_slope': [name for name in self.features_names if 'rr_slope' in name],
        #     'RR_std': [name for name in self.features_names if 'rr_std' in name],
        #     'SPO2_intercept': [name for name in self.features_names if 'spo2_constant' in name],
        #     'SPO2_slope': [name for name in self.features_names if 'spo2_slope' in name],
        #     'SPO2_std': [name for name in self.features_names if 'spo2_std' in name],
        #     'ETCO2_intercept': [name for name in self.features_names if 'etco2_constant' in name],
        #     'ETCO2_slope': [name for name in self.features_names if 'etco2_slope' in name],
        #     'ETCO2_std': [name for name in self.features_names if 'etco2_std' in name],
        #     'PROPO_intercept': [name for name in self.features_names if 'pp_ct_constant' in name],
        #     'PROPO_slope': [name for name in self.features_names if 'pp_ct_slope' in name],
        #     'PROPO_std': [name for name in self.features_names if 'pp_ct_std' in name],
        #     'REMI_intercept': [name for name in self.features_names if 'rf_ct_constant' in name],
        #     'REMI_slope': [name for name in self.features_names if 'rf_ct_slope' in name],
        #     'REMI_std': [name for name in self.features_names if 'rf_ct_std' in name],
        #     'TEMP_intercept': [name for name in self.features_names if 'body_temp_constant' in name],
        #     'TEMP_slope': [name for name in self.features_names if 'body_temp_slope' in name],
        #     'TEMP_std': [name for name in self.features_names if 'body_temp_std' in name],
        #     'AGE': ['age'],
        #     'BMI': ['bmi'],
        #     'ASA': ['asa'],
        #     'PREOP_CR': ['preop_cr'],
        #     'PREOP_HTN': ['preop_htn'],
        # }

        groups = {
            'MAP': [name for name in self.features_names if 'mbp' in name],
            'DAP': [name for name in self.features_names if 'dbp' in name],
            'SAP': [name for name in self.features_names if 'sbp' in name],
            'MAC': [name for name in self.features_names if 'mac' in name],
            'HR': [name for name in self.features_names if 'hr' in name],
            'RR': [name for name in self.features_names if 'rr' in name],
            'SPO2': [name for name in self.features_names if 'spo2' in name],
            'ETCO2': [name for name in self.features_names if 'etco2' in name],
            'PROPO': [name for name in self.features_names if 'pp_ct' in name],
            'REMI': [name for name in self.features_names if 'rf_ct' in name],
            'TEMP': [name for name in self.features_names if 'body_temp' in name],
            'AGE': ['age'],
            'BMI': ['bmi'],
            'ASA': ['asa'],
            'PREOP_CR': ['preop_cr'],
            'PREOP_HTN': ['preop_htn'],
            'cycle_mean': [name for name in self.features_names if 'cycle_mean' in name],
            'cycle_std': [name for name in self.features_names if 'cycle_std' in name],
            'cycle_systol': [name for name in self.features_names if 'cycle_systol' in name],
            'cycle_diastol': [name for name in self.features_names if 'cycle_diastol' in name],
            'cycle_duration': [name for name in self.features_names if 'cycle_duration' in name],
            'dP/dt_max': [name for name in self.features_names if 'dpdt_max' in name],
            'dP/dt_min': [name for name in self.features_names if 'dpdt_min' in name],
            'dP/dt_mean': [name for name in self.features_names if 'dpdt_mean' in name],
            'dP/dt_std': [name for name in self.features_names if 'dpdt_std' in name],
        }

        self.shap_grouped = _grouped_shap(self.shap_values, self.features_names, groups)
        self.test_data_group = _grouped_shap(self.test_data[self.features_names], self.features_names, groups)
        return

    def plot_shap_grouped(self, nb_max_feature=10):
        font_size = 16

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        for i in range(nb_max_feature):
            plt.axhline(y=i, color='black', linestyle='--', linewidth=0.5)
        shap.summary_plot(self.shap_grouped.values, self.test_data_group.values, feature_names=self.shap_grouped.columns,
                          show=False, plot_type="bar", max_display=nb_max_feature)
        plt.xlabel('mean($|$SHAP value$|$)', fontsize=font_size)
        ax = plt.gca()
        ax.tick_params(axis='y', labelsize=font_size)

        plt.subplot(1, 2, 2)
        shap.summary_plot(self.shap_grouped.values, self.test_data_group.values,
                          max_display=nb_max_feature, show=False)
        plt.xlabel('SHAP value', fontsize=font_size)
        # remove the y thick label
        ax = plt.gca()
        ax.set_yticklabels([])
        for i in range(nb_max_feature):
            plt.axhline(y=i, color='black', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()
