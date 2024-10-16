from pathlib import Path
from itertools import chain, repeat

import pandas as pd
import shap
from matplotlib import pyplot as plt
import xgboost as xgb


nb_max_feature = 10
model_file = 'data/models/xgb.json'
database = 'base_dataset'
name_fig_1 = 'shap_xgboost_manu.pdf'
name_fig_2 = 'shap_xgboost_group.pdf'

output_folder = Path('output')
output_folder.mkdir(exist_ok=True)

SIGNAL_FEATURE = ['mbp', 'sbp', 'dbp', 'hr', 'rr', 'spo2', 'etco2', 'mac', 'pp_ct']
STATIC_FEATURE = ["age", "bmi", "asa"]
HALF_TIME_FILTERING = [60, 3*60, 10*60]
FEATURE_NAME = (
    [
        f"{signal}_constant_{half_time}"
        for signal in SIGNAL_FEATURE
        for half_time in HALF_TIME_FILTERING
    ]
    + [
        f"{signal}_slope_{half_time}"
        for signal in SIGNAL_FEATURE
        for half_time in HALF_TIME_FILTERING
    ]
    + [
        f"{signal}_std_{half_time}"
        for signal in SIGNAL_FEATURE
        for half_time in HALF_TIME_FILTERING
    ]
    + STATIC_FEATURE
)




#load data
data = pd.read_parquet(Path(f'data/datasets/{database}/cases/'))

static = pd.read_parquet(f'data/datasets/{database}/meta.parquet')

data = data.merge(static, on='caseid')

test = data[data['split'] == "test"]

#load model
model = xgb.Booster()
model.load_model(model_file)

#explain the model

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(test[FEATURE_NAME])
test_data = test[FEATURE_NAME]
print("Data loaded and model explained")

#plot the figures
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for i in range(nb_max_feature):
    plt.axhline(y=i, color='black', linestyle='--', linewidth=0.5)
shap.summary_plot(shap_values, test_data, feature_names=FEATURE_NAME, show=False, plot_type="bar", max_display=nb_max_feature)
plt.xlabel('mean($|$SHAP value$|$)')
plt.subplot(1, 2, 2)
shap.summary_plot(shap_values, test_data, feature_names=FEATURE_NAME, show=False, max_display=nb_max_feature)
#remove the y thick label
plt.gca().set_yticklabels([])
plt.xlabel('SHAP value')
plt.tight_layout()
#add horizontal line for each feture
for i in range(nb_max_feature):
    plt.axhline(y=i, color='black', linestyle='--', linewidth=0.5)
plt.savefig(f'./output/{name_fig_1}', bbox_inches='tight', dpi=300)
plt.close()

print("SHAP values plotted")

revert_dict = lambda d: dict(chain(*[zip(val, repeat(key)) for key, val in d.items()]))

def grouped_shap(shap_vals, features, groups):
    groupmap = revert_dict(groups)
    shap_Tdf = pd.DataFrame(shap_vals, columns=pd.Index(features, name='features')).T
    shap_Tdf['group'] = shap_Tdf.reset_index().features.map(groupmap).values
    shap_grouped = shap_Tdf.groupby('group').sum().T
    return shap_grouped

# Group features
feature_group = {
    'MAP_std': [name for name in FEATURE_NAME if 'mbp_std' in name],
    'DAP_std': [name for name in FEATURE_NAME if 'dbp_std' in name],
    'SAP_std': [name for name in FEATURE_NAME if 'sbp_std' in name],
    'MAC_std': [name for name in FEATURE_NAME if 'mac_std' in name],
    'HR_std': [name for name in FEATURE_NAME if 'hr_std' in name],
    'RR_std': [name for name in FEATURE_NAME if 'rr_std' in name],
    'SPO2_std': [name for name in FEATURE_NAME if 'spo2_std' in name],
    'ETCO2_std': [name for name in FEATURE_NAME if 'etco2_std' in name],
    'PROPO_std': [name for name in FEATURE_NAME if 'pp_ct_std' in name],
    'AGE': ['age'],
    'BMI': ['bmi'],
    'ASA': ['asa'],
    'PREOP_CR': ['preop_cr'],
    'PREOP_HTN': ['preop_htn'],
    'MAP_constant': [name for name in FEATURE_NAME if 'mbp_constant' in name],
    'MAP_slope': [name for name in FEATURE_NAME if 'mbp_slope' in name],
    'DAP_constant': [name for name in FEATURE_NAME if 'dbp_constant' in name],
    'DAP_slope': [name for name in FEATURE_NAME if 'dbp_slope' in name],
    'SAP_constant': [name for name in FEATURE_NAME if 'sbp_constant' in name],
    'SAP_slope': [name for name in FEATURE_NAME if 'sbp_slope' in name],
    'MAC_constant': [name for name in FEATURE_NAME if 'mac_constant' in name],
    'MAC_slope': [name for name in FEATURE_NAME if 'mac_slope' in name],
    'HR_constant': [name for name in FEATURE_NAME if 'hr_constant' in name],
    'HR_slope': [name for name in FEATURE_NAME if 'hr_slope' in name],
    'RR_constant': [name for name in FEATURE_NAME if 'rr_constant' in name],
    'RR_slope': [name for name in FEATURE_NAME if 'rr_slope' in name],
    'SPO2_constant': [name for name in FEATURE_NAME if 'spo2_constant' in name],
    'SPO2_slope': [name for name in FEATURE_NAME if 'spo2_slope' in name],
    'ETCO2_constant': [name for name in FEATURE_NAME if 'etco2_constant' in name],
    'ETCO2_slope': [name for name in FEATURE_NAME if 'etco2_slope' in name],
    'PROPO_constant': [name for name in FEATURE_NAME if 'pp_ct_constant' in name],
    'PROPO_slope': [name for name in FEATURE_NAME if 'pp_ct_slope' in name],
}

shap_group = grouped_shap(shap_values, FEATURE_NAME, feature_group)
test_data_group = grouped_shap(test_data, FEATURE_NAME, feature_group)

font_size = 16

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for i in range(nb_max_feature):
    plt.axhline(y=i, color='black', linestyle='--', linewidth=0.5)
shap.summary_plot(shap_group, test_data_group, feature_names=shap_group.columns, show=False, plot_type="bar", max_display=nb_max_feature)
plt.xlabel('mean($|$SHAP value$|$)', fontsize=font_size)
ax = plt.gca()
ax.tick_params(axis='y', labelsize=font_size)

plt.subplot(1, 2, 2)
shap.summary_plot(shap_group.values, test_data_group, feature_names=shap_group.columns, max_display=nb_max_feature, show=False)
plt.xlabel('SHAP value', fontsize=font_size)
#remove the y thick label
ax = plt.gca()
ax.set_yticklabels([])
for i in range(nb_max_feature):
    plt.axhline(y=i, color='black', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(f'./output/{name_fig_1}', bbox_inches='tight', dpi=600)
plt.close()

print("SHAP values grouped and plotted")