# Hypotension_pred

Use a data-based approach to predict intra-operative hypotension.

## Installation

Use a new virtual env and Python 3.11 (with pyenv) for maximal compatibility.

```bash
git clone https://github.com/BobAubouin/hypotension_pred hp_pred
cd hp_pred
pip install .
```

### Dev / Contribution

In addition, you can add the optional build `dev`. So you will download the Python packages required to develop the project (unit test, linter, formatter).

```bash
git clone https://github.com/BobAubouin/hypotension_pred hp_pred
cd hp_pred
pip install -e .[dev]
```

## Use

### Download raw data from VitalDB

The data used are from the [VitalDB](https://vitaldb.net/) open dataset. You must read the [Data Use Agreement](https://vitaldb.net/dataset/#h.vcpgs1yemdb5) before using it.

 To download the data you can use the package's command `python -m hp_pred.dataset_download`. The help command outputs the following:

```bash
usage: dataset_download.py [-h] [-l {CRITICAL,FATAL,ERROR,WARN,WARNING,INFO,DEBUG,NOTSET}] [-s GROUP_SIZE] [-o OUTPUT_FOLDER]

Download the VitalDB data for hypertension prediction.

options:
  -h, --help            show this help message and exit
  -l {CRITICAL,FATAL,ERROR,WARN,WARNING,INFO,DEBUG,NOTSET}, --log_level_name {CRITICAL,FATAL,ERROR,WARN,WARNING,INFO,DEBUG,NOTSET}
                        The logger level name to generate logs. (default: INFO)
  -s GROUP_SIZE, --group_size GROUP_SIZE
                        Amount of cases dowloaded and processed. (default: 950)
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        The folder to store the data and logs. (default: data)
```

### Create the segmented dataset

The class `hp_pred.databuilder.DataBuilder` is used to create the segmented dataset with a sliding window approach. An example of use is given in the `scripts/dataset_build/base_dataset.py` scripts.

### Recreate CDC results

The results associated with our paper can be replicated using the version of the git tagged 'cdc_XP'. You should create the database running the previously mentioned step. Then you should run the notebooks in the `scripts/experiments` folder. This exact order of running must be respected:

- `baseline.ipynb`
- `xgboos-model.ipynb`
- `study_leading_time.ipynb`

## Citation

If you use this code in your research, please cite our paper.
