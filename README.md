# hypotension_pred
Use data-based approach to predict intra-operative hypotension.

## Installation

Use a new virtual env and Python 3.10 (with pyenv) for maximal compatibility.

```bash
git clone https://github.com/BobAubouin/hypotension_pred hp_pred
cd hp_pred
pip install .
```

### Dev / Contribution

In addition, you can add the optional build `dev`. So you will download the python packages required to develop on the project (unit test, linter, formatter).

```bash
git clone https://github.com/BobAubouin/hypotension_pred hp_pred
cd hp_pred
pip install -e .[dev]
```

## Use

### Download the dataset

The data used are from the [VitalDB](https://vitaldb.net/) open dataset. To download them you can use the package's command `python -m hp_pred.dataset_download`. The help command outputs the following:

```
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
