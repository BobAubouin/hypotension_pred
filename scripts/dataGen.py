import multiprocessing as mp
from functools import partial
import vitaldb as vdb
import pandas as pd
from typing import List
from tqdm import tqdm


def downloadOneCase(caseid: int,
                    signal_list: List[str],
                    static_data: List[str],
                    sampling_time: float,
                    df_cases: pd.DataFrame) -> pd.DataFrame:
    """downloadOneCase function downloads the data from the vitaldb for a single case and returns the data in a pandas dataframe.

    Parameters
    ----------
    caseid : int
        Case id to download.
    signal_list : List[str]
        signal list to download.
    static_data : List[str]
        static data to download.
    sampling_time : float
        sampling time for the signals (in seconds).
    df_cases : pd.DataFrame
        dataframe containing the cases information.

    Returns
    -------
    pd.DataFrame
        Returns the data in a pandas dataframe or None if empty.
    """
    data_patient = vdb.VitalFile(caseid, signal_list)
    data_patient = data_patient.to_pandas(signal_list, sampling_time)
    data_patient.insert(0, 'caseid', caseid)
    # check if there is enought data for each signal
    flag_signal = False
    for signal in signal_list:
        if 'Solar8000' in signal:  # Solar8000 has 0.5Hz sampling rate
            device_sampling_time = 2
        elif 'Orchestra' in signal:  # ok if missing propofol data
            continue
        elif 'Primus' in signal:  # Primus has 1/7Hz sampling rate
            device_sampling_time = 7

        # if more than 40% of the data is missing, skip the case
        if data_patient[signal].isna().sum() > len(data_patient) * (1 - sampling_time/device_sampling_time)*(1+0.4):
            # print(f"Case {caseid} has more than 30% missing data for {signal}. Skipping case.")
            return None

    # download static data
    data_patient_static = df_cases[df_cases['caseid'] == caseid][static_data]
    data_patient_static['caseid'] = caseid
    if data_patient_static.isna().sum().sum() > 0:  # if there is missing static data, skip the case
        return None

    # merge the data
    data_patient = pd.merge(data_patient, data_patient_static, on='caseid')

    return data_patient


def dataGen(sampling_time: float, signal_list: List[str], static_data: List[str]) -> pd.DataFrame:
    """dataGEn function downloads the data from the vitaldb and returns the data in a pandas dataframe.
    Columns of the output dataframe are the caseid + signals name +  static_data.

    Parameters
    ----------
    sampling_time : int
        Sampling time of the downloaded data.
    signal_list : List[str]
        List of signals name to download.
    static_data : List[str]
        List of static data to download.

    Returns
    -------
    pd.DataFrame
        Returns the data in a pandas dataframe.
    """
    df_trks = pd.read_csv('https://api.vitaldb.net/trks')  # read track list
    df_cases = pd.read_csv("https://api.vitaldb.net/cases")  # read case information

    # select case list
    caseids = list(
        set(df_trks[df_trks['tname'] == 'Solar8000/ART_MBP']['caseid']) &  # select cases with arterial blood pressure
        set(df_cases[df_cases['age'] > 18]['caseid']) &  # select cases with age > 18
        set(df_cases[df_cases['caseend'] > 3600]['caseid']) &  # select cases with duration > 1 hour
        set(df_cases[~df_cases['opname'].str.contains("transplant", case=False)]
            ['caseid'])  # exclude cases with 'transplant' in operation name
    )

    print(f"Number of cases to consider: {len(caseids)}")

    # download the selected data
    data = pd.DataFrame()

    temp_function = partial(downloadOneCase, signal_list=signal_list, static_data=static_data,
                            sampling_time=sampling_time, df_cases=df_cases)
    with mp.Pool(mp.cpu_count()) as pool:
        data_patient = list(tqdm(pool.map(temp_function, caseids), total=len(caseids), desc="Downloading data"))
    data_patient = [x for x in data_patient if x is not None]
    data = pd.concat(data_patient)

    print(f"Number of cases with enough data: {len(data['caseid'].unique())}")
    return data


if __name__ == "__main__":
    sampling_time = 1
    # list of features
    solar8000_signals = ['ART_MBP', 'ART_SBP', 'ART_DBP', 'HR', 'RR', 'PLETH_SPO2', 'ETCO2']
    orchestra_signals = ['PPF20_CT']
    primus_signals = ['MAC']
    signals = ['Solar8000/'+f for f in solar8000_signals] + \
        ['Orchestra/'+f for f in orchestra_signals] + \
        ['Primus/'+f for f in primus_signals]

    # list of static data
    static = ['age', 'bmi', 'asa', 'preop_cr', 'preop_htn', 'opname']

    dataframe = dataGen(sampling_time, signals, static)
    print(dataframe.head())
    dataframe.to_csv('data.csv', index=False)
