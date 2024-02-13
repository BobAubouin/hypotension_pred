import vitaldb as vdb
import numpy as np
import pandas as pd


def dataGen(Windows_length: int) -> pd.DataFrame:
    """_summary_

    Parameters
    ----------
    Windows_length : int
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
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

    # list of features
    solar8000_features = ['ART_MBP', 'ART_SBP', 'ART_DBP', 'HR', 'RR', 'PLETH_SPO2', 'ETCO2']
