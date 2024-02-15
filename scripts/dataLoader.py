from typing import List
import pandas as pd
import numpy as np


def dataLoader(halft_times: List[int] = [10, 60, 5*60],
               signal_name: List[str] = ['mbp'],
               static_data: List[str] = [],
               delay_prediction: int = 5*60,
               sampling_time: int = 1
               ):
    """dataLoader function loads data genrated by dataGen function and process the data to be used in the model.

    Parameters
    ----------
    halft_times : List[int] 
        List of half-time to use in exponential moving average and variance (seconds). Default is [10, 60, 5*60].
    signal_name : List[str]
        List of signals name to consider. Default is ['mbp'].
    static_data : List[str]
        List of static data to consider. Default is [].
    delay_prediction : int
        Delay time for the prediction (seconds). Default is 5*60.
    sampling_time : int
        Sampling time of the generated data (seconds).

    Returns
    -------
    pd.DataFrame
        Returns the data in a pandas dataframe.
    """

    # Load the data
    rawData = pd.read_csv(f'data/rawData_{sampling_time}s.csv')
    # rename the raw to have simpler names
    rawData.rename(columns={'Solar8000/ART_MBP': 'mbp',
                            'Solar8000/ART_SBP': 'sbp',
                            'Solar8000/ART_DBP': 'dbp',
                            'Solar8000/HR': 'hr',
                            'Solar8000/RR': 'rr',
                            'Solar8000/PLETH_SPO2': 'spo2',
                            'Solar8000/ETCO2': 'etco2',
                            'Orchestra/PPF20_CT': 'pp_ct',
                            'Primus/MAC': 'mac'}, inplace=True)

    formattedData = pd.DataFrame()

    for df_case in rawData.groupby('caseid'):
        formattedData_case = pd.DataFrame()
        # process mean arterial value data
        # remove too low value (before the start of the measurement)
        df_case.mbp.mask(df_case.mbp < 40, inplace=True)
        df_case.mbp.mask(df_case.mbp > 150, inplace=True)  # remove too high value (due to peak or flush)

        # removing the nan values at the beginning and the ending
        case_valid_mask = ~df_case.mbp.isna()
        df_case = df_case.iloc[(np.cumsum(case_valid_mask) > 0) & (np.cumsum(case_valid_mask[::-1])[::-1] > 0)]

        # create the time series features
        for half_time in halft_times:
            for signal in signal_name:
                formattedData_case[f'{signal}_ema_{half_time}'] = df_case[signal].ewm(halflife=half_time).mean()
                formattedData_case[f'{signal}_var_{half_time}'] = df_case[signal].ewm(halflife=half_time).std()

        # create the static features
        for static in static_data:
            formattedData_case[static] = df_case[static].iloc[0]

        # create the label
        formattedData_case['pred_mbp'] = df_case.mbp.shift(-delay_prediction//sampling_time)
        formattedData_case['pred_hp'] = False
        # standard label with mbp < 65 mmHg for 1 consecutive minutes
        for i in range(len(df_case) - (60 + delay_prediction)//sampling_time):
            if (df_case.mbp.iloc[i+delay_prediction//sampling_time:i+(60 + delay_prediction)//sampling_time] < 65).all():
                formattedData_case['pred_hp'].iloc[i] = True

        formattedData = pd.concat([formattedData, formattedData_case])

    return formattedData
