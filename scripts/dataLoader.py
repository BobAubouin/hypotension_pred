import pandas as pd
import numpy as np
from tqdm import tqdm

MIN_TIME_IOH = 60  # minimum time for the IOH to be considered as IOH (in seconds)
MIN_VALUE_IOH = 65  # minimum value for the mean arterial pressure to be considered as IOH (in mmHg)
MIN_MPB_SGEMENT = 40  # minimum acceptable value for the mean arterial pressure (in mmHg)
MAX_MPB_SGEMENT = 150  # maximum acceptable value for the mean arterial pressure (in mmHg)
MAX_DELTA_MPB_SGEMENT = 30  # maximum acceptable value for the mean arterial pressure variation (in mmHg)
MAX_NAN_SGEMENT = 0.1  # maximum acceptable value for the nan in the segment (in %)
NUMBER_CV_FOLD = 5  # number of cross-validation fold

DEVICE_NAME_TO_SAMPLING_RATE = {
    "Solar8000": 2,
    "Primus": 7,
}
POSSIBLE_SIGNAL_NAME = ['mbp', 'sbp', 'dbp', 'hr', 'rr', 'spo2', 'etco2', 'mac', 'pp_ct']


def validate_segment(segment: pd.DataFrame, sampling_time: int, observation_windows: int, leading_time: int):
    """ Validate_segment function validate the segment to be used in the model.

    The cionditions to validate the segment are:
    - The mean arterial pressure (mbp) is between 40 and 150 mmHg.
    - The variation of the mean arterial pressure (mbp) is less than 30 mmHg/minutes.
    - The label is not in the observation window.
    - The nan values are less than 10% of the segment.

    Parameters
    ----------
    segment : pd.DataFrame
        The segment data containing the features and labels.
    sampling_time : int
        The time interval between consecutive samples in seconds.
    observation_windows : int
        The length of the observation window in seconds.
    leading_time : int
        The length of the leading time in seconds.

    Returns
    -------
    bool
        True if the segment is valid, False otherwise.
    """
    segment_length = segment.shape[0]
    # test any map<40mmHg
    if (segment.mbp < MIN_MPB_SGEMENT).any():
        return False
    # test any map>150mmHg
    if (segment.mbp > MAX_MPB_SGEMENT).any():
        return False
    # test any delta MAP > 30mmHg/minutes
    if (segment.mbp.max() - segment.mbp.min()) > MAX_DELTA_MPB_SGEMENT:
        return False
    # test any label in the observation window
    if segment.label[:(observation_windows+leading_time)//sampling_time].any():
        return False
    # test too much nan in the segment
    # flag = False
    # for signal in POSSIBLE_SIGNAL_NAME:
    #     if signal == 'mac':
    #         device_rate = DEVICE_NAME_TO_SAMPLING_RATE['Primus']
    #     elif signal == 'pp_ct':
    #         device_rate = 1
    #     else:
    #         device_rate = DEVICE_NAME_TO_SAMPLING_RATE['Solar8000']

    #     nan_ratio = max(0, 1 - sampling_time/device_rate)
    #     threshold = nan_ratio + (1 - nan_ratio)*MAX_NAN_SGEMENT

    #     if segment[signal].isna().sum() > threshold*segment_length:
    #         flag = True
    #         break
    # if flag:
    #     return False

    return True


def dataLoader(half_times: list[int] = [10, 60, 5*60],
               signal_name: list[str] = ['mbp'],
               static_data: list[str] = [],
               leading_time: int = 3*60,
               prediction_windows: int = 7*60,
               observation_windows: int = 5*60,
               sampling_time: int = 2,
               window_shift: int = 30,
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
    leading_time : int
        Delay time for the prediction (seconds). Default is 3*60.
    prediction_windows : int
        Length of the prediction window (seconds). Default is 7*60.
    observation_windows : int
        Length of the observation window (seconds). Default is 5*60.
    sampling_time : int
        Sampling time of the generated data (seconds).
    window_shift : int
        Shift between the segments (seconds). Default is 30.

    Returns
    -------
    pd.DataFrame
        Returns the data in a pandas dataframe.
    """
    for signal in signal_name:
        if signal not in POSSIBLE_SIGNAL_NAME:
            raise ValueError(f'{signal} is not a possible signal name')

    # Load the data
    rawData = pd.read_csv(f'data/data_async.csv')
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

    # fill the nan value by 0 in the propo target
    rawData['pp_ct'].fillna(0, inplace=True)

    formattedData = pd.DataFrame()

    number_of_raw_segment = 0
    number_of_selected_segement = 0

    # get all caseid and separate them in 5 groups
    caseid_list = rawData['caseid'].unique()
    np.random.seed(0)
    np.random.shuffle(caseid_list)
    caseid_list = np.array_split(caseid_list, NUMBER_CV_FOLD)
    pbar = tqdm(total=len(caseid_list), desc='Processing caseid')
    j = 0
    for caseid, df_case in rawData.groupby('caseid'):
        j += 1
        if j > 100:
            break
        pbar.update(1)
        formattedData_case = pd.DataFrame()
        # process mean arterial value data
        # remove too low value (before the start of the measurement)
        df_case.mbp.mask(df_case.mbp < MIN_MPB_SGEMENT, inplace=True)
        # removing the nan values at the beginning and the ending
        case_valid_mask = ~df_case.mbp.isna()
        df_case = df_case[(np.cumsum(case_valid_mask) > 0) & (np.cumsum(case_valid_mask[::-1])[::-1] > 0)]

        # create label
        label = (np.convolve((df_case['mbp'] < MIN_VALUE_IOH).astype(int), np.ones(
            MIN_TIME_IOH), mode='valid') == MIN_TIME_IOH).astype(int)
        # complete the label to have the same length as the data
        label = np.concatenate([np.zeros(df_case.shape[0] - label.shape[0]), label])
        # add label to the data
        df_case.insert(0, 'label', label)
        # create time series
        time = np.arange(df_case.shape[0])*sampling_time
        df_case.insert(0, 'time', time)

        # create the segments
        segment_length = (observation_windows + leading_time + prediction_windows)//sampling_time
        segment_shift = window_shift // sampling_time

        for time_start in range(0, df_case.shape[0] - segment_length, segment_shift):
            segment = df_case.iloc[time_start:time_start + segment_length]
            number_of_raw_segment += 1
            if not validate_segment(segment, sampling_time, observation_windows, leading_time):
                continue

            number_of_selected_segement += 1

            # create the time series features
            segment_data = pd.DataFrame()
            for half_time in half_times:
                for signal in signal_name:
                    segment_data[f'{signal}_ema_{half_time}'] = [segment[signal].ewm(
                        halflife=half_time).mean().iloc[-1]]
                    segment_data[f'{signal}_var_{half_time}'] = [segment[signal].ewm(
                        halflife=half_time).std().iloc[-1]]
            # add the label of the segment
            segment_data['label'] = [(segment.label.iloc[(observation_windows + leading_time) //
                                     sampling_time:].sum() > 0).astype(int)]

            # add the segment to the data
            formattedData_case = pd.concat([formattedData_case, segment_data])

        # add the static features
        for static in static_data:
            formattedData_case[static] = df_case[static].iloc[0]

        # add the caseid and the group of the caseid
        formattedData_case['caseid'] = df_case['caseid'].iloc[0]
        # find in which group the caseid is
        for i, group in enumerate(caseid_list):
            if df_case['caseid'].iloc[0] in group:
                formattedData_case['cv_group'] = i
                break

        formattedData = pd.concat([formattedData, formattedData_case])
    pbar.close()
    print(f'Number of raw segment: {number_of_raw_segment}')
    print(f'Number of selected segment: {number_of_selected_segement}')
    return formattedData


dataframe = dataLoader(
    half_times=[10],
)
dataframe.to_csv('data/data_baseline.csv', index=False)
