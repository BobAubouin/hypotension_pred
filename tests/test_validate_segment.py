import pandas as pd
import numpy as np
import pytest
from copy import deepcopy
from scripts.dataLoader import validate_segment
import matplotlib.pyplot as plt

# Sample data for testing
signal_names = ['mbp', 'sbp', 'dbp', 'hr', 'rr', 'spo2', 'etco2', 'mac', 'pp_ct', 'bis']
data = {
    'mbp': np.ones(100)*100,  # np.random.uniform(60, 120, 100),
    'sbp': np.random.uniform(90, 180, 100),
    'dbp': np.random.uniform(60, 90, 100),
    'hr': np.random.uniform(60, 100, 100),
    'rr': np.random.uniform(10, 20, 100),
    'spo2': np.random.uniform(95, 100, 100),
    'etco2': np.random.uniform(30, 40, 100),
    'mac': np.random.uniform(0, 2, 100),
    'pp_ct': np.random.uniform(0, 5, 100),
    'bis': np.random.uniform(40, 60, 100),
    'label': np.zeros(100)
}

df_valid_segment = pd.DataFrame(data)

data_invalid_mbp = deepcopy(data)
data_invalid_mbp['mbp'][20:30] = 35  # Invalid: mbp < 40
df_invalid_mbp = pd.DataFrame(data_invalid_mbp)

data_invalid_label = deepcopy(data)
data_invalid_label['label'][2:4] = 1  # Invalid: label in observation window
df_invalid_label = pd.DataFrame(data_invalid_label)

data_invalid_previous_label = deepcopy(data)
data_invalid_previous_label['label'][80:90] = 1  # Invalid: label in previous segment
df_invalid_previous_label = pd.DataFrame(data_invalid_previous_label)

data_invalid_nan = deepcopy(data)
data_invalid_nan['spo2'][10:31] = np.nan  # Invalid: too much nan in the segment
df_invalid_nan = pd.DataFrame(data_invalid_nan)

data_valid_previous_label = deepcopy(data)
df_valid_previous_label = pd.DataFrame(data_valid_previous_label)


def test_validate_segment():
    # Set parameters
    sampling_time = 2
    observation_windows = 5
    leading_time = 2

    # Test valid segment
    assert validate_segment(df_valid_segment, df_valid_previous_label, sampling_time, observation_windows, leading_time)

    # Test invalid segment: mbp < 40
    assert not validate_segment(df_invalid_mbp, df_valid_previous_label,
                                sampling_time, observation_windows, leading_time)

    # Test invalid segment: label in observation window
    assert not validate_segment(df_invalid_label, df_valid_previous_label,
                                sampling_time, observation_windows, leading_time)

    # Test invalid segment: label in previous segment
    assert not validate_segment(df_valid_segment, df_invalid_previous_label,
                                sampling_time, observation_windows, leading_time)

    # Test invalid segment: too much nan in the segment
    assert not validate_segment(df_invalid_nan, df_valid_previous_label,
                                sampling_time, observation_windows, leading_time)


if __name__ == '__main__':
    pytest.main(['-v', 'test_validate_segment.py'])
