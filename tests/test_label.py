import pandas as pd
import numpy as np
import pytest
from scripts.dataLoader import label_caseid

# Sample data for testing
mbp = np.ones(200)*100
mbp[35:76] = 64  # must be labeled as 1
mbp[110:140] = 55  # must be labeled as 1
mbp[120:125] = np.nan  # despite the presence of nan
mbp[150:180] = np.nan  # must be labeled as 0

df_sample = pd.DataFrame({'mbp': mbp})


def test_label_caseid():
    # Set the sampling time
    sampling_time = 2

    # Apply the function to the sample data
    labeled_df = label_caseid(df_sample.copy(), sampling_time)

    # plot the reuslts

    # Check if the 'label' column is added
    assert 'label' in labeled_df.columns

    # Check the correctness of the labels based on the provided sample data
    expected_labels = np.zeros(200)
    expected_labels[35:76] = 1
    expected_labels[110:140] = 1
    # import matplotlib.pyplot as plt
    # plt.plot(expected_labels)
    # plt.plot(labeled_df['label'])
    # plt.plot(mbp/100)
    # label_id = labeled_df.label.diff().clip(lower=0).cumsum().fillna(0)
    # label_id[labeled_df.label == 0] = np.nan
    # plt.plot(label_id)
    # plt.show()
    assert all(labeled_df['label'] == expected_labels)

    # Add more test cases if needed


if __name__ == '__main__':
    pytest.main(['-v', 'test_label.py'])
