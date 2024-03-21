import numpy as np

def test_label_caseid(data_builder, import_raw_results):
    case_data, _ = import_raw_results

    label, _ = data_builder._labelize(case_data)

    oracle_labels = np.zeros(len(case_data))
    oracle_labels[35:76] = 1
    oracle_labels[110:140] = 1

    assert (label == oracle_labels).all()