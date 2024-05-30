import pandas as pd
import numpy as np

# Ratio of train samples (segments) in percent
TRAIN_RATIO = 0.7


def create_random_split(
    seed: int, label_stats: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(seed)
    case_ids = label_stats.index.to_numpy(copy=True)
    np.random.shuffle(case_ids)

    case_ids_shuffled = pd.Index(case_ids)

    n_train_case_ids = int(len(case_ids_shuffled) * TRAIN_RATIO)

    train_cases_ids = label_stats[:n_train_case_ids]
    test_cases_ids = label_stats[n_train_case_ids:]

    return train_cases_ids, test_cases_ids


def create_balanced_split(
    label_stats: pd.DataFrame,
    tolerance_segment_split: float,
    tolerance_label_split: float,
    n_max_iter_split: int,
):
    n_iter = 0
    best_cost = np.inf

    tolerance_segment_is_ok = False
    tolerance_label_is_ok = False
    while (
        (n_iter < n_max_iter_split)
        and not tolerance_segment_is_ok
        and not tolerance_label_is_ok
    ):
        n_iter += 1

        train_label_stats, test_label_stats = create_random_split(n_iter, label_stats)

        train_ratio_segment = compute_ratio_segment(train_label_stats, label_stats)
        train_ratio_label = compute_ratio_label(train_label_stats)
        test_ratio_label = compute_ratio_label(test_label_stats)

        cost_ratio_segment = (
            abs(train_ratio_segment - TRAIN_RATIO) / tolerance_segment_split
        )
        cost_ratio_label = (
            abs(train_ratio_label - test_ratio_label) / tolerance_label_split
        )
        cost = cost_ratio_segment + cost_ratio_label

        if cost < best_cost:
            best_cost = cost
            best_iter = n_iter

        tolerance_segment_is_ok = cost_ratio_segment < 1
        tolerance_label_is_ok = cost_ratio_label < 1

    return create_random_split(best_iter, label_stats)


def create_cv_balanced_split(
    label_stats: pd.DataFrame,
    general_ratio_segment: float,
    n_cv_splits: int,
    tolerance_segment_split: float,
    tolerance_label_split: float,
    n_max_iter_split: int,
) -> list[pd.DataFrame]:
    general_ratio_label = compute_ratio_label(label_stats)

    n_iter = 0
    best_cost = np.inf
    best_split = None

    tolerance_segment_is_ok = False
    tolerance_label_is_ok = False
    while (
        (n_iter < n_max_iter_split)
        and not tolerance_segment_is_ok
        and not tolerance_label_is_ok
    ):
        n_iter += 1

        np.random.seed(n_iter)
        # case_ids = label_stats.index.values.copy()
        case_ids = label_stats.index.to_numpy(copy=True)
        np.random.shuffle(case_ids)
        case_ids_split = np.array_split(case_ids, n_cv_splits)

        label_stats_splits = [label_stats.loc[index] for index in case_ids_split]

        ratio_segment_splits = [
            compute_ratio_segment(split_label_stats, label_stats)
            for split_label_stats in label_stats_splits
        ]
        ratio_label_splits = [
            compute_ratio_label(split_label_stats)
            for split_label_stats in label_stats_splits
        ]

        label_costs = [
            abs(ratio - general_ratio_label) / tolerance_segment_split
            for ratio in ratio_segment_splits
        ]
        segment_costs = [
            abs(ratio - general_ratio_segment) / tolerance_label_split
            for ratio in ratio_label_splits
        ]
        cost = sum(label_costs) + sum(segment_costs)

        if cost < best_cost:
            best_cost = cost
            best_split = n_iter

        tolerance_segment_is_ok = max(segment_costs) < 1
        tolerance_label_is_ok = max(label_costs) < 1

    np.random.seed(best_split)
    case_ids = label_stats.index.to_numpy(copy=True)
    np.random.shuffle(case_ids)
    case_ids_split = np.array_split(case_ids, n_cv_splits)

    return [label_stats.loc[index] for index in case_ids_split]


def compute_ratio_segment(
    split_label_stats: pd.DataFrame, label_stats: pd.DataFrame
) -> float:
    return split_label_stats.segment_count.sum() / label_stats.segment_count.sum()


def compute_ratio_label(label_stats: pd.DataFrame) -> float:
    return label_stats.label_count.sum() / label_stats.segment_count.sum()
