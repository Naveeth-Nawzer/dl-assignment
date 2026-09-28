import numpy as np
import torch
from torch.utils.data import Dataset


def generate_sequences(
    data,
    lookback=28
):
    """
    Generate supervised time-series sequences.

    Parameters
    ----------
    data : numpy.ndarray
        Array with shape:
        (number_of_series, number_of_time_steps)

    lookback : int
        Number of historical time steps used to predict
        the next time step.

    Returns
    -------
    X : numpy.ndarray
        Shape:
        (number_of_sequences, lookback, 1)

    y : numpy.ndarray
        Shape:
        (number_of_sequences,)

    target_indices : numpy.ndarray
        Original time-step index corresponding to each target.
    """

    if data.ndim != 2:
        raise ValueError(
            "data must have shape (number_of_series, number_of_time_steps)"
        )

    if lookback <= 0:
        raise ValueError("lookback must be greater than zero.")

    number_of_series, number_of_days = data.shape

    if number_of_days <= lookback:
        raise ValueError(
            "Number of time steps must be greater than lookback."
        )

    X = []
    y = []
    target_indices = []

    for series_index in range(number_of_series):

        series = data[series_index]

        for target_index in range(
            lookback,
            number_of_days
        ):
            X.append(
                series[
                    target_index - lookback:
                    target_index
                ]
            )

            y.append(
                series[target_index]
            )

            target_indices.append(target_index)

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.float32
    )

    target_indices = np.asarray(
        target_indices,
        dtype=np.int64
    )

    X = X.reshape(
        X.shape[0],
        X.shape[1],
        1
    )

    return X, y, target_indices


def generate_sequences_with_global_indices(
    data,
    start_index,
    end_index,
    lookback=28
):
    """
    Generate sequences whose target indices fall within
    [start_index, end_index).

    Historical context may come from earlier time steps.

    Parameters
    ----------
    data : numpy.ndarray
        Complete scaled time-series data.

    start_index : int
        First allowed target index.

    end_index : int
        First index after the allowed target period.

    lookback : int
        Historical lookback window.

    Returns
    -------
    X : numpy.ndarray
    y : numpy.ndarray
    target_indices : numpy.ndarray
    """

    if data.ndim != 2:
        raise ValueError(
            "data must have shape (number_of_series, number_of_time_steps)"
        )

    if start_index < lookback:
        raise ValueError(
            "start_index must be at least equal to lookback."
        )

    if end_index <= start_index:
        raise ValueError(
            "end_index must be greater than start_index."
        )

    X = []
    y = []
    target_indices = []

    number_of_series = data.shape[0]

    for series_index in range(number_of_series):

        series = data[series_index]

        for target_index in range(
            start_index,
            end_index
        ):

            X.append(
                series[
                    target_index - lookback:
                    target_index
                ]
            )

            y.append(
                series[target_index]
            )

            target_indices.append(target_index)

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.float32
    )

    target_indices = np.asarray(
        target_indices,
        dtype=np.int64
    )

    X = X.reshape(
        X.shape[0],
        X.shape[1],
        1
    )

    return X, y, target_indices


def convert_to_tensors(X, y):
    """
    Convert NumPy arrays to PyTorch tensors.
    """

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32
    )

    y_tensor = torch.tensor(
        y,
        dtype=torch.float32
    )

    return X_tensor, y_tensor


def validate_sequence_shapes(
    X,
    y,
    target_indices,
    lookback
):
    """
    Validate generated sequence dimensions.
    """

    validation = {
        "number_of_sequences": len(X),
        "lookback": lookback,
        "number_of_features": X.shape[2],
        "X_dimensions": X.ndim,
        "y_dimensions": y.ndim,
        "target_indices_dimensions": target_indices.ndim,
        "X_y_count_match": len(X) == len(y),
        "X_target_count_match": len(X) == len(target_indices),
        "correct_lookback": X.shape[1] == lookback,
        "single_feature": X.shape[2] == 1
    }

    validation["sequence_shape_valid"] = (
        validation["X_dimensions"] == 3
        and validation["y_dimensions"] == 1
        and validation["target_indices_dimensions"] == 1
        and validation["X_y_count_match"]
        and validation["X_target_count_match"]
        and validation["correct_lookback"]
        and validation["single_feature"]
    )

    return validation


def check_sequence_leakage(
    X,
    y,
    target_indices
):
    """
    Check that every sequence contains observations strictly
    before its corresponding target.
    """

    leakage_count = 0

    for sequence_index in range(len(X)):

        target_index = target_indices[sequence_index]

        if target_index <= 0:
            leakage_count += 1
            continue

    return {
        "leakage_count": leakage_count,
        "leakage_free": leakage_count == 0
    }

class RetailSequenceDataset(Dataset):
    """
    Memory-efficient PyTorch Dataset for retail time-series forecasting.

    Each sample represents:

        previous `lookback` observations -> next-day target

    Parameters
    ----------
    data : numpy.ndarray
        Shape:
        (number_of_series, number_of_days)

    start_target : int
        First global target index.

    end_target : int
        Exclusive upper bound for target index.

    lookback : int
        Number of historical observations.
    """

    def __init__(
        self,
        data,
        start_target,
        end_target,
        lookback=28
    ):

        if data.ndim != 2:
            raise ValueError(
                "data must have shape "
                "(number_of_series, number_of_days)"
            )

        if start_target < lookback:
            raise ValueError(
                "start_target must be at least lookback."
            )

        if end_target <= start_target:
            raise ValueError(
                "end_target must be greater than start_target."
            )

        self.data = data
        self.start_target = start_target
        self.end_target = end_target
        self.lookback = lookback

        self.number_of_series = data.shape[0]
        self.number_of_targets = (
            end_target - start_target
        )

    def __len__(self):
        return (
            self.number_of_series *
            self.number_of_targets
        )

    def __getitem__(self, index):

        target_period = (
            index %
            self.number_of_targets
        )

        series_index = (
            index //
            self.number_of_targets
        )

        target_index = (
            self.start_target +
            target_period
        )

        start_index = (
            target_index -
            self.lookback
        )

        X = self.data[
            series_index,
            start_index:target_index
        ]

        y = self.data[
            series_index,
            target_index
        ]

        X = torch.tensor(
            X,
            dtype=torch.float32
        ).unsqueeze(-1)

        y = torch.tensor(
            y,
            dtype=torch.float32
        )

        return X, y
