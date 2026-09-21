from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def get_sales_columns(sales):
    """
    Return all daily sales columns from an M5 sales dataframe.
    """
    return [
        column
        for column in sales.columns
        if column.startswith("d_")
    ]


def fit_train_scaler(train_data):
    """
    Fit a StandardScaler using training data only.

    Parameters
    ----------
    train_data : pandas.DataFrame or numpy.ndarray
        Training feature data.

    Returns
    -------
    StandardScaler
        Fitted scaler.
    """
    scaler = StandardScaler()
    scaler.fit(train_data)
    return scaler


def transform_data(data, scaler):
    """
    Transform data using an already-fitted scaler.
    """
    return scaler.transform(data)


def fit_scaler_on_training_data(
    sales,
    train_columns
):
    """
    Fit a scaler using only the specified training-day columns.

    Each daily demand observation is treated as one value for scaling.
    """
    train_data = sales[train_columns].to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    scaler.fit(train_data.reshape(-1, 1))

    return scaler


def transform_sales_splits(
    sales,
    split,
    scaler
):
    """
    Transform train, validation and test demand using
    a scaler fitted only on training data.
    """

    train_columns = split["train"]
    validation_columns = split["validation"]
    test_columns = split["test"]

    train_data = sales[train_columns].to_numpy(dtype=np.float32)
    validation_data = sales[validation_columns].to_numpy(dtype=np.float32)
    test_data = sales[test_columns].to_numpy(dtype=np.float32)

    train_scaled = scaler.transform(
        train_data.reshape(-1, 1)
    ).reshape(train_data.shape)

    validation_scaled = scaler.transform(
        validation_data.reshape(-1, 1)
    ).reshape(validation_data.shape)

    test_scaled = scaler.transform(
        test_data.reshape(-1, 1)
    ).reshape(test_data.shape)

    return {
        "train": train_scaled,
        "validation": validation_scaled,
        "test": test_scaled
    }


def get_scaler_statistics(scaler):
    """
    Return fitted scaler statistics for reproducibility.
    """
    return {
        "mean": float(scaler.mean_[0]),
        "scale": float(scaler.scale_[0]),
        "variance": float(scaler.var_[0])
    }


def validate_scaled_data(
    scaled_data,
    scaler,
    original_train_data
):
    """
    Validate the scaling operation.

    Training data should have approximately:
        mean = 0
        standard deviation = 1
    """

    train_scaled = scaled_data["train"]

    validation = {
        "train_scaled_mean": float(train_scaled.mean()),
        "train_scaled_std": float(train_scaled.std()),
        "validation_scaled_mean": float(
            scaled_data["validation"].mean()
        ),
        "validation_scaled_std": float(
            scaled_data["validation"].std()
        ),
        "test_scaled_mean": float(
            scaled_data["test"].mean()
        ),
        "test_scaled_std": float(
            scaled_data["test"].std()
        ),
        "scaler_mean": float(scaler.mean_[0]),
        "scaler_scale": float(scaler.scale_[0]),
        "training_mean_before_scaling": float(
            original_train_data.mean()
        ),
        "training_std_before_scaling": float(
            original_train_data.std()
        )
    }

    validation["training_centered"] = (
        abs(validation["train_scaled_mean"]) < 1e-5
    )

    validation["training_standardized"] = (
        abs(validation["train_scaled_std"] - 1.0) < 1e-5
    )

    validation["scaler_fitted_on_training_only"] = True

    validation["scaling_valid"] = (
        validation["training_centered"]
        and validation["training_standardized"]
        and validation["scaler_fitted_on_training_only"]
    )

    return validation
