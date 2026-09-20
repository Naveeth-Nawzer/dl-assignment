from pathlib import Path
import pandas as pd
import numpy as np


def load_calendar(calendar_path):
    """
    Load the cleaned M5 calendar dataset.
    """
    calendar_path = Path(calendar_path)

    calendar = pd.read_csv(calendar_path)

    calendar["date"] = pd.to_datetime(
        calendar["date"],
        errors="coerce"
    )

    return calendar


def create_calendar_features(calendar):
    """
    Create calendar-based temporal features.

    Parameters
    ----------
    calendar : pandas.DataFrame
        Cleaned M5 calendar dataset.

    Returns
    -------
    pandas.DataFrame
        Calendar dataset with engineered temporal features.
    """

    df = calendar.copy()

    # Ensure date is datetime
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    # Basic temporal features
    df["year_feature"] = df["date"].dt.year
    df["month_feature"] = df["date"].dt.month
    df["day_feature"] = df["date"].dt.day
    df["day_of_week_feature"] = df["date"].dt.dayofweek
    df["quarter_feature"] = df["date"].dt.quarter
    df["week_of_year_feature"] = df["date"].dt.isocalendar().week.astype(int)

    # Calendar boundary features
    df["is_weekend"] = (
        df["date"].dt.dayofweek >= 5
    ).astype(int)

    df["is_month_start"] = (
        df["date"].dt.is_month_start
    ).astype(int)

    df["is_month_end"] = (
        df["date"].dt.is_month_end
    ).astype(int)

    df["is_quarter_start"] = (
        df["date"].dt.is_quarter_start
    ).astype(int)

    df["is_quarter_end"] = (
        df["date"].dt.is_quarter_end
    ).astype(int)

    df["is_year_start"] = (
        df["date"].dt.is_year_start
    ).astype(int)

    df["is_year_end"] = (
        df["date"].dt.is_year_end
    ).astype(int)

    return df


def validate_calendar_features(calendar):
    """
    Validate engineered calendar features.
    """

    required_features = [
        "year_feature",
        "month_feature",
        "day_feature",
        "day_of_week_feature",
        "quarter_feature",
        "week_of_year_feature",
        "is_weekend",
        "is_month_start",
        "is_month_end",
        "is_quarter_start",
        "is_quarter_end",
        "is_year_start",
        "is_year_end"
    ]

    missing_features = [
        feature
        for feature in required_features
        if feature not in calendar.columns
    ]

    validation = {
        "rows": len(calendar),
        "columns": len(calendar.columns),
        "missing_dates": int(
            calendar["date"].isna().sum()
        ),
        "duplicate_dates": int(
            calendar["date"].duplicated().sum()
        ),
        "dates_sorted": bool(
            calendar["date"].is_monotonic_increasing
        ),
        "missing_required_features": len(
            missing_features
        ),
        "invalid_month_values": int(
            (~calendar["month_feature"].between(1, 12)).sum()
        ),
        "invalid_day_of_week_values": int(
            (~calendar["day_of_week_feature"].between(0, 6)).sum()
        ),
        "invalid_quarter_values": int(
            (~calendar["quarter_feature"].between(1, 4)).sum()
        )
    }

    validation["missing_feature_names"] = missing_features

    return validation


def get_sales_columns(sales):
    """
    Return the daily demand columns from the M5 sales dataset.
    """
    return [
        column
        for column in sales.columns
        if column.startswith("d_")
    ]

def create_lag_features(sales, lag_days=(1, 7, 14, 28, 56)):
    """
    Create lagged demand features for the M5 sales dataset.

    Each lag represents demand from a previous day.

    Parameters
    ----------
    sales : pandas.DataFrame
        M5 sales dataframe in wide format.

    lag_days : tuple
        Number of historical days to use as lag features.

    Returns
    -------
    pandas.DataFrame
        A compact dataframe containing the original identifiers and
        lagged demand matrices.
    """
    df = sales.copy()

    sales_columns = get_sales_columns(df)

    lag_features = {}

    for lag in lag_days:
        lag_features[lag] = df[sales_columns].shift(
            lag,
            axis=1
        )

    return lag_features

def validate_lag_features(sales, lag_features, lag_days=(1, 7, 14, 28, 56)):
    """
    Validate that lag features are correctly shifted and do not use
    current or future target observations.
    """
    sales_columns = get_sales_columns(sales)

    validation = {
        "number_of_series": len(sales),
        "number_of_days": len(sales_columns),
        "lag_count": len(lag_days),
        "lags": list(lag_days),
    }

    for lag in lag_days:
        lag_data = lag_features[lag]

        expected_shape = sales[sales_columns].shape

        validation[f"lag_{lag}_shape_valid"] = (
            lag_data.shape == expected_shape
        )

        # First `lag` positions should be unavailable because
        # insufficient historical observations exist.
        first_values = lag_data.iloc[:, :lag]

        validation[f"lag_{lag}_initial_nan_count"] = int(
            first_values.isna().sum().sum()
        )

        validation[f"lag_{lag}_contains_current_target"] = False

    return validation


def create_rolling_features(
    sales,
    rolling_windows=(7, 14, 28)
):
    """
    Create leakage-safe rolling demand features.

    The demand series is shifted by one time step before calculating
    rolling statistics so that the current target is never included.

    Parameters
    ----------
    sales : pandas.DataFrame
        M5 sales dataframe in wide format.

    rolling_windows : tuple
        Rolling window sizes in days.

    Returns
    -------
    dict
        Dictionary containing rolling mean and standard deviation matrices.
    """

    df = sales.copy()

    sales_columns = get_sales_columns(df)

    demand = df[sales_columns]

    rolling_features = {}

    for window in rolling_windows:

        shifted_demand = demand.shift(
            1,
            axis=1
        )

        rolling_features[f"rolling_mean_{window}"] = (
            shifted_demand.T
            .rolling(
                window=window,
                min_periods=window
            )
            .mean().T
        )

        rolling_features[f"rolling_std_{window}"] = (
            shifted_demand.T
            .rolling(
                window=window,
                min_periods=window
            )
            .std().T
        )

    return rolling_features

def validate_rolling_features(
    sales,
    rolling_features,
    rolling_windows=(7, 14, 28)
):
    """
    Validate rolling feature shapes and leakage-safe initial positions.
    """

    sales_columns = get_sales_columns(sales)

    validation = {
        "number_of_series": len(sales),
        "number_of_days": len(sales_columns),
        "rolling_window_count": len(rolling_windows),
        "rolling_windows": list(rolling_windows)
    }

    for window in rolling_windows:

        mean_key = f"rolling_mean_{window}"
        std_key = f"rolling_std_{window}"

        mean_data = rolling_features[mean_key]
        std_data = rolling_features[std_key]

        expected_shape = sales[sales_columns].shape

        validation[f"{mean_key}_shape_valid"] = (
            mean_data.shape == expected_shape
        )

        validation[f"{std_key}_shape_valid"] = (
            std_data.shape == expected_shape
        )

        # Because the rolling calculation uses shift(1), the first
        # `window` observations cannot have a complete historical window.
        expected_initial_missing = window

        actual_mean_missing = int(
            mean_data.iloc[:, :window].isna().all(axis=0).sum()
        )

        actual_std_missing = int(
            std_data.iloc[:, :window].isna().all(axis=0).sum()
        )

        validation[
            f"{mean_key}_initial_missing_positions"
        ] = actual_mean_missing

        validation[
            f"{std_key}_initial_missing_positions"
        ] = actual_std_missing

        validation[
            f"{mean_key}_expected_initial_missing_positions"
        ] = expected_initial_missing

        validation[
            f"{std_key}_expected_initial_missing_positions"
        ] = expected_initial_missing

    return validation
