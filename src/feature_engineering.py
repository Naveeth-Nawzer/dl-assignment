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
