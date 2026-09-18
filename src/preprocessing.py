from pathlib import Path
import pandas as pd
import numpy as np


def load_raw_data(data_path):
    """
    Load the M5 raw calendar, price, and sales datasets.
    """
    data_path = Path(data_path)

    calendar = pd.read_csv(
        data_path / "calendar.csv"
    )

    prices = pd.read_csv(
        data_path / "sell_prices.csv"
    )

    sales = pd.read_csv(
        data_path / "sales_train_validation.csv"
    )

    return calendar, prices, sales


def clean_calendar(calendar):
    """
    Clean and validate calendar data.
    """
    df = calendar.copy()

    df = (
        df
        .drop_duplicates()
        .reset_index(drop=True)
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df = (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )

    return df


def clean_prices(prices):
    """
    Clean and validate selling-price data.
    """
    df = prices.copy()

    df = (
        df
        .drop_duplicates()
        .reset_index(drop=True)
    )

    df["sell_price"] = pd.to_numeric(
        df["sell_price"],
        errors="coerce"
    )

    df = (
        df
        .sort_values(
            ["store_id", "item_id", "wm_yr_wk"]
        )
        .reset_index(drop=True)
    )

    return df


def clean_sales(sales):
    """
    Clean and validate sales data.
    """
    df = sales.copy()

    df = (
        df
        .drop_duplicates()
        .reset_index(drop=True)
    )

    sales_columns = [
        col for col in df.columns
        if col.startswith("d_")
    ]

    df[sales_columns] = (
        df[sales_columns]
        .apply(pd.to_numeric, errors="coerce")
    )

    return df


def handle_calendar_missing_values(calendar):
    """
    Handle meaningful missing values in calendar
    event-related categorical columns.

    Missing event values represent days without
    recorded events and are therefore replaced
    with the explicit category 'None'.
    """
    df = calendar.copy()

    event_columns = [
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2"
    ]

    for column in event_columns:
        if column in df.columns:
            df[column] = (
                df[column]
                .fillna("None")
            )

    return df


def missing_value_summary(df):
    """
    Return missing-value counts and percentages.
    """
    summary = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percentage": (
            df.isna().mean() * 100
        )
    })

    return summary[
        summary["missing_count"] > 0
    ].sort_values(
        "missing_percentage",
        ascending=False
    )


def demand_statistics(sales, sales_columns):
    """
    Calculate overall demand distribution statistics.
    """
    values = sales[sales_columns].to_numpy().flatten()

    return {
        "count": len(values),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "p99_9": float(np.percentile(values, 99.9)),
    }


def series_demand_statistics(sales, sales_columns):
    """
    Calculate demand statistics for each item-store series.
    """
    numeric_sales = sales[sales_columns]

    stats = pd.DataFrame({
        "item_id": sales["item_id"],
        "store_id": sales["store_id"],
        "dept_id": sales["dept_id"],
        "cat_id": sales["cat_id"],
        "state_id": sales["state_id"],
        "mean_demand": numeric_sales.mean(axis=1),
        "std_demand": numeric_sales.std(axis=1),
        "total_demand": numeric_sales.sum(axis=1),
        "zero_days": (numeric_sales == 0).sum(axis=1)
    })

    stats["zero_percentage"] = (
        stats["zero_days"] /
        len(sales_columns)
    ) * 100

    stats["cv"] = (
        stats["std_demand"] /
        stats["mean_demand"].replace(0, np.nan)
    )

    return stats
