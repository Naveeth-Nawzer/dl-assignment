from pathlib import Path
import pandas as pd


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
