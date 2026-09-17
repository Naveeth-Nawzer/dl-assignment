
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

    df = df.drop_duplicates().reset_index(drop=True)

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

    df = df.drop_duplicates().reset_index(drop=True)

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

    df = df.drop_duplicates().reset_index(drop=True)

    sales_columns = [
        col for col in df.columns
        if col.startswith("d_")
    ]

    df[sales_columns] = (
        df[sales_columns]
        .apply(pd.to_numeric, errors="coerce")
    )

    return df
