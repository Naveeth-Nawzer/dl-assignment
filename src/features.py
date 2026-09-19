from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing import (
    load_raw_data,
    clean_calendar,
    clean_prices,
    clean_sales,
    handle_calendar_missing_values,
)


FEATURE_NAMES = [
    "sales_scaled",
    "rolling_mean_7",
    "rolling_mean_28",
    "relative_price",
    "price_change_7",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_event",
    "snap",
    "cat_FOODS",
    "cat_HOBBIES",
    "cat_HOUSEHOLD",
]

CATEGORIES = ["FOODS", "HOBBIES", "HOUSEHOLD"]


def select_series(sales, sample_frac, seed):
    """
    Select a stratified sample of item-store series.

    Each store_id x cat_id group keeps the same fraction, so all stores and
    categories stay in the subset.
    """
    return (
        sales
        .groupby(["store_id", "cat_id"], group_keys=False)
        .sample(frac=sample_frac, random_state=seed)
        .sort_values(["store_id", "item_id"])
        .reset_index(drop=True)
    )


def build_price_matrix(sales, prices, calendar, n_days):
    """
    Return a [series, days] array of daily sell prices.

    Days before the first recorded price are NaN.
    """
    keys = sales[["store_id", "item_id"]].copy()
    keys["row"] = np.arange(len(keys))

    subset = prices.merge(keys, on=["store_id", "item_id"])

    weeks = np.sort(calendar["wm_yr_wk"].unique())
    week_index = {week: i for i, week in enumerate(weeks)}
    subset = subset[subset["wm_yr_wk"].isin(week_index)]

    weekly = np.full((len(sales), len(weeks)), np.nan, dtype=np.float32)
    weekly[
        subset["row"].to_numpy(),
        subset["wm_yr_wk"].map(week_index).to_numpy(),
    ] = subset["sell_price"].to_numpy()

    day_weeks = calendar["wm_yr_wk"].iloc[:n_days].map(week_index).to_numpy()

    return weekly[:, day_weeks]


def rolling_mean(values, window):
    """
    Return the trailing mean over `window` days, including the current day.
    """
    cumsum = np.cumsum(values, axis=1, dtype=np.float64)
    shifted = np.zeros_like(cumsum)
    shifted[:, window:] = cumsum[:, :-window]
    counts = np.minimum(np.arange(1, values.shape[1] + 1), window)

    return ((cumsum - shifted) / counts).astype(np.float32)


def build_features(sales, prices, calendar, cfg):
    """
    Build the model inputs for the selected series.

    Returns a dict with:
    - features: [series, days, n_features] float32
    - sales: [series, days] raw daily demand
    - scale: [series] training-period mean demand
    - first_valid: [series] first day index with a sell price
    - series: DataFrame with the series identifiers
    """
    n_days = cfg["n_days"]
    train_end = cfg["train_end"]

    sales_columns = [f"d_{i}" for i in range(1, n_days + 1)]
    demand = sales[sales_columns].to_numpy(dtype=np.float32)
    calendar = calendar.iloc[:n_days].reset_index(drop=True)

    price = build_price_matrix(sales, prices, calendar, n_days)
    has_price = ~np.isnan(price)
    first_valid = np.where(
        has_price.any(axis=1),
        has_price.argmax(axis=1),
        n_days,
    )

    # Items are not on sale before their first price, so these days do not count as zero demand.
    day_index = np.arange(n_days)
    active = day_index[None, :] >= first_valid[:, None]
    in_train = active & (day_index[None, :] < train_end)

    price = (
        pd.DataFrame(price)
        .ffill(axis=1)
        .bfill(axis=1)
        .to_numpy(dtype=np.float32)
    )

    # Scale statistics come from the training period only, to prevent leakage.
    train_days = np.maximum(in_train.sum(axis=1), 1)
    scale = np.maximum(
        (demand * in_train).sum(axis=1) / train_days,
        0.1,
    ).astype(np.float32)
    mean_price = (
        (price * in_train).sum(axis=1) / train_days
    ).astype(np.float32)
    mean_price = np.where(mean_price > 0, mean_price, price[:, -1])

    sales_scaled = demand / scale[:, None]

    price_change = np.zeros_like(price)
    previous = price[:, :-7]
    price_change[:, 7:] = np.divide(
        price[:, 7:],
        previous,
        out=np.ones_like(previous),
        where=previous > 0,
    ) - 1.0

    dates = pd.to_datetime(calendar["date"])
    weekday = dates.dt.weekday.to_numpy()
    month = dates.dt.month.to_numpy() - 1

    calendar_features = np.stack([
        np.sin(2 * np.pi * weekday / 7),
        np.cos(2 * np.pi * weekday / 7),
        np.sin(2 * np.pi * month / 12),
        np.cos(2 * np.pi * month / 12),
        (weekday >= 5).astype(float),
        (calendar["event_name_1"] != "None").astype(float).to_numpy(),
    ], axis=1).astype(np.float32)

    snap = np.stack([
        calendar[f"snap_{state}"].to_numpy(dtype=np.float32)
        for state in sales["state_id"]
    ])

    category = np.stack([
        (sales["cat_id"] == cat).to_numpy(dtype=np.float32)
        for cat in CATEGORIES
    ], axis=1)

    n_series = len(sales)
    features = np.concatenate([
        sales_scaled[..., None],
        rolling_mean(sales_scaled, 7)[..., None],
        rolling_mean(sales_scaled, 28)[..., None],
        (price / mean_price[:, None])[..., None],
        price_change[..., None],
        np.broadcast_to(calendar_features, (n_series, n_days, 6)),
        snap[..., None],
        np.broadcast_to(category[:, None, :], (n_series, n_days, 3)),
    ], axis=2).astype(np.float32)

    return {
        "features": features,
        "sales": demand,
        "scale": scale,
        "first_valid": first_valid,
        "series": sales[
            ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
        ].reset_index(drop=True),
    }


def filter_series(data, cfg):
    """
    Drop series with less than `min_train_history` active training days.
    """
    history = cfg["train_end"] - cfg["lookback"] * 2 - data["first_valid"]
    keep = history >= cfg["min_train_history"]

    return {
        "features": data["features"][keep],
        "sales": data["sales"][keep],
        "scale": data["scale"][keep],
        "first_valid": data["first_valid"][keep],
        "series": data["series"][keep].reset_index(drop=True),
    }


def prepare_dataset(data_path, cfg, cache_path=None):
    """
    Run cleaning, series selection and feature building, with an optional cache.

    The cache is an .npz file plus a CSV of the selected series.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            return load_dataset(cache_path)

    calendar, prices, sales = load_raw_data(data_path)
    calendar = handle_calendar_missing_values(clean_calendar(calendar))
    prices = clean_prices(prices)
    sales = clean_sales(sales)

    sales = select_series(sales, cfg["sample_frac"], cfg["sample_seed"])
    data = filter_series(build_features(sales, prices, calendar, cfg), cfg)

    if cache_path is not None:
        save_dataset(data, cache_path)

    return data


def save_dataset(data, cache_path):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        cache_path,
        features=data["features"],
        sales=data["sales"],
        scale=data["scale"],
        first_valid=data["first_valid"],
    )
    data["series"].to_csv(cache_path.with_suffix(".series.csv"), index=False)


def load_dataset(cache_path):
    cache_path = Path(cache_path)
    arrays = np.load(cache_path)

    return {
        "features": arrays["features"],
        "sales": arrays["sales"],
        "scale": arrays["scale"],
        "first_valid": arrays["first_valid"],
        "series": pd.read_csv(cache_path.with_suffix(".series.csv")),
    }
