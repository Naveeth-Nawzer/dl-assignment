import numpy as np

from metrics import regression_metrics


def window_arrays(sales, index, cfg):
    """
    Return the raw input history and targets for each window, in units.
    """
    lookback = cfg["lookback"]
    horizon = cfg["horizon"]

    series = index[:, 0:1]
    t = index[:, 1:2]

    history = sales[series, t + np.arange(-lookback, 0)]
    targets = sales[series, t + np.arange(horizon)]

    return history, targets


def naive_forecast(history, horizon):
    """Repeat the last observed day."""
    return np.repeat(history[:, -1:], horizon, axis=1)


def seasonal_naive_forecast(history, horizon):
    """Repeat the same weekday from the last observed week."""
    last_week = history[:, -7:]
    repeats = -(-horizon // 7)
    return np.tile(last_week, (1, repeats))[:, :horizon]


def moving_average_forecast(history, horizon):
    """Repeat the mean of the full input window."""
    return np.repeat(history.mean(axis=1, keepdims=True), horizon, axis=1)


BASELINES = {
    "naive": naive_forecast,
    "seasonal_naive": seasonal_naive_forecast,
    "moving_average_28": moving_average_forecast,
}


def evaluate_baselines(sales, index, cfg):
    """
    Return metrics for each baseline on the given windows.
    """
    history, targets = window_arrays(sales, index, cfg)

    return {
        name: regression_metrics(targets, forecast(history, cfg["horizon"]))
        for name, forecast in BASELINES.items()
    }
