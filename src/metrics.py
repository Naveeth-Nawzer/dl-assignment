import numpy as np


def regression_metrics(y_true, y_pred):
    """
    Return MAE, RMSE, R², WAPE and sMAPE for demand in units.

    sMAPE counts a day as 0% error when both the actual and the forecast are 0.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    error = y_pred - y_true

    denominator = np.abs(y_true) + np.abs(y_pred)
    smape_terms = np.divide(
        2 * np.abs(error),
        denominator,
        out=np.zeros_like(error),
        where=denominator > 0,
    )

    total_variance = np.sum((y_true - y_true.mean()) ** 2)

    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "r2": float(1 - np.sum(error ** 2) / total_variance) if total_variance > 0 else float("nan"),
        "wape": float(np.sum(np.abs(error)) / np.sum(np.abs(y_true)) * 100) if y_true.sum() > 0 else float("nan"),
        "smape": float(np.mean(smape_terms) * 100),
    }


def horizon_metrics(y_true, y_pred):
    """
    Return one row of metrics per forecast day (1 to horizon).
    """
    return [
        {"horizon_day": day + 1, **regression_metrics(y_true[:, day], y_pred[:, day])}
        for day in range(y_true.shape[1])
    ]
