import pandas as pd

def get_sales_columns(sales):
    """
    Return all daily demand columns from the M5 sales dataframe.
    """
    return [
        column
        for column in sales.columns
        if column.startswith("d_")
    ]

def create_chronological_split(
    sales,
    train_ratio=0.80,
    validation_ratio=0.10
):
    """
    Create a chronological train/validation/test split.

    The split is performed across the temporal dimension of the
    wide-format M5 sales dataframe.

    Parameters
    ----------
    sales : pandas.DataFrame
        M5 sales dataframe.

    train_ratio : float
        Proportion assigned to training.

    validation_ratio : float
        Proportion assigned to validation.

    Returns
    -------
    dict
        Train, validation, and test day columns and their sizes.
    """

    if train_ratio <= 0:
        raise ValueError(
            "train_ratio must be greater than zero."
        )

    if validation_ratio <= 0:
        raise ValueError(
            "validation_ratio must be greater than zero."
        )

    if train_ratio + validation_ratio >= 1:
        raise ValueError(
            "Train and validation ratios must leave data for testing."
        )

    sales_columns = get_sales_columns(sales)

    n_days = len(sales_columns)

    train_days = int(
        n_days * train_ratio
    )

    validation_days = int(
        n_days * validation_ratio
    )

    test_days = (
        n_days
        - train_days
        - validation_days
    )

    train_columns = sales_columns[
        :train_days
    ]

    validation_columns = sales_columns[
        train_days:
        train_days + validation_days
    ]

    test_columns = sales_columns[
        train_days + validation_days:
    ]

    return {
        "train": train_columns,
        "validation": validation_columns,
        "test": test_columns,
        "train_days": train_days,
        "validation_days": validation_days,
        "test_days": test_days
    }

def validate_chronological_split(split):
    """
    Validate that train, validation, and test periods are
    mutually exclusive and chronologically ordered.
    """

    train = split["train"]
    validation = split["validation"]
    test = split["test"]

    train_validation_overlap = len(
        set(train).intersection(validation)
    )

    train_test_overlap = len(
        set(train).intersection(test)
    )

    validation_test_overlap = len(
        set(validation).intersection(test)
    )

    train_before_validation = (
        train[-1] < validation[0]
    )

    validation_before_test = (
        validation[-1] < test[0]
    )

    results = {
        "train_days": len(train),
        "validation_days": len(validation),
        "test_days": len(test),
        "train_validation_overlap": (
            train_validation_overlap
        ),
        "train_test_overlap": (
            train_test_overlap
        ),
        "validation_test_overlap": (
            validation_test_overlap
        ),
        "train_before_validation": (
            train_before_validation
        ),
        "validation_before_test": (
            validation_before_test
        )
    }

    results["split_valid"] = (
        train_validation_overlap == 0
        and train_test_overlap == 0
        and validation_test_overlap == 0
        and train_before_validation
        and validation_before_test
    )

    return results

def create_split_summary(
    split,
    date_lookup
):
    """
    Create a human-readable chronological split summary.
    """

    return pd.DataFrame({
        "split": [
            "train",
            "validation",
            "test"
        ],
        "start_day": [
            split["train"][0],
            split["validation"][0],
            split["test"][0]
        ],
        "end_day": [
            split["train"][-1],
            split["validation"][-1],
            split["test"][-1]
        ],
        "start_date": [
            date_lookup[split["train"][0]],
            date_lookup[split["validation"][0]],
            date_lookup[split["test"][0]]
        ],
        "end_date": [
            date_lookup[split["train"][-1]],
            date_lookup[split["validation"][-1]],
            date_lookup[split["test"][-1]]
        ],
        "number_of_days": [
            len(split["train"]),
            len(split["validation"]),
            len(split["test"])
        ]
    })
