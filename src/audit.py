from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Column definitions
# -------------------------------------------------------------------

ID_COLUMNS = [
    "Test_ID",
]

NUMERIC_COLUMNS = [
    "Applied_Voltage_kV",
    "Load_Current_A",
    "Ambient_Temperature_C",
    "Test_Duration_min",
    "Sensor_S1",
    "Sensor_S2",
    "Sensor_S3",
    "Sensor_S4",
]

TARGET_COLUMN = "Reference_Parameter"

LABEL_COLUMN = "Validity_Label"

SENSOR_COLUMNS = [
    "Sensor_S1",
    "Sensor_S2",
    "Sensor_S3",
    "Sensor_S4",
]

# These are the measurements that describe the test condition and sensors.
# We deliberately exclude Test_ID, Reference_Parameter and Validity_Label
# when analysing duplicate measurement combinations.
MEASUREMENT_COLUMNS = [
    "Applied_Voltage_kV",
    "Load_Current_A",
    "Ambient_Temperature_C",
    "Test_Duration_min",
    "Sensor_S1",
    "Sensor_S2",
    "Sensor_S3",
    "Sensor_S4",
]


# -------------------------------------------------------------------
# General profile
# -------------------------------------------------------------------

def create_basic_profile(
    df: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Generate a column-level data-quality profile.

    The function does not modify the dataframe.
    """

    rows = []

    for column in df.columns:

        series = df[column]

        rows.append(
            {
                "dataset": dataset_name,
                "column": column,
                "dtype": str(series.dtype),
                "row_count": len(series),
                "missing_count": int(series.isna().sum()),
                "missing_percentage": round(
                    series.isna().mean() * 100,
                    4,
                ),
                "unique_count": int(
                    series.nunique(dropna=False)
                ),
                "duplicate_value_count": int(
                    series.duplicated().sum()
                ),
            }
        )

    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# Numerical summary
# -------------------------------------------------------------------

def create_numeric_summary(
    df: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Generate numerical descriptive statistics.
    """

    available_columns = [
        column
        for column in NUMERIC_COLUMNS
        if column in df.columns
    ]

    if not available_columns:
        return pd.DataFrame()

    summary = (
        df[available_columns]
        .describe()
        .T
        .reset_index()
        .rename(columns={"index": "column"})
    )

    summary.insert(
        0,
        "dataset",
        dataset_name,
    )

    return summary


# -------------------------------------------------------------------
# Higher-order statistics
# -------------------------------------------------------------------

def create_distribution_statistics(
    df: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Generate additional statistics useful for understanding
    skewness and tail behaviour.
    """

    rows = []

    for column in NUMERIC_COLUMNS:

        if column not in df.columns:
            continue

        series = df[column]

        rows.append(
            {
                "dataset": dataset_name,
                "column": column,
                "mean": series.mean(),
                "median": series.median(),
                "std": series.std(),
                "variance": series.var(),
                "skewness": series.skew(),
                "kurtosis": series.kurtosis(),
                "min": series.min(),
                "q01": series.quantile(0.01),
                "q05": series.quantile(0.05),
                "q25": series.quantile(0.25),
                "q50": series.quantile(0.50),
                "q75": series.quantile(0.75),
                "q95": series.quantile(0.95),
                "q99": series.quantile(0.99),
                "max": series.max(),
            }
        )

    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# Duplicate analysis
# -------------------------------------------------------------------

def create_duplicate_summary(
    df: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Summarize different types of duplication.
    """

    duplicate_test_id_rows = np.nan

    if "Test_ID" in df.columns:
        duplicate_test_id_rows = int(
            df["Test_ID"]
            .duplicated(keep=False)
            .sum()
        )

    duplicate_complete_rows = int(
        df.duplicated(keep=False).sum()
    )

    available_measurement_columns = [
        column
        for column in MEASUREMENT_COLUMNS
        if column in df.columns
    ]

    if available_measurement_columns:

        duplicate_measurement_rows = int(
            df[
                available_measurement_columns
            ]
            .duplicated(keep=False)
            .sum()
        )

    else:

        duplicate_measurement_rows = np.nan

    return pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "duplicate_test_id_rows": duplicate_test_id_rows,
                "duplicate_complete_rows": duplicate_complete_rows,
                "duplicate_measurement_rows": duplicate_measurement_rows,
            }
        ]
    )


def get_duplicate_measurement_records(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return all rows participating in duplicate measurement
    combinations.

    Test_ID, target and label are intentionally excluded from
    the duplicate key.
    """

    available_columns = [
        column
        for column in MEASUREMENT_COLUMNS
        if column in df.columns
    ]

    if not available_columns:
        return pd.DataFrame()

    mask = (
        df[
            available_columns
        ]
        .duplicated(
            keep=False
        )
    )

    return (
        df.loc[mask]
        .sort_values(
            available_columns
        )
        .reset_index(drop=True)
    )


# -------------------------------------------------------------------
# Missingness analysis
# -------------------------------------------------------------------

def create_missingness_pattern_summary(
    df: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Summarize combinations of missing sensor values.

    Example:
        1000 -> no sensor missing
        1001 -> only Sensor_S4 missing
    """

    available_sensor_columns = [
        column
        for column in SENSOR_COLUMNS
        if column in df.columns
    ]

    if not available_sensor_columns:
        return pd.DataFrame()

    pattern = (
        df[
            available_sensor_columns
        ]
        .isna()
        .astype(int)
        .astype(str)
        .agg(
            "".join,
            axis=1,
        )
    )

    result = (
        pattern
        .value_counts()
        .rename_axis(
            "missing_pattern"
        )
        .reset_index(
            name="row_count"
        )
    )

    result.insert(
        0,
        "dataset",
        dataset_name,
    )

    result["percentage"] = (
        result["row_count"]
        / len(df)
        * 100
    )

    return result


def create_missingness_by_column(
    df: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Return missing-value statistics for each column.
    """

    rows = []

    for column in df.columns:

        missing_count = int(
            df[column].isna().sum()
        )

        rows.append(
            {
                "dataset": dataset_name,
                "column": column,
                "missing_count": missing_count,
                "missing_percentage": round(
                    missing_count / len(df) * 100,
                    4,
                ),
            }
        )

    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# Valid / Invalid analysis
# -------------------------------------------------------------------

def create_validity_summary(
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Count historical Valid/Invalid labels.
    """

    if LABEL_COLUMN not in train_df.columns:
        return pd.DataFrame()

    result = (
        train_df[LABEL_COLUMN]
        .value_counts(
            dropna=False
        )
        .rename_axis(LABEL_COLUMN)
        .reset_index(
            name="row_count"
        )
    )

    result["percentage"] = (
        result["row_count"]
        / len(train_df)
        * 100
    )

    return result


def create_missingness_by_validity(
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare sensor missingness between Valid and Invalid
    historical records.
    """

    if LABEL_COLUMN not in train_df.columns:
        return pd.DataFrame()

    available_sensors = [
        column
        for column in SENSOR_COLUMNS
        if column in train_df.columns
    ]

    rows = []

    for label, group in train_df.groupby(
        LABEL_COLUMN,
        dropna=False,
    ):

        for sensor in available_sensors:

            missing_count = int(
                group[sensor]
                .isna()
                .sum()
            )

            rows.append(
                {
                    LABEL_COLUMN: label,
                    "sensor": sensor,
                    "row_count": len(group),
                    "missing_count": missing_count,
                    "missing_percentage": round(
                        missing_count
                        / len(group)
                        * 100,
                        4,
                    ),
                }
            )

    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# Train vs Test range comparison
# -------------------------------------------------------------------

def create_train_test_range_comparison(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare min/max ranges in historical training data against test.
    Also calculate how many test values fall outside the historical
    training range.
    """

    rows = []

    for column in NUMERIC_COLUMNS:

        if (
            column not in train_df.columns
            or column not in test_df.columns
        ):
            continue

        train_series = train_df[column]
        test_series = test_df[column]

        train_min = train_series.min()
        train_max = train_series.max()

        valid_test = test_series.dropna()

        if len(valid_test) == 0:
            outside_count = 0
            outside_percentage = 0.0

        else:

            outside_mask = (
                (valid_test < train_min)
                | (valid_test > train_max)
            )

            outside_count = int(
                outside_mask.sum()
            )

            outside_percentage = (
                outside_count
                / len(valid_test)
                * 100
            )

        rows.append(
            {
                "column": column,
                "train_min": train_min,
                "train_max": train_max,
                "test_min": test_series.min(),
                "test_max": test_series.max(),
                "test_outside_train_range_count": outside_count,
                "test_outside_train_range_percentage": round(
                    outside_percentage,
                    4,
                ),
            }
        )

    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# Save all outputs
# -------------------------------------------------------------------

def save_audit_outputs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """
    Generate all audit tables and save them as CSV files.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Basic profiles
    create_basic_profile(
        train_df,
        "training",
    ).to_csv(
        output_dir / "training_profile.csv",
        index=False,
    )

    create_basic_profile(
        test_df,
        "test",
    ).to_csv(
        output_dir / "test_profile.csv",
        index=False,
    )

    # Numerical summaries
    create_numeric_summary(
        train_df,
        "training",
    ).to_csv(
        output_dir / "training_numeric_summary.csv",
        index=False,
    )

    create_numeric_summary(
        test_df,
        "test",
    ).to_csv(
        output_dir / "test_numeric_summary.csv",
        index=False,
    )

    # Distribution statistics
    create_distribution_statistics(
        train_df,
        "training",
    ).to_csv(
        output_dir / "training_distribution_statistics.csv",
        index=False,
    )

    create_distribution_statistics(
        test_df,
        "test",
    ).to_csv(
        output_dir / "test_distribution_statistics.csv",
        index=False,
    )

    # Duplicates
    create_duplicate_summary(
        train_df,
        "training",
    ).to_csv(
        output_dir / "training_duplicates.csv",
        index=False,
    )

    create_duplicate_summary(
        test_df,
        "test",
    ).to_csv(
        output_dir / "test_duplicates.csv",
        index=False,
    )

    duplicate_records = (
        get_duplicate_measurement_records(
            train_df
        )
    )

    duplicate_records.to_csv(
        output_dir / "training_duplicate_measurement_records.csv",
        index=False,
    )

    duplicate_test_records = (
        get_duplicate_measurement_records(
            test_df
        )
    )

    duplicate_test_records.to_csv(
        output_dir / "test_duplicate_measurement_records.csv",
        index=False,
    )

    # Missingness
    create_missingness_by_column(
        train_df,
        "training",
    ).to_csv(
        output_dir / "training_missingness.csv",
        index=False,
    )

    create_missingness_by_column(
        test_df,
        "test",
    ).to_csv(
        output_dir / "test_missingness.csv",
        index=False,
    )

    create_missingness_pattern_summary(
        train_df,
        "training",
    ).to_csv(
        output_dir / "training_missingness_patterns.csv",
        index=False,
    )

    create_missingness_pattern_summary(
        test_df,
        "test",
    ).to_csv(
        output_dir / "test_missingness_patterns.csv",
        index=False,
    )

    # Valid/Invalid
    create_validity_summary(
        train_df
    ).to_csv(
        output_dir / "validity_summary.csv",
        index=False,
    )

    create_missingness_by_validity(
        train_df
    ).to_csv(
        output_dir / "missingness_by_validity.csv",
        index=False,
    )

    # Train/Test coverage
    create_train_test_range_comparison(
        train_df,
        test_df,
    ).to_csv(
        output_dir / "train_test_range_comparison.csv",
        index=False,
    )