from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


# -------------------------------------------------------------------
# Expected schema from the competition workbook
# -------------------------------------------------------------------

TRAIN_REQUIRED_COLUMNS = [
    "Test_ID",
    "Applied_Voltage_kV",
    "Load_Current_A",
    "Ambient_Temperature_C",
    "Test_Duration_min",
    "Sensor_S1",
    "Sensor_S2",
    "Sensor_S3",
    "Sensor_S4",
    "Reference_Parameter",
    "Validity_Label",
]

TEST_REQUIRED_COLUMNS = [
    "Test_ID",
    "Applied_Voltage_kV",
    "Load_Current_A",
    "Ambient_Temperature_C",
    "Test_Duration_min",
    "Sensor_S1",
    "Sensor_S2",
    "Sensor_S3",
    "Sensor_S4",
]


# Columns that should be numeric
NUMERIC_COLUMNS = [
    "Applied_Voltage_kV",
    "Load_Current_A",
    "Ambient_Temperature_C",
    "Test_Duration_min",
    "Sensor_S1",
    "Sensor_S2",
    "Sensor_S3",
    "Sensor_S4",
    "Reference_Parameter",
]


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names.

    This only removes accidental leading/trailing spaces and converts
    internal whitespace into underscores.
    """
    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
    )

    return df


def validate_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    dataset_name: str,
) -> None:
    """
    Ensure all expected columns are present.
    """

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: "
            f"{missing_columns}"
        )


def convert_numeric_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert expected numerical columns to numeric dtype.

    Invalid numeric strings become NaN.
    We intentionally do NOT fill missing values here.
    """
    df = df.copy()

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    return df


def standardize_validity_label(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Standardize historical Validity_Label values.
    """
    df = df.copy()

    if "Validity_Label" in df.columns:
        df["Validity_Label"] = (
            df["Validity_Label"]
            .astype("string")
            .str.strip()
            .str.title()
        )

    return df


def load_excel(
    file_path: str | Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Training_Data and Test_Data from the competition workbook.

    Parameters
    ----------
    file_path:
        Path to the competition Excel workbook.

    Returns
    -------
    train_df:
        Historical training data.

    test_df:
        New test data.
    """

    file_path = Path(file_path)

    # ---------------------------------------------------------------
    # 1. Check that the file exists
    # ---------------------------------------------------------------

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file was not found:\n"
            f"{file_path.resolve()}"
        )

    # ---------------------------------------------------------------
    # 2. Read workbook structure
    # ---------------------------------------------------------------

    workbook = pd.ExcelFile(file_path)

    required_sheets = {
        "Training_Data",
        "Test_Data",
    }

    missing_sheets = (
        required_sheets
        - set(workbook.sheet_names)
    )

    if missing_sheets:
        raise ValueError(
            f"Workbook is missing required sheets: "
            f"{sorted(missing_sheets)}\n"
            f"Available sheets: {workbook.sheet_names}"
        )

    # ---------------------------------------------------------------
    # 3. Load the two datasets
    # ---------------------------------------------------------------

    train_df = pd.read_excel(
        workbook,
        sheet_name="Training_Data",
    )

    test_df = pd.read_excel(
        workbook,
        sheet_name="Test_Data",
    )

    # ---------------------------------------------------------------
    # 4. Standardize column names
    # ---------------------------------------------------------------

    train_df = clean_column_names(train_df)
    test_df = clean_column_names(test_df)

    # ---------------------------------------------------------------
    # 5. Validate schemas
    # ---------------------------------------------------------------

    validate_columns(
        train_df,
        TRAIN_REQUIRED_COLUMNS,
        "Training_Data",
    )

    validate_columns(
        test_df,
        TEST_REQUIRED_COLUMNS,
        "Test_Data",
    )

    # ---------------------------------------------------------------
    # 6. Convert numeric columns
    # ---------------------------------------------------------------

    train_df = convert_numeric_columns(train_df)
    test_df = convert_numeric_columns(test_df)

    # ---------------------------------------------------------------
    # 7. Standardize training labels
    # ---------------------------------------------------------------

    train_df = standardize_validity_label(
        train_df
    )

    return train_df, test_df