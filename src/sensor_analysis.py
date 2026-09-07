from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SENSORS = [
    "Sensor_S1",
    "Sensor_S2",
    "Sensor_S3",
]


def add_sensor_consistency_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add diagnostic sensor-consistency features.

    IMPORTANT:
    These features are used for analysis only at this stage.
    The function does not change or remove original sensor values.
    """

    result = df.copy()

    # ---------------------------------------------------------------
    # Pairwise sensor differences
    # ---------------------------------------------------------------

    result["S2_minus_S1"] = (
        result["Sensor_S2"]
        - result["Sensor_S1"]
    )

    result["S3_minus_S2"] = (
        result["Sensor_S3"]
        - result["Sensor_S2"]
    )

    result["S3_minus_S1"] = (
        result["Sensor_S3"]
        - result["Sensor_S1"]
    )

    # ---------------------------------------------------------------
    # Overall spread among the three primary sensors
    # ---------------------------------------------------------------

    result["Sensor_123_Max"] = (
        result[SENSORS].max(axis=1)
    )

    result["Sensor_123_Min"] = (
        result[SENSORS].min(axis=1)
    )

    result["Sensor_123_Spread"] = (
        result["Sensor_123_Max"]
        - result["Sensor_123_Min"]
    )

    result["Sensor_123_Mean"] = (
        result[SENSORS].mean(axis=1)
    )

    result["Sensor_123_Median"] = (
        result[SENSORS].median(axis=1)
    )

    # ---------------------------------------------------------------
    # Deviation of each sensor from the median
    # ---------------------------------------------------------------

    result["S1_Median_Deviation"] = (
        result["Sensor_S1"]
        - result["Sensor_123_Median"]
    )

    result["S2_Median_Deviation"] = (
        result["Sensor_S2"]
        - result["Sensor_123_Median"]
    )

    result["S3_Median_Deviation"] = (
        result["Sensor_S3"]
        - result["Sensor_123_Median"]
    )

    result["Max_Absolute_Median_Deviation"] = (
        result[
            [
                "S1_Median_Deviation",
                "S2_Median_Deviation",
                "S3_Median_Deviation",
            ]
        ]
        .abs()
        .max(axis=1)
    )

    return result


def create_sensor_consistency_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate summary statistics for sensor-consistency features.
    """

    features = [
        "S2_minus_S1",
        "S3_minus_S2",
        "S3_minus_S1",
        "Sensor_123_Spread",
        "Max_Absolute_Median_Deviation",
    ]

    available = [
        column
        for column in features
        if column in df.columns
    ]

    if not available:
        return pd.DataFrame()

    return (
        df[available]
        .describe()
        .T
        .reset_index()
        .rename(columns={"index": "feature"})
    )


def create_sensor_consistency_by_validity(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare sensor-consistency statistics between Valid
    and Invalid historical records.
    """

    if "Validity_Label" not in df.columns:
        return pd.DataFrame()

    features = [
        "S2_minus_S1",
        "S3_minus_S2",
        "S3_minus_S1",
        "Sensor_123_Spread",
        "Max_Absolute_Median_Deviation",
    ]

    rows = []

    for label, group in df.groupby(
        "Validity_Label",
        dropna=False,
    ):

        for feature in features:

            if feature not in group.columns:
                continue

            series = group[feature].dropna()

            if series.empty:
                continue

            rows.append(
                {
                    "Validity_Label": label,
                    "feature": feature,
                    "count": len(series),
                    "mean": series.mean(),
                    "median": series.median(),
                    "std": series.std(),
                    "q25": series.quantile(0.25),
                    "q75": series.quantile(0.75),
                    "q95": series.quantile(0.95),
                    "max_abs": series.abs().max(),
                }
            )

    return pd.DataFrame(rows)


def create_robust_sensor_scores(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate robust standardized scores for sensor-consistency
    features using the IQR/MAD-style approach.

    These are diagnostic scores only. They are not final anomaly
    classifications.
    """

    result = df.copy()

    features = [
        "S2_minus_S1",
        "S3_minus_S2",
        "S3_minus_S1",
        "Sensor_123_Spread",
        "Max_Absolute_Median_Deviation",
    ]

    for feature in features:

        if feature not in result.columns:
            continue

        median = result[feature].median()

        mad = (
            result[feature] - median
        ).abs().median()

        if pd.isna(mad) or mad == 0:

            result[f"{feature}_robust_score"] = 0.0

        else:

            result[f"{feature}_robust_score"] = (
                0.6745
                * (
                    result[feature]
                    - median
                )
                / mad
            )

    return result