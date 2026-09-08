from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


NUMERIC_FEATURES = [
    "Applied_Voltage_kV",
    "Load_Current_A",
    "Ambient_Temperature_C",
    "Test_Duration_min",
    "Sensor_S1",
    "Sensor_S2",
    "Sensor_S3",
    "Sensor_S4",
]

TARGET = "Reference_Parameter"


def calculate_pearson_correlations(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate Pearson correlation between each candidate variable
    and Reference_Parameter.
    """

    rows = []

    for feature in NUMERIC_FEATURES:

        if (
            feature not in df.columns
            or TARGET not in df.columns
        ):
            continue

        pair = df[
            [feature, TARGET]
        ].dropna()

        if len(pair) < 3:
            continue

        correlation = pair[
            feature
        ].corr(
            pair[TARGET],
            method="pearson",
        )

        rows.append(
            {
                "feature": feature,
                "n": len(pair),
                "pearson_r": correlation,
                "absolute_pearson_r": abs(
                    correlation
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "absolute_pearson_r",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def calculate_spearman_correlations(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate Spearman rank correlation between each candidate
    variable and Reference_Parameter.
    """

    rows = []

    for feature in NUMERIC_FEATURES:

        if (
            feature not in df.columns
            or TARGET not in df.columns
        ):
            continue

        pair = df[
            [feature, TARGET]
        ].dropna()

        if len(pair) < 3:
            continue

        rho, p_value = spearmanr(
            pair[feature],
            pair[TARGET],
        )

        rows.append(
            {
                "feature": feature,
                "n": len(pair),
                "spearman_rho": rho,
                "absolute_spearman_rho": abs(
                    rho
                ),
                "p_value": p_value,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "absolute_spearman_rho",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def calculate_feature_target_relationships(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine Pearson and Spearman relationships.
    """

    pearson = calculate_pearson_correlations(
        df
    )

    spearman = calculate_spearman_correlations(
        df
    )

    result = pearson.merge(
        spearman,
        on=[
            "feature",
            "n",
        ],
        how="outer",
    )

    return result


def calculate_feature_correlation_matrix(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate Spearman correlation matrix among candidate features.
    """

    available = [
        feature
        for feature in NUMERIC_FEATURES
        if feature in df.columns
    ]

    return df[
        available
    ].corr(
        method="spearman"
    )


def calculate_mutual_information(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate nonlinear dependency between candidate features and
    Reference_Parameter using mutual information.

    Missing rows are removed for each feature separately.
    """

    from sklearn.feature_selection import mutual_info_regression

    rows = []

    for feature in NUMERIC_FEATURES:

        if (
            feature not in df.columns
            or TARGET not in df.columns
        ):
            continue

        pair = df[
            [feature, TARGET]
        ].dropna()

        if len(pair) < 20:
            continue

        X = pair[
            [feature]
        ]

        y = pair[TARGET]

        try:
            score = mutual_info_regression(
                X,
                y,
                random_state=42,
            )[0]
        except Exception:
            score = np.nan

        rows.append(
            {
                "feature": feature,
                "n": len(pair),
                "mutual_information": score,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "mutual_information",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def calculate_candidate_interaction_correlations(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Explore physically motivated derived variables.

    These are candidates only. They are NOT final model features.
    """

    result = df.copy()

    result["Voltage_Current_Product"] = (
        result["Applied_Voltage_kV"]
        * result["Load_Current_A"]
    )

    result["Current_Squared"] = (
        result["Load_Current_A"] ** 2
    )

    result["Voltage_Squared"] = (
        result["Applied_Voltage_kV"] ** 2
    )

    result["Current_Ambient_Product"] = (
        result["Load_Current_A"]
        * result["Ambient_Temperature_C"]
    )

    result["Current_Duration_Product"] = (
        result["Load_Current_A"]
        * result["Test_Duration_min"]
    )

    result["Voltage_Ambient_Product"] = (
        result["Applied_Voltage_kV"]
        * result["Ambient_Temperature_C"]
    )

    candidate_features = [
        "Voltage_Current_Product",
        "Current_Squared",
        "Voltage_Squared",
        "Current_Ambient_Product",
        "Current_Duration_Product",
        "Voltage_Ambient_Product",
    ]

    rows = []

    for feature in candidate_features:

        pair = result[
            [
                feature,
                TARGET,
            ]
        ].dropna()

        if len(pair) < 3:
            continue

        pearson = pair[
            feature
        ].corr(
            pair[TARGET],
            method="pearson",
        )

        spearman = pair[
            feature
        ].corr(
            pair[TARGET],
            method="spearman",
        )

        rows.append(
            {
                "feature": feature,
                "n": len(pair),
                "pearson_r": pearson,
                "spearman_rho": spearman,
                "absolute_spearman": abs(
                    spearman
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "absolute_spearman",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def save_relationship_outputs(
    valid_df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """
    Generate and save all relationship-analysis outputs.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Feature-target relationships
    # ---------------------------------------------------------------

    relationships = (
        calculate_feature_target_relationships(
            valid_df
        )
    )

    relationships.to_csv(
        output_dir
        / "feature_target_relationships.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Feature correlation matrix
    # ---------------------------------------------------------------

    feature_corr = (
        calculate_feature_correlation_matrix(
            valid_df
        )
    )

    feature_corr.to_csv(
        output_dir
        / "feature_spearman_matrix.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Mutual information
    # ---------------------------------------------------------------

    mutual_information = (
        calculate_mutual_information(
            valid_df
        )
    )

    mutual_information.to_csv(
        output_dir
        / "feature_mutual_information.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Candidate interactions
    # ---------------------------------------------------------------

    interactions = (
        calculate_candidate_interaction_correlations(
            valid_df
        )
    )

    interactions.to_csv(
        output_dir
        / "candidate_interaction_relationships.csv",
        index=False,
    )