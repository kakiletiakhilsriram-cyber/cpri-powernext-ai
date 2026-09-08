from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


OPERATING_COLUMNS = [
    "Applied_Voltage_kV",
    "Load_Current_A",
]

CONDITION_COLUMNS = [
    "Ambient_Temperature_C",
    "Test_Duration_min",
]

RESPONSE_COLUMNS = [
    "Sensor_S1",
    "Sensor_S2",
    "Sensor_S3",
    "Reference_Parameter",
]


def add_current_voltage_regimes(
    df: pd.DataFrame,
    q_current: int = 4,
    q_voltage: int = 4,
) -> pd.DataFrame:
    """
    Add current and voltage quantile regimes for conditional analysis.

    These are exploratory bins, not final physical operating limits.
    """

    result = df.copy()

    result["Current_Regime"] = pd.qcut(
        result["Load_Current_A"],
        q=q_current,
        duplicates="drop",
    )

    result["Voltage_Regime"] = pd.qcut(
        result["Applied_Voltage_kV"],
        q=q_voltage,
        duplicates="drop",
    )

    return result


def create_condition_regime_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize the response as a function of one conditioning variable
    at a time.

    Uses Valid historical observations only.
    """

    rows = []

    for condition in CONDITION_COLUMNS:

        if condition not in df.columns:
            continue

        temp = df.copy()

        temp[f"{condition}_Regime"] = pd.qcut(
            temp[condition],
            q=4,
            duplicates="drop",
        )

        for regime, group in temp.groupby(
            f"{condition}_Regime",
            observed=True,
        ):

            row = {
                "condition": condition,
                "regime": str(regime),
                "record_count": len(group),
                "condition_mean": group[condition].mean(),
            }

            for response in RESPONSE_COLUMNS:

                if response not in group.columns:
                    continue

                row[f"{response}_mean"] = (
                    group[response].mean()
                )

                row[f"{response}_median"] = (
                    group[response].median()
                )

            rows.append(row)

    return pd.DataFrame(rows)


def create_condition_correlations(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate Spearman correlations between ambient/duration and
    sensor/reference responses.

    This is an exploratory monotonic association measure.
    """

    rows = []

    for condition in CONDITION_COLUMNS:

        for response in RESPONSE_COLUMNS:

            pair = df[
                [
                    condition,
                    response,
                ]
            ].dropna()

            if len(pair) < 5:
                continue

            rho, p_value = spearmanr(
                pair[condition],
                pair[response],
            )

            rows.append(
                {
                    "condition": condition,
                    "response": response,
                    "n": len(pair),
                    "spearman_rho": rho,
                    "p_value": p_value,
                }
            )

    return pd.DataFrame(rows)


def create_within_operating_regime_correlations(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate condition-response correlations separately within
    Current × Voltage regimes.

    This helps determine whether apparent ambient/duration effects
    remain when comparing observations within similar operating
    regions.
    """

    if (
        "Current_Regime" not in df.columns
        or "Voltage_Regime" not in df.columns
    ):
        df = add_current_voltage_regimes(df)

    rows = []

    grouped = df.groupby(
        [
            "Current_Regime",
            "Voltage_Regime",
        ],
        observed=True,
    )

    for (
        current_regime,
        voltage_regime,
    ), group in grouped:

        # Skip extremely small cells.
        if len(group) < 10:
            continue

        for condition in CONDITION_COLUMNS:

            for response in RESPONSE_COLUMNS:

                pair = group[
                    [
                        condition,
                        response,
                    ]
                ].dropna()

                if len(pair) < 5:
                    continue

                rho, p_value = spearmanr(
                    pair[condition],
                    pair[response],
                )

                rows.append(
                    {
                        "current_regime": str(
                            current_regime
                        ),
                        "voltage_regime": str(
                            voltage_regime
                        ),
                        "record_count": len(group),
                        "condition": condition,
                        "response": response,
                        "n": len(pair),
                        "spearman_rho": rho,
                        "p_value": p_value,
                    }
                )

    return pd.DataFrame(rows)


def create_regime_condition_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare Reference and primary sensor behaviour across
    Ambient and Duration bins within Current × Voltage regimes.
    """

    if (
        "Current_Regime" not in df.columns
        or "Voltage_Regime" not in df.columns
    ):
        df = add_current_voltage_regimes(df)

    rows = []

    grouped = df.groupby(
        [
            "Current_Regime",
            "Voltage_Regime",
        ],
        observed=True,
    )

    for (
        current_regime,
        voltage_regime,
    ), group in grouped:

        if len(group) < 10:
            continue

        for condition in CONDITION_COLUMNS:

            temp = group.copy()

            try:
                temp["Condition_Regime"] = pd.qcut(
                    temp[condition],
                    q=3,
                    duplicates="drop",
                )
            except ValueError:
                continue

            for condition_regime, sub_group in temp.groupby(
                "Condition_Regime",
                observed=True,
            ):

                if len(sub_group) < 3:
                    continue

                rows.append(
                    {
                        "current_regime": str(
                            current_regime
                        ),
                        "voltage_regime": str(
                            voltage_regime
                        ),
                        "condition": condition,
                        "condition_regime": str(
                            condition_regime
                        ),
                        "record_count": len(sub_group),
                        "condition_mean": (
                            sub_group[condition].mean()
                        ),
                        "reference_median": (
                            sub_group[
                                "Reference_Parameter"
                            ].median()
                        ),
                        "reference_mean": (
                            sub_group[
                                "Reference_Parameter"
                            ].mean()
                        ),
                        "s1_median": (
                            sub_group["Sensor_S1"]
                            .median()
                        ),
                        "s2_median": (
                            sub_group["Sensor_S2"]
                            .median()
                        ),
                        "s3_median": (
                            sub_group["Sensor_S3"]
                            .median()
                        ),
                    }
                )

    return pd.DataFrame(rows)


def save_conditional_analysis(
    valid_df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """
    Run the complete Ambient/Duration conditional analysis
    and save results.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Add shared operating-context bins
    # ---------------------------------------------------------------

    analysis_df = add_current_voltage_regimes(
        valid_df
    )

    # ---------------------------------------------------------------
    # Save analysis data
    # ---------------------------------------------------------------

    analysis_df.to_csv(
        output_dir
        / "valid_conditional_analysis_data.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Overall condition summaries
    # ---------------------------------------------------------------

    create_condition_regime_summary(
        analysis_df
    ).to_csv(
        output_dir
        / "ambient_duration_regime_summary.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Overall correlations
    # ---------------------------------------------------------------

    create_condition_correlations(
        analysis_df
    ).to_csv(
        output_dir
        / "ambient_duration_correlations.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Within Current × Voltage regime correlations
    # ---------------------------------------------------------------

    create_within_operating_regime_correlations(
        analysis_df
    ).to_csv(
        output_dir
        / "within_operating_regime_correlations.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Condition effects within operating regimes
    # ---------------------------------------------------------------

    create_regime_condition_summary(
        analysis_df
    ).to_csv(
        output_dir
        / "regime_condition_summary.csv",
        index=False,
    )