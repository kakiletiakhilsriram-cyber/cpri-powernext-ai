from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OPERATING_COLUMNS = [
    "Applied_Voltage_kV",
    "Load_Current_A",
    "Ambient_Temperature_C",
    "Test_Duration_min",
]

RESPONSE_COLUMNS = [
    "Sensor_S1",
    "Sensor_S2",
    "Sensor_S3",
    "Reference_Parameter",
]


def add_operating_bins(
    df: pd.DataFrame,
    q: int = 4,
) -> pd.DataFrame:
    """
    Add quantile-based operating-regime indicators.

    Quantile bins are used for exploratory analysis so that
    each regime contains a reasonably comparable number of
    observations.

    This function does not modify original measurements.
    """

    result = df.copy()

    for column in OPERATING_COLUMNS:

        if column not in result.columns:
            continue

        # qcut can fail if a variable contains too many
        # identical values. duplicates='drop' handles this.
        result[f"{column}_Regime"] = pd.qcut(
            result[column],
            q=q,
            duplicates="drop",
        )

    return result


def create_current_regime_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize response variables across current regimes.
    """

    required = [
        "Load_Current_A_Regime",
        *RESPONSE_COLUMNS,
    ]

    available = [
        column
        for column in required
        if column in df.columns
    ]

    if "Load_Current_A_Regime" not in available:
        return pd.DataFrame()

    summary = (
        df.groupby(
            "Load_Current_A_Regime",
            observed=True,
        )
        .agg(
            record_count=(
                "Load_Current_A",
                "size",
            ),
            current_mean=(
                "Load_Current_A",
                "mean",
            ),
            voltage_mean=(
                "Applied_Voltage_kV",
                "mean",
            ),
            ambient_mean=(
                "Ambient_Temperature_C",
                "mean",
            ),
            duration_mean=(
                "Test_Duration_min",
                "mean",
            ),
            sensor_s1_mean=(
                "Sensor_S1",
                "mean",
            ),
            sensor_s1_std=(
                "Sensor_S1",
                "std",
            ),
            sensor_s2_mean=(
                "Sensor_S2",
                "mean",
            ),
            sensor_s2_std=(
                "Sensor_S2",
                "std",
            ),
            sensor_s3_mean=(
                "Sensor_S3",
                "mean",
            ),
            sensor_s3_std=(
                "Sensor_S3",
                "std",
            ),
            reference_mean=(
                "Reference_Parameter",
                "mean",
            ),
            reference_std=(
                "Reference_Parameter",
                "std",
            ),
        )
        .reset_index()
    )

    return summary


def create_voltage_regime_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize response variables across voltage regimes.
    """

    if "Applied_Voltage_kV_Regime" not in df.columns:
        return pd.DataFrame()

    summary = (
        df.groupby(
            "Applied_Voltage_kV_Regime",
            observed=True,
        )
        .agg(
            record_count=(
                "Applied_Voltage_kV",
                "size",
            ),
            voltage_mean=(
                "Applied_Voltage_kV",
                "mean",
            ),
            current_mean=(
                "Load_Current_A",
                "mean",
            ),
            sensor_s1_mean=(
                "Sensor_S1",
                "mean",
            ),
            sensor_s2_mean=(
                "Sensor_S2",
                "mean",
            ),
            sensor_s3_mean=(
                "Sensor_S3",
                "mean",
            ),
            reference_mean=(
                "Reference_Parameter",
                "mean",
            ),
            reference_std=(
                "Reference_Parameter",
                "std",
            ),
        )
        .reset_index()
    )

    return summary


def create_validity_by_current_regime(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare Valid/Invalid proportions within current regimes.

    This is important because an unusual value may simply
    belong to a different operating region.
    """

    required_columns = [
        "Load_Current_A_Regime",
        "Validity_Label",
    ]

    if not all(
        column in df.columns
        for column in required_columns
    ):
        return pd.DataFrame()

    result = (
        pd.crosstab(
            df["Load_Current_A_Regime"],
            df["Validity_Label"],
            normalize="index",
        )
        .reset_index()
    )

    return result


def create_response_quantiles_by_current_regime(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate response quantiles within current regimes.
    """

    if "Load_Current_A_Regime" not in df.columns:
        return pd.DataFrame()

    rows = []

    for regime, group in df.groupby(
        "Load_Current_A_Regime",
        observed=True,
    ):

        for response in [
            "Sensor_S1",
            "Sensor_S2",
            "Sensor_S3",
            "Reference_Parameter",
        ]:

            if response not in group.columns:
                continue

            series = group[response].dropna()

            if series.empty:
                continue

            rows.append(
                {
                    "current_regime": str(
                        regime
                    ),
                    "response": response,
                    "count": len(series),
                    "q05": series.quantile(0.05),
                    "q25": series.quantile(0.25),
                    "q50": series.quantile(0.50),
                    "q75": series.quantile(0.75),
                    "q95": series.quantile(0.95),
                }
            )

    return pd.DataFrame(rows)


def create_current_vs_sensor_relationship(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate correlations between current and sensor/target
    response variables.
    """

    columns = [
        "Load_Current_A",
        "Sensor_S1",
        "Sensor_S2",
        "Sensor_S3",
        "Reference_Parameter",
    ]

    available = [
        column
        for column in columns
        if column in df.columns
    ]

    if len(available) < 2:
        return pd.DataFrame()

    pearson = (
        df[available]
        .corr(method="pearson")
        ["Load_Current_A"]
        .rename("pearson")
    )

    spearman = (
        df[available]
        .corr(method="spearman")
        ["Load_Current_A"]
        .rename("spearman")
    )

    result = pd.concat(
        [pearson, spearman],
        axis=1,
    ).reset_index()

    result.rename(
        columns={"index": "variable"},
        inplace=True,
    )

    return result


def save_regime_outputs(
    train_df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """
    Generate and save operating-regime analysis outputs.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Add regime indicators
    # ---------------------------------------------------------------

    analysis_df = add_operating_bins(
        train_df
    )

    # ---------------------------------------------------------------
    # Save full analysis dataframe
    # ---------------------------------------------------------------

    analysis_df.to_csv(
        output_dir
        / "training_operating_regimes.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Current regime
    # ---------------------------------------------------------------

    create_current_regime_summary(
        analysis_df
    ).to_csv(
        output_dir
        / "current_regime_summary.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Voltage regime
    # ---------------------------------------------------------------

    create_voltage_regime_summary(
        analysis_df
    ).to_csv(
        output_dir
        / "voltage_regime_summary.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Validity by current regime
    # ---------------------------------------------------------------

    create_validity_by_current_regime(
        analysis_df
    ).to_csv(
        output_dir
        / "validity_by_current_regime.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Response distributions by current regime
    # ---------------------------------------------------------------

    create_response_quantiles_by_current_regime(
        analysis_df
    ).to_csv(
        output_dir
        / "response_quantiles_by_current_regime.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Correlations
    # ---------------------------------------------------------------

    create_current_vs_sensor_relationship(
        analysis_df
    ).to_csv(
        output_dir
        / "current_response_correlations.csv",
        index=False,
    )

def create_valid_only_regime_data(
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return historical records labelled Valid.

    Invalid records are retained in the original dataframe and are
    only excluded from this particular expected-behaviour analysis.
    """

    if "Validity_Label" not in train_df.columns:
        raise ValueError(
            "Validity_Label column is required."
        )

    return train_df[
        train_df["Validity_Label"] == "Valid"
    ].copy()


def add_two_dimensional_operating_bins(
    df: pd.DataFrame,
    q_current: int = 4,
    q_voltage: int = 4,
) -> pd.DataFrame:
    """
    Add independent current and voltage quantile bins.

    These bins are for exploratory analysis only.
    They do not represent final physical operating boundaries.
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


def create_current_voltage_regime_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize equipment response across Current × Voltage regimes.
    """

    required = [
        "Current_Regime",
        "Voltage_Regime",
        "Load_Current_A",
        "Applied_Voltage_kV",
        "Sensor_S1",
        "Sensor_S2",
        "Sensor_S3",
        "Reference_Parameter",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns for 2D regime analysis: {missing}"
        )

    summary = (
        df.groupby(
            [
                "Current_Regime",
                "Voltage_Regime",
            ],
            observed=True,
        )
        .agg(
            record_count=(
                "Test_ID",
                "size",
            ),
            current_mean=(
                "Load_Current_A",
                "mean",
            ),
            voltage_mean=(
                "Applied_Voltage_kV",
                "mean",
            ),
            sensor_s1_mean=(
                "Sensor_S1",
                "mean",
            ),
            sensor_s1_median=(
                "Sensor_S1",
                "median",
            ),
            sensor_s2_mean=(
                "Sensor_S2",
                "mean",
            ),
            sensor_s2_median=(
                "Sensor_S2",
                "median",
            ),
            sensor_s3_mean=(
                "Sensor_S3",
                "mean",
            ),
            sensor_s3_median=(
                "Sensor_S3",
                "median",
            ),
            reference_mean=(
                "Reference_Parameter",
                "mean",
            ),
            reference_median=(
                "Reference_Parameter",
                "median",
            ),
            reference_std=(
                "Reference_Parameter",
                "std",
            ),
        )
        .reset_index()
    )

    return summary


def compare_valid_all_current_regimes(
    all_df: pd.DataFrame,
    valid_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare current-regime behaviour using all historical records
    versus Valid-only historical records.
    """

    all_regime = add_operating_bins(
        all_df
    )

    valid_regime = add_operating_bins(
        valid_df
    )

    all_summary = (
        all_regime
        .groupby(
            "Load_Current_A_Regime",
            observed=True,
        )
        .agg(
            all_count=(
                "Test_ID",
                "size",
            ),
            all_reference_median=(
                "Reference_Parameter",
                "median",
            ),
            all_reference_mean=(
                "Reference_Parameter",
                "mean",
            ),
            all_s1_median=(
                "Sensor_S1",
                "median",
            ),
            all_s2_median=(
                "Sensor_S2",
                "median",
            ),
            all_s3_median=(
                "Sensor_S3",
                "median",
            ),
        )
        .reset_index()
    )

    valid_summary = (
        valid_regime
        .groupby(
            "Load_Current_A_Regime",
            observed=True,
        )
        .agg(
            valid_count=(
                "Test_ID",
                "size",
            ),
            valid_reference_median=(
                "Reference_Parameter",
                "median",
            ),
            valid_reference_mean=(
                "Reference_Parameter",
                "mean",
            ),
            valid_s1_median=(
                "Sensor_S1",
                "median",
            ),
            valid_s2_median=(
                "Sensor_S2",
                "median",
            ),
            valid_s3_median=(
                "Sensor_S3",
                "median",
            ),
        )
        .reset_index()
    )

    result = all_summary.merge(
        valid_summary,
        on="Load_Current_A_Regime",
        how="outer",
    )

    return result

def find_nearest_valid_operating_points(
    train_df: pd.DataFrame,
    test_row: pd.Series,
    n_neighbors: int = 10,
) -> pd.DataFrame:
    """
    Find historically Valid records with similar operating
    conditions to a test observation.

    This is an exploratory diagnostic, not a final model.
    """

    valid_df = train_df[
        train_df["Validity_Label"] == "Valid"
    ].copy()

    operating_columns = [
        "Applied_Voltage_kV",
        "Load_Current_A",
        "Ambient_Temperature_C",
        "Test_Duration_min",
    ]

    # Keep only rows where all operating variables exist.
    valid_df = valid_df.dropna(
        subset=operating_columns
    )

    if valid_df.empty:
        return pd.DataFrame()

    # ---------------------------------------------------------------
    # Standardize using historical VALID population.
    # This prevents current, voltage, temperature and duration from
    # dominating purely because of their numerical scale.
    # ---------------------------------------------------------------

    mean = valid_df[
        operating_columns
    ].mean()

    std = valid_df[
        operating_columns
    ].std()

    # Avoid division by zero.
    std = std.replace(
        0,
        1,
    )

    valid_scaled = (
        valid_df[operating_columns]
        - mean
    ) / std

    test_values = (
        test_row[operating_columns]
        .astype(float)
    )

    test_scaled = (
        test_values
        - mean
    ) / std

    distances = np.sqrt(
        (
            valid_scaled
            - test_scaled
        )
        .pow(2)
        .sum(axis=1)
    )

    result = valid_df.copy()

    result["operating_distance"] = distances

    result = (
        result
        .sort_values(
            "operating_distance"
        )
        .head(n_neighbors)
        .reset_index(drop=True)
    )

    return result

def create_shared_current_bins(
    all_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    q: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create current-regime boundaries from the full historical
    dataset and apply the exact same boundaries to both ALL
    and VALID-only datasets.

    This avoids comparing different quantile boundaries.
    """

    all_result = all_df.copy()
    valid_result = valid_df.copy()

    # ---------------------------------------------------------------
    # Determine boundaries using ALL historical observations
    # ---------------------------------------------------------------

    _, bin_edges = pd.qcut(
        all_result["Load_Current_A"],
        q=q,
        retbins=True,
        duplicates="drop",
    )

    # Ensure the extreme endpoints include the full data.
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Create readable labels.
    regime_labels = [
        f"Q{i + 1}"
        for i in range(len(bin_edges) - 1)
    ]

    # ---------------------------------------------------------------
    # Apply identical bins to both datasets
    # ---------------------------------------------------------------

    all_result["Current_Regime_Shared"] = pd.cut(
        all_result["Load_Current_A"],
        bins=bin_edges,
        labels=regime_labels,
        include_lowest=True,
    )

    valid_result["Current_Regime_Shared"] = pd.cut(
        valid_result["Load_Current_A"],
        bins=bin_edges,
        labels=regime_labels,
        include_lowest=True,
    )

    return all_result, valid_result

def create_shared_current_regime_comparison(
    all_df: pd.DataFrame,
    valid_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare ALL vs VALID-only behaviour using identical
    current-regime boundaries.
    """

    all_result, valid_result = (
        create_shared_current_bins(
            all_df,
            valid_df,
        )
    )

    all_summary = (
        all_result
        .groupby(
            "Current_Regime_Shared",
            observed=True,
        )
        .agg(
            all_count=("Test_ID", "size"),
            all_current_mean=(
                "Load_Current_A",
                "mean",
            ),
            all_reference_mean=(
                "Reference_Parameter",
                "mean",
            ),
            all_reference_median=(
                "Reference_Parameter",
                "median",
            ),
            all_s1_median=(
                "Sensor_S1",
                "median",
            ),
            all_s2_median=(
                "Sensor_S2",
                "median",
            ),
            all_s3_median=(
                "Sensor_S3",
                "median",
            ),
        )
        .reset_index()
    )

    valid_summary = (
        valid_result
        .groupby(
            "Current_Regime_Shared",
            observed=True,
        )
        .agg(
            valid_count=("Test_ID", "size"),
            valid_current_mean=(
                "Load_Current_A",
                "mean",
            ),
            valid_reference_mean=(
                "Reference_Parameter",
                "mean",
            ),
            valid_reference_median=(
                "Reference_Parameter",
                "median",
            ),
            valid_s1_median=(
                "Sensor_S1",
                "median",
            ),
            valid_s2_median=(
                "Sensor_S2",
                "median",
            ),
            valid_s3_median=(
                "Sensor_S3",
                "median",
            ),
        )
        .reset_index()
    )

    comparison = all_summary.merge(
        valid_summary,
        on="Current_Regime_Shared",
        how="outer",
    )

    return comparison