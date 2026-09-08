from pathlib import Path

from data_loader import load_excel

from regime_analysis import (
    add_operating_bins,
    add_two_dimensional_operating_bins,
    create_current_regime_summary,
    create_voltage_regime_summary,
    create_validity_by_current_regime,
    create_response_quantiles_by_current_regime,
    create_current_vs_sensor_relationship,
    create_valid_only_regime_data,
    create_current_voltage_regime_summary,
    create_shared_current_regime_comparison,
)


# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "raw"
    / "CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "data_profiles"
)


def main() -> None:

    print("=" * 70)
    print("CPRI — OPERATING REGIME ANALYSIS")
    print("=" * 70)

    # ---------------------------------------------------------------
    # 1. Load historical training data
    # ---------------------------------------------------------------

    print("\nLoading historical training data...")

    train_df, _ = load_excel(
        DATA_PATH
    )

    print(
        f"Historical records: {len(train_df)}"
    )

    # ---------------------------------------------------------------
    # 2. Create Valid-only dataset
    # ---------------------------------------------------------------

    valid_df = create_valid_only_regime_data(
        train_df
    )

    print(
        f"Valid historical records: {len(valid_df)}"
    )

    # ---------------------------------------------------------------
    # 3. Create one-dimensional operating bins
    # ---------------------------------------------------------------

    print(
        "\nCreating one-dimensional operating regimes..."
    )

    all_regime_df = add_operating_bins(
        train_df
    )

    valid_regime_df = add_operating_bins(
        valid_df
    )

    # Save full regime-labelled datasets
    all_regime_df.to_csv(
        OUTPUT_DIR
        / "training_operating_regimes.csv",
        index=False,
    )

    valid_regime_df.to_csv(
        OUTPUT_DIR
        / "valid_operating_regimes.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # 4. Current regime summaries
    # ---------------------------------------------------------------

    create_current_regime_summary(
        all_regime_df
    ).to_csv(
        OUTPUT_DIR
        / "current_regime_summary.csv",
        index=False,
    )

    create_current_regime_summary(
        valid_regime_df
    ).to_csv(
        OUTPUT_DIR
        / "valid_current_regime_summary.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # 5. Voltage regime summaries
    # ---------------------------------------------------------------

    create_voltage_regime_summary(
        all_regime_df
    ).to_csv(
        OUTPUT_DIR
        / "voltage_regime_summary.csv",
        index=False,
    )

    create_voltage_regime_summary(
        valid_regime_df
    ).to_csv(
        OUTPUT_DIR
        / "valid_voltage_regime_summary.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # 6. Valid/Invalid distribution by current regime
    # ---------------------------------------------------------------

    create_validity_by_current_regime(
        all_regime_df
    ).to_csv(
        OUTPUT_DIR
        / "validity_by_current_regime.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # 7. Response quantiles by current regime
    #    Use VALID records for expected behaviour
    # ---------------------------------------------------------------

    create_response_quantiles_by_current_regime(
        valid_regime_df
    ).to_csv(
        OUTPUT_DIR
        / "valid_response_quantiles_by_current_regime.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # 8. Current-response correlations
    #    Use VALID records
    # ---------------------------------------------------------------

    create_current_vs_sensor_relationship(
        valid_regime_df
    ).to_csv(
        OUTPUT_DIR
        / "valid_current_response_correlations.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # 9. Correct ALL vs VALID comparison
    #
    # Important:
    # The same current-regime boundaries are used for both datasets.
    # ---------------------------------------------------------------

    shared_comparison = (
        create_shared_current_regime_comparison(
            train_df,
            valid_df,
        )
    )

    shared_comparison.to_csv(
        OUTPUT_DIR
        / "all_vs_valid_current_regimes_shared.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # 10. Two-dimensional Current × Voltage analysis
    #    Use VALID historical records
    # ---------------------------------------------------------------

    valid_2d = (
        add_two_dimensional_operating_bins(
            valid_df
        )
    )

    valid_2d.to_csv(
        OUTPUT_DIR
        / "valid_current_voltage_regimes.csv",
        index=False,
    )

    create_current_voltage_regime_summary(
        valid_2d
    ).to_csv(
        OUTPUT_DIR
        / "valid_current_voltage_regime_summary.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # 11. Final status
    # ---------------------------------------------------------------

    print(
        "\nOperating-regime analysis completed successfully."
    )

    print(
        "\nImportant outputs:"
    )

    output_files = [
        "valid_current_regime_summary.csv",
        "valid_voltage_regime_summary.csv",
        "valid_response_quantiles_by_current_regime.csv",
        "valid_current_response_correlations.csv",
        "all_vs_valid_current_regimes_shared.csv",
        "valid_current_voltage_regime_summary.csv",
    ]

    for filename in output_files:
        path = OUTPUT_DIR / filename

        status = "OK" if path.exists() else "MISSING"

        print(
            f"  [{status}] {filename}"
        )


if __name__ == "__main__":
    main()