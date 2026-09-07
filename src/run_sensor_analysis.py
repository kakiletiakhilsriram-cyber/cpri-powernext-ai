from pathlib import Path

from data_loader import load_excel

from sensor_analysis import (
    add_sensor_consistency_features,
    create_sensor_consistency_summary,
    create_sensor_consistency_by_validity,
    create_robust_sensor_scores,
)


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
    print("CPRI — SENSOR CONSISTENCY ANALYSIS")
    print("=" * 70)

    train_df, test_df = load_excel(
        DATA_PATH
    )

    # ---------------------------------------------------------------
    # Add diagnostic features
    # ---------------------------------------------------------------

    train_analysis = (
        add_sensor_consistency_features(
            train_df
        )
    )

    test_analysis = (
        add_sensor_consistency_features(
            test_df
        )
    )

    # ---------------------------------------------------------------
    # Robust diagnostic scores
    # ---------------------------------------------------------------

    train_analysis = (
        create_robust_sensor_scores(
            train_analysis
        )
    )

    test_analysis = (
        create_robust_sensor_scores(
            test_analysis
        )
    )

    # ---------------------------------------------------------------
    # Save complete diagnostic datasets
    # ---------------------------------------------------------------

    train_analysis.to_csv(
        OUTPUT_DIR
        / "training_sensor_consistency.csv",
        index=False,
    )

    test_analysis.to_csv(
        OUTPUT_DIR
        / "test_sensor_consistency.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    summary = (
        create_sensor_consistency_summary(
            train_analysis
        )
    )

    summary.to_csv(
        OUTPUT_DIR
        / "sensor_consistency_summary.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Valid vs Invalid comparison
    # ---------------------------------------------------------------

    by_validity = (
        create_sensor_consistency_by_validity(
            train_analysis
        )
    )

    by_validity.to_csv(
        OUTPUT_DIR
        / "sensor_consistency_by_validity.csv",
        index=False,
    )

    print("\nSensor consistency analysis completed.")

    print(
        "\nGenerated files:"
    )

    for path in [
        "training_sensor_consistency.csv",
        "test_sensor_consistency.csv",
        "sensor_consistency_summary.csv",
        "sensor_consistency_by_validity.csv",
    ]:
        print(f"  - {path}")


if __name__ == "__main__":
    main()