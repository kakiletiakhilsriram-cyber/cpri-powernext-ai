from pathlib import Path

from data_loader import load_excel

from conditional_analysis import (
    save_conditional_analysis,
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
    print("CPRI — CONDITIONAL OPERATING ANALYSIS")
    print("=" * 70)

    train_df, _ = load_excel(
        DATA_PATH
    )

    # ---------------------------------------------------------------
    # IMPORTANT:
    # For expected normal behaviour, use engineer-verified
    # historical VALID records.
    # ---------------------------------------------------------------

    valid_df = train_df[
        train_df["Validity_Label"] == "Valid"
    ].copy()

    print(
        f"\nHistorical records: {len(train_df)}"
    )

    print(
        f"Valid records used: {len(valid_df)}"
    )

    save_conditional_analysis(
        valid_df=valid_df,
        output_dir=OUTPUT_DIR,
    )

    print(
        "\nConditional analysis completed successfully."
    )

    print("\nGenerated files:")

    files = [
        "ambient_duration_regime_summary.csv",
        "ambient_duration_correlations.csv",
        "within_operating_regime_correlations.csv",
        "regime_condition_summary.csv",
    ]

    for filename in files:
        path = OUTPUT_DIR / filename
        status = "OK" if path.exists() else "MISSING"
        print(f"  [{status}] {filename}")


if __name__ == "__main__":
    main()