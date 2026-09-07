from pathlib import Path

from data_loader import load_excel
from audit import save_audit_outputs


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
    print("CPRI — PART 1 DATA AUDIT")
    print("=" * 70)

    print("\nLoading dataset...")

    train_df, test_df = load_excel(
        DATA_PATH
    )

    print(
        f"Training records: {len(train_df)}"
    )

    print(
        f"Training columns: {len(train_df.columns)}"
    )

    print(
        f"Test records: {len(test_df)}"
    )

    print(
        f"Test columns: {len(test_df.columns)}"
    )

    print("\nGenerating audit outputs...")

    save_audit_outputs(
        train_df=train_df,
        test_df=test_df,
        output_dir=OUTPUT_DIR,
    )

    print("\nAudit completed successfully.")

    print(
        f"Outputs saved to:\n{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()