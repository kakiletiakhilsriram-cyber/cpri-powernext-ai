from pathlib import Path

from data_loader import load_excel


# Project root:
# cpri-powernext-ai/
ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "raw"
    / "CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx"
)


def main() -> None:

    print("=" * 70)
    print("CPRI DATA LOADER TEST")
    print("=" * 70)

    train_df, test_df = load_excel(DATA_PATH)

    print("\nTraining shape:")
    print(train_df.shape)

    print("\nTest shape:")
    print(test_df.shape)

    print("\nTraining columns:")
    for column in train_df.columns:
        print(f"  {column}")

    print("\nTest columns:")
    for column in test_df.columns:
        print(f"  {column}")

    print("\nTraining validity labels:")
    print(
        train_df["Validity_Label"]
        .value_counts(dropna=False)
    )

    print("\nTraining data types:")
    print(train_df.dtypes)

    print("\nTest data types:")
    print(test_df.dtypes)

    print("\nData loader test PASSED.")


if __name__ == "__main__":
    main()