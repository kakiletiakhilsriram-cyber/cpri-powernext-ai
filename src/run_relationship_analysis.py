from pathlib import Path

from data_loader import load_excel

from relationship_analysis import (
    save_relationship_outputs,
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
    print("CPRI — FEATURE / TARGET RELATIONSHIP ANALYSIS")
    print("=" * 70)

    train_df, _ = load_excel(
        DATA_PATH
    )

    valid_df = train_df[
        train_df["Validity_Label"] == "Valid"
    ].copy()

    print(
        f"\nAll historical records: {len(train_df)}"
    )

    print(
        f"Valid historical records: {len(valid_df)}"
    )

    save_relationship_outputs(
        valid_df=valid_df,
        output_dir=OUTPUT_DIR,
    )

    print(
        "\nRelationship analysis completed."
    )

    print("\nGenerated outputs:")

    files = [
        "feature_target_relationships.csv",
        "feature_spearman_matrix.csv",
        "feature_mutual_information.csv",
        "candidate_interaction_relationships.csv",
    ]

    for filename in files:

        path = OUTPUT_DIR / filename

        status = (
            "OK"
            if path.exists()
            else "MISSING"
        )

        print(
            f"  [{status}] {filename}"
        )


if __name__ == "__main__":
    main()