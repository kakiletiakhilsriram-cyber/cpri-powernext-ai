from pathlib import Path

import pandas as pd

from data_loader import load_excel
from regime_analysis import (
    find_nearest_valid_operating_points,
)


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "raw"
    / "CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx"
)


def main() -> None:

    train_df, test_df = load_excel(
        DATA_PATH
    )

    row = test_df[
        test_df["Test_ID"] == "TST-0213"
    ]

    if row.empty:
        raise ValueError(
            "TST-0213 was not found."
        )

    test_row = row.iloc[0]

    print("=" * 70)
    print("NEAREST VALID OPERATING POINTS — TST-0213")
    print("=" * 70)

    print("\nTest observation:")

    print(
        test_row[
            [
                "Test_ID",
                "Applied_Voltage_kV",
                "Load_Current_A",
                "Ambient_Temperature_C",
                "Test_Duration_min",
                "Sensor_S1",
                "Sensor_S2",
                "Sensor_S3",
                "Sensor_S4",
            ]
        ]
    )

    neighbours = (
        find_nearest_valid_operating_points(
            train_df,
            test_row,
            n_neighbors=10,
        )
    )

    print(
        "\nNearest VALID historical observations:"
    )

    display_columns = [
        "Test_ID",
        "Applied_Voltage_kV",
        "Load_Current_A",
        "Ambient_Temperature_C",
        "Test_Duration_min",
        "Sensor_S1",
        "Sensor_S2",
        "Sensor_S3",
        "Reference_Parameter",
        "operating_distance",
    ]

    print(
        neighbours[display_columns]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()