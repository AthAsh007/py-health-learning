"""
Example 01 — Explore the datasets.

Run from the repo root:

    python examples/01_explore_data.py
"""

from pyhealth_learning import data, utils


def main() -> None:
    utils.section("Breast-cancer diagnostic dataset")
    df = data.load_breast_cancer_classification()
    print(f"shape: {df.shape}")
    print(df[["mean radius", "mean texture", "malignant"]].head())
    print("\nclass balance:")
    print(df["malignant"].value_counts(normalize=True).round(3))

    utils.section("Diabetes classification dataset")
    dia = data.load_diabetes_classification()
    print(f"shape: {dia.shape}")
    print("class balance:")
    print(dia["diabetes"].value_counts(normalize=True).round(3))

    utils.section("Synthetic patient table (with injected mess)")
    patients = data.make_synthetic_patients(n=300)
    print(f"shape: {patients.shape}")
    print(patients.head())
    print("\nmissing values per column:")
    print(patients.isna().sum())


if __name__ == "__main__":
    main()
