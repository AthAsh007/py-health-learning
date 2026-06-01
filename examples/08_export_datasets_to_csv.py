"""
Example 08 — Export the synthetic datasets to CSV files.

Writes reproducible CSV copies of every synthetic dataset into ``data/samples/``
so you can open them in a spreadsheet or load them without importing the
package. Real bundled datasets (breast-cancer, diabetes) load instantly from
scikit-learn and are deliberately not exported.

    python examples/08_export_datasets_to_csv.py
"""

from pyhealth_learning import data, utils


def main() -> None:
    utils.section("Exporting synthetic datasets")
    written = data.save_sample_datasets(dest="data/samples")
    for path in written:
        print("  wrote", path)
    print(f"\n{len(written)} files written. These contain NO real patient data.")


if __name__ == "__main__":
    main()
