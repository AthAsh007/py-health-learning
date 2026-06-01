"""
Regenerate the CSV sample files in data/samples/ from the package generators.

Run this whenever you change the synthetic-data functions so the committed
sample files stay in sync:

    python data/generate_samples.py

It produces the *full-size* synthetic datasets. The files contain no real
patient data. (The small CSVs already committed under data/samples/ are just a
convenience preview.)
"""

from pyhealth_learning import data


if __name__ == "__main__":
    for path in data.save_sample_datasets(dest="data/samples"):
        print("wrote", path)
