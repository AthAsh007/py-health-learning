# data/

This folder is intentionally (almost) empty.

The learning examples **don't ship any patient data**. Instead:

- Real datasets come bundled with scikit-learn and load with no download
  (`load_breast_cancer`, `load_diabetes`) — see `src/pyhealth_learning/data.py`.
- Practice "messy" data is generated on the fly and reproducibly by
  `data.make_synthetic_patients()` — it contains **no real patient
  information**.

If you add your own CSVs here for experiments, they're covered by `.gitignore`
patterns where appropriate — never commit protected health information (PHI).
