# Predicting oxygen adsorption energies on alloy surfaces

**MASC 520 final project | Bowen Zhang**

I trained a small neural network to estimate the reaction energy for atomic oxygen on ideal pure-metal and bimetallic alloy surfaces. The question is whether composition and surface descriptors known before a new DFT calculation can help narrow a list of candidate surfaces. This repository contains the data preparation, training, evaluation, and prediction code, together with the final report and poster.

The target is the lowest valid O-site energy on each surface for `H2O(g) - H2(g) + * -> O*`, in eV. This is a reaction energy referenced to H2O and H2, not a measured catalytic rate. I selected records at O coverage 0.25. That coverage is a project filter, not a property of every record in the source publication.

## Data

The source is [Mamun et al. (2019)](https://doi.org/10.1038/s41597-019-0080-z), accessed as parsed reactions through [Catalysis-Hub](https://www.catalysis-hub.org/publications/MamunHighT2019). The paper describes 2,035 ideal A1, L1₂, and L1₀ surfaces. My reaction, coverage, energy, and supported-composition filters retained **1,711 surface/facet labels**. I normalized formula order, grouped valid site records by surface, and kept the lowest energy. The preparation script records its filtering counts in `data/processed/surfaces.summary.json`.

Model inputs are available before an O adsorption calculation: formula, ideal structure, facet, elemental atomic numbers, periodic groups and periods, composition fractions, mean atomic number, and the atomic-number difference. Adsorption-site identity, relaxed coordinates, and DFT energies are not inputs. The model supports pure metals (A1), AB alloys (L1₀), and A₃B/AB₃ alloys (L1₂). Omitted facets default to 111 for A1/L1₂ and 101 for L1₀.

For the course dataset deliverable, copy the locally generated CSV to `data/release/surfaces.csv` and keep its [provenance note](data/release/README.md). The data file is not included in the starter package because the actual records were downloaded on my local computer. Raw API responses and credentials are not committed.

## Model and results

I split surfaces by **unordered metal pair**, keeping a given pair out of the other partitions. Seed 42 yielded 1,198 training, 253 validation, and 260 test rows. The imputer, scaler, and categorical encoder were fitted to training data only. The PyTorch multilayer perceptron (MLP) has hidden layers of 64 and 32 units, ReLU, dropout 0.1, AdamW, and validation-MAE early stopping. Ridge regression, random forest, and histogram gradient boosting used the same split.

| Regressor | Validation MAE (eV) | Test MAE (eV) | Test RMSE (eV) | Test R² |
| --- | ---: | ---: | ---: | ---: |
| MLP | **0.301** | 0.373 | 0.620 | 0.855 |
| Ridge | 0.803 | 0.839 | 1.082 | 0.559 |
| Random forest | 0.303 | 0.365 | 0.591 | 0.869 |
| Histogram gradient boosting | 0.304 | **0.356** | **0.577** | **0.875** |

The MLP had the lowest validation MAE by less than 0.004 eV; histogram gradient boosting had the lowest observed test MAE. These are results from **one grouped split**. I have not estimated variation across splits or validated new surfaces by DFT. The prediction command below loads the MLP. Its test MAE of 0.373 eV supports first-pass comparison, not a claim that any individual prediction is accurate to that tolerance.

## Reproduce the run

Use Python 3.11 or 3.12. In Git Bash on Windows, from the repository root:

```bash
py -3.12 -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Obtain a personal key from the [Catalysis-Hub sign-in page](https://api-catalysis-hub.slac.stanford.edu/auth/login), then enter it locally:

```bash
read -rs -p 'CATHUB_API_KEY: ' CATHUB_API_KEY; echo
export CATHUB_API_KEY
python -m oxygen_ml.data fetch --output data/raw/cathub_o.json
unset CATHUB_API_KEY
python -m oxygen_ml.data prepare --input data/raw/cathub_o.json --output data/processed/surfaces.csv
python -m oxygen_ml.train --data data/processed/surfaces.csv --output artifacts/run1
```

The fetch command writes its JSON only when all pages arrive. Review `surfaces.summary.json` after preparation. To place the curated dataset and filtering counts in the public course repository, check their contents and then run `cp data/processed/surfaces.csv data/processed/surfaces.summary.json data/release/`.

## Predict candidate surfaces

`examples/candidates.csv` shows the format. For a real screening list, create a new UTF-8 CSV with one `surface_formula` per row and optional `facet`:

```bash
python -m oxygen_ml.predict --model-dir artifacts/run1 --candidates examples/candidates.csv --output artifacts/run1/predictions.csv
```

The output gives predicted O reaction energy and flags for `already_in_source_dataset`, `new_metal_pair`, and `outside_training_domain`. A surface using familiar elements may still contain a new pair, so inspect both novelty flags. Exclude already-labeled surfaces from claims of novel screening. If a reaction-specific target energy has independent justification, `--target-ev VALUE` sorts candidates by distance to it. Stronger binding alone does not establish better catalytic activity.

The saved MLP needs `model.pt`, `preprocessor.joblib`, and `metadata.json` from the **same run**. The four-row example run verifies the prediction pipeline; it does not validate new catalysts. To share a ready-to-run copy for the course, place those files and `metrics.json` in `models/release/` as described in its [README](models/release/README.md). The source package cannot include the weights because they were trained on my local computer.

## Repository contents

| Path | Purpose |
| --- | --- |
| `src/oxygen_ml/` and `tests/` | Model pipeline and synthetic checks |
| `data/release/` | Curated dataset and provenance note after the local copy step |
| `models/release/` | Fitted MLP bundle and metrics after the local copy step |
| `examples/candidates.csv` | Input-format example |
| `deliverables/Final_Report.docx` | IEEE-format report |
| `deliverables/Final_Poster.pptx` | Poster based on the supplied USC template |
| `docs/MODEL_CARD.md` | Intended use and limitations |

`data/raw/`, `data/processed/`, `.venv/`, and `artifacts/` stay local by default. The reviewed copies under `data/release/` and `models/release/` can be committed for the course. Never commit an API key. The [Git Bash guide](docs/WINDOWS_GITBASH.md) explains local setup and pushing selected files.

## References

1. O. Mamun, K. T. Winther, J. R. Boes, and T. Bligaard, “High-throughput calculations of catalytic properties of bimetallic alloy surfaces,” *Scientific Data* **6**, 76 (2019), https://doi.org/10.1038/s41597-019-0080-z.
2. K. T. Winther *et al.*, “Catalysis-Hub.org, an open electronic structure database for surface reactions,” *Scientific Data* **6**, 75 (2019), https://doi.org/10.1038/s41597-019-0081-y.
3. A. Paszke *et al.*, “PyTorch: An Imperative Style, High-Performance Deep Learning Library,” *NeurIPS* **32** (2019).
4. F. Pedregosa *et al.*, “Scikit-learn: Machine Learning in Python,” *JMLR* **12**, 2825–2830 (2011).
