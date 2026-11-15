# Model card: surface-level O adsorption regression

## Target and data unit

- One row per ideal pure-metal or binary-alloy surface, identified by composition, structure and facet.
- Target: the minimum valid site-level `reactionEnergy` in eV for `H2O(g) - H2(g) + * -> O*` on that surface, after the dataset's reconstruction and coverage checks.
- The minimum is over **observed valid** DFT site calculations, not a guarantee that every possible site converged.

## Inputs available at screening time

The element symbols and fractions determine deterministic atomic-number, group and period descriptors. Structure (A1, L10 or L12) and ideal facet are also included. Relaxed O site, DFT total energies and quantities derived from adsorption calculations are excluded.

## Evaluation

Use disjoint unordered element pairs for train, validation and held-out test. Fit preprocessing only on train. Pick the best epoch by validation MAE and inspect final test MAE/RMSE/R² once. Compare with Ridge, random forest and histogram gradient boosting on the same split. The split tests new pair combinations of known elements; it does not prove extrapolation to new elements, other structures or catalytic activity.

## Limits

The dataset is small for neural networks, and its calculated energies inherit the source DFT functional and slab setup. This first version returns point predictions only. It does not estimate a calibrated uncertainty interval or infer experimental performance under operating conditions. Check candidate domain flags, make a reaction-specific screening criterion, then validate selected surfaces with DFT.

The source publication already enumerates ideal A1/L10/L12 surfaces for combinations of its 37 elements. Many candidate formulas in that closed space will therefore have existing DFT labels. The inference CSV marks already-labeled surfaces; truly unmeasured candidates may lie outside the model's observed element/structure domain, where errors are less certain.
