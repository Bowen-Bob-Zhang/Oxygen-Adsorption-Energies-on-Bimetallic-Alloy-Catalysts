"""Apply a saved model and feature pipeline to proposed alloy surfaces."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import annotate_candidates, canonical_surface_key, make_feature_frame


def run(args):
    try:
        import torch
    except ImportError as exc:
        raise SystemExit('PyTorch is missing. Run: python -m pip install -e .') from exc
    from .model import SurfaceMLP

    artifact = Path(args.model_dir)
    metadata = json.loads((artifact / 'metadata.json').read_text(encoding='utf-8'))
    preprocessor = joblib.load(artifact / 'preprocessor.joblib')
    candidates = annotate_candidates(pd.read_csv(args.candidates))
    x = preprocessor.transform(make_feature_frame(candidates)).astype('float32')
    if x.shape[1] != metadata['input_dim']:
        raise ValueError('Saved preprocessing dimensions do not match model weights.')
    model = SurfaceMLP(x.shape[1], dropout=metadata['dropout'])
    model.load_state_dict(torch.load(artifact / 'model.pt', map_location='cpu',
                                     weights_only=True))
    model.eval()
    with torch.no_grad():
        predictions = model(torch.from_numpy(x)).numpy()
    candidates['predicted_o_adsorption_ev'] = (
        predictions * metadata['target_std'] + metadata['target_mean']
    )
    seen_metals = set(metadata['seen_metals'])
    seen_structures = set(metadata['seen_structures'])
    seen_facets = set(metadata['seen_facets'])
    trained_pairs = set(metadata['train_metal_pairs'])
    candidates['new_metal_pair'] = ~candidates['metal_pair'].isin(trained_pairs)
    surface_keys = [canonical_surface_key(formula, facet) for formula, facet in
                    zip(candidates['surface_formula'], candidates['facet'])]
    known_keys = set(metadata['known_surface_keys'])
    candidates['already_in_source_dataset'] = [
        key in known_keys for key in surface_keys
    ]
    candidates['outside_training_domain'] = [
        bool(set(pair.split('|')) - seen_metals or structure not in seen_structures
             or str(facet) not in seen_facets)
        for pair, structure, facet in zip(candidates['metal_pair'],
                                          candidates['structure'], candidates['facet'])
    ]
    if args.target_ev is not None:
        candidates['distance_to_target_ev'] = np.abs(
            candidates['predicted_o_adsorption_ev'] - args.target_ev
        )
        candidates = candidates.sort_values('distance_to_target_ev')
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(destination, index=False)
    print(f'Wrote {len(candidates)} predictions to {destination}')
    if candidates['outside_training_domain'].any():
        print('Some candidates use elements/structures/facets absent from training; '
              'flagged predictions are extrapolations.')
    if candidates['already_in_source_dataset'].any():
        print('Some examples already have labels in the source dataset; '
              'exclude them when screening genuinely unmeasured surfaces.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', default='artifacts/run1')
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--output', default='artifacts/run1/predictions.csv')
    parser.add_argument('--target-ev', type=float, default=None,
                        help='If supplied, sort by closeness to this chosen energy in eV.')
    run(parser.parse_args())


if __name__ == '__main__':
    main()

