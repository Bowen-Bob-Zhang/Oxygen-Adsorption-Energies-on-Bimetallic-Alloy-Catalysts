"""Train a compact MLP and three tabular baselines on the same grouped split."""

import argparse
import copy
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import (CATEGORICAL_FEATURES, NUMERIC_FEATURES,
                       annotate_candidates, canonical_surface_key, make_feature_frame)


def grouped_split(df: pd.DataFrame, seed: int = 42):
    """Reserve whole unordered element pairs for each disjoint partition."""
    if df['metal_pair'].nunique() < 10:
        raise ValueError('At least 10 distinct metal pairs are needed for a meaningful split.')
    groups = df['metal_pair'].to_numpy()
    first = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train, remaining = next(first.split(df, groups=groups))
    second = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1)
    val_rel, test_rel = next(
        second.split(df.iloc[remaining], groups=groups[remaining])
    )
    val, test = remaining[val_rel], remaining[test_rel]
    partitions = [set(groups[index]) for index in (train, val, test)]
    assert not (partitions[0] & partitions[1] or partitions[0] & partitions[2]
                or partitions[1] & partitions[2])
    return train, val, test


def make_preprocessor():
    numeric = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
    ])
    categories = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    return ColumnTransformer([
        ('numeric', numeric, NUMERIC_FEATURES),
        ('categorical', categories, CATEGORICAL_FEATURES),
    ], sparse_threshold=0)


def metrics(truth, predicted):
    return {
        'mae_ev': float(mean_absolute_error(truth, predicted)),
        'rmse_ev': float(np.sqrt(mean_squared_error(truth, predicted))),
        'r2': float(r2_score(truth, predicted)),
    }


def fit_neural_net(x_train, y_train, x_val, y_val, args):
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise SystemExit('PyTorch is missing. Run: python -m pip install -e .') from exc

    from .model import SurfaceMLP

    torch.manual_seed(args.seed)
    model = SurfaceMLP(input_dim=x_train.shape[1], dropout=args.dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    loss_fn = torch.nn.SmoothL1Loss()
    mean = float(np.mean(y_train))
    std = float(np.std(y_train))
    if std <= 0:
        raise ValueError('Training target has zero variance.')

    train_x = torch.from_numpy(x_train)
    train_y = torch.from_numpy(((y_train - mean) / std).astype('float32'))
    val_x = torch.from_numpy(x_val)
    batches = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size,
                         shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    best_state = None
    best_val_mae = float('inf')
    best_epoch = 0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_x, batch_y in batches:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = model(val_x).numpy() * std + mean
        val_mae = float(mean_absolute_error(y_val, prediction))
        if val_mae < best_val_mae - 1e-5:
            best_val_mae, best_epoch = val_mae, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {'target_mean': mean, 'target_std': std,
                   'best_epoch': best_epoch, 'epochs_run': epoch,
                   'input_dim': int(x_train.shape[1]), 'dropout': args.dropout}


def neural_predictions(model, x, target_mean, target_std):
    import torch
    model.eval()
    with torch.no_grad():
        scaled = model(torch.from_numpy(x)).numpy()
    return scaled * target_std + target_mean


def run(args):
    import torch

    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError('epochs, patience, and batch_size must be positive.')
    random.seed(args.seed)
    np.random.seed(args.seed)
    data = pd.read_csv(args.data)
    needed = {'surface_formula', 'facet', 'structure', 'metal_pair', 'target_ev'}
    if missing := needed.difference(data.columns):
        raise ValueError(f'Prepared dataset is missing columns: {sorted(missing)}')
    data['target_ev'] = pd.to_numeric(data['target_ev'], errors='coerce')
    if not np.isfinite(data['target_ev']).all():
        raise ValueError('Prepared target column has missing or non-finite energies.')
    canonical = annotate_candidates(data)
    keys = [canonical_surface_key(formula, facet) for formula, facet in
            zip(canonical['surface_formula'], canonical['facet'])]
    if len(keys) != len(set(keys)):
        raise ValueError('Prepared dataset must contain one row per surface.')
    for column in ('facet', 'structure', 'metal_pair'):
        if (data[column].astype(str).to_numpy()
                != canonical[column].astype(str).to_numpy()).any():
            raise ValueError(f'Prepared {column} disagrees with the surface formula.')
    data = canonical

    features = make_feature_frame(data)
    train_idx, val_idx, test_idx = grouped_split(data, args.seed)
    y = data['target_ev'].to_numpy(dtype='float64')
    preprocessor = make_preprocessor()
    x_train = preprocessor.fit_transform(features.iloc[train_idx]).astype('float32')
    x_val = preprocessor.transform(features.iloc[val_idx]).astype('float32')
    x_test = preprocessor.transform(features.iloc[test_idx]).astype('float32')

    model, config = fit_neural_net(x_train, y[train_idx], x_val, y[val_idx], args)
    results = {'mlp': {
        'validation': metrics(y[val_idx], neural_predictions(model, x_val,
                                    config['target_mean'], config['target_std'])),
        'test': metrics(y[test_idx], neural_predictions(model, x_test,
                              config['target_mean'], config['target_std'])),
    }}
    baselines = {
        'ridge': Ridge(alpha=10.0),
        'random_forest': RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                               n_jobs=-1, random_state=args.seed),
        'hist_gradient_boosting': HistGradientBoostingRegressor(max_iter=200,
                                     max_leaf_nodes=15, l2_regularization=1.0,
                                     random_state=args.seed),
    }
    for name, baseline in baselines.items():
        baseline.fit(x_train, y[train_idx])
        results[name] = {
            'validation': metrics(y[val_idx], baseline.predict(x_val)),
            'test': metrics(y[test_idx], baseline.predict(x_test)),
        }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / 'model.pt')
    joblib.dump(preprocessor, output / 'preprocessor.joblib')
    split_labels = np.full(len(data), '', dtype=object)
    for index, label in ((train_idx, 'train'), (val_idx, 'validation'),
                         (test_idx, 'test')):
        split_labels[index] = label
    data.assign(split=split_labels).to_csv(output / 'splits.csv', index=False)
    metadata = {
        **config,
        'seed': args.seed,
        'numeric_features': NUMERIC_FEATURES,
        'categorical_features': CATEGORICAL_FEATURES,
        'seen_metals': sorted(set('|'.join(data.iloc[train_idx]['metal_pair']).split('|'))),
        'seen_structures': sorted(set(data.iloc[train_idx]['structure'])),
        'seen_facets': sorted(set(data.iloc[train_idx]['facet'].astype(str))),
        'train_metal_pairs': sorted(set(data.iloc[train_idx]['metal_pair'])),
        'known_surface_keys': sorted(set(
            canonical_surface_key(formula, facet)
            for formula, facet in zip(data['surface_formula'], data['facet'])
        )),
        'target': 'minimum valid atomic O reactionEnergy on each surface, eV',
        'prediction_model': 'mlp',
        'reference_reaction': 'H2O(g) - H2(g) + * -> O*',
        'sklearn_version': sklearn.__version__,
        'torch_version': torch.__version__,
        'counts': {name: int(len(index)) for name, index in (
            ('train', train_idx), ('validation', val_idx), ('test', test_idx))},
    }
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (output / 'metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    winner = min(results, key=lambda name: results[name]['validation']['mae_ev'])
    print(json.dumps({'rows': len(data), 'split_counts': metadata['counts'],
                      'prediction_model': 'mlp',
                      'lowest_validation_mae_model': winner,
                      'metrics': results}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/processed/surfaces.csv')
    parser.add_argument('--output', default='artifacts/run1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.1)
    run(parser.parse_args())


if __name__ == '__main__':
    main()

