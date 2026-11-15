import unittest

import pandas as pd

from oxygen_ml.train import grouped_split, make_preprocessor
from oxygen_ml.features import make_feature_frame


class SplitAndPreprocessingTests(unittest.TestCase):
    def test_all_surfaces_of_each_metal_pair_stay_together(self):
        data = pd.DataFrame([
            {'surface_formula': f'{a}{b}', 'facet': '101', 'structure': 'L10',
             'metal_pair': f'{a}|{b}', 'target_ev': float(i), 'n_sites': 1}
            for i, (a, b) in enumerate([
                ('Pt', 'Ni'), ('Cu', 'Ag'), ('Co', 'Fe'), ('Ru', 'Mo'),
                ('Pd', 'Au'), ('Zn', 'Cd'), ('Rh', 'Ir'), ('Al', 'Ga'),
                ('Sn', 'Pb'), ('Ti', 'V'), ('Nb', 'Ta'), ('W', 'Re'),
                ('Os', 'Mn'), ('Zr', 'Hf'), ('Sc', 'Y'), ('In', 'Bi'),
            ])
        ] + [
            {'surface_formula': 'Pt3Ni', 'facet': '111', 'structure': 'L12',
             'metal_pair': 'Pt|Ni', 'target_ev': -1.0, 'n_sites': 2}
        ])
        train, val, test = grouped_split(data, seed=12)
        pairs = [set(data.iloc[index].metal_pair) for index in (train, val, test)]
        self.assertFalse(pairs[0] & pairs[1])
        self.assertFalse(pairs[0] & pairs[2])
        self.assertFalse(pairs[1] & pairs[2])
        self.assertEqual(len(train) + len(val) + len(test), len(data))

    def test_feature_pipeline_is_fitted_only_to_supplied_training_rows(self):
        train = pd.DataFrame({'surface_formula': ['Pt', 'Pt3Ni'],
                              'facet': ['111', '111'],
                              'structure': ['A1', 'L12']})
        candidate = pd.DataFrame({'surface_formula': ['PtNi'],
                                  'facet': ['101'], 'structure': ['L10']})
        transformer = make_preprocessor()
        transformer.fit(make_feature_frame(train))
        columns_before = list(transformer.get_feature_names_out())
        transformed = transformer.transform(make_feature_frame(candidate))
        self.assertEqual(transformed.shape[1], len(columns_before))
        self.assertEqual(columns_before, list(transformer.get_feature_names_out()))


if __name__ == '__main__':
    unittest.main()
