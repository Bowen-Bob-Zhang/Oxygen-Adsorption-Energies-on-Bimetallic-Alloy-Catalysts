"""Small synthetic fixtures verify the extraction rules without a CatHub key."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from oxygen_ml.data import fetch_reactions, prepare_reactions
from oxygen_ml.features import (
    CATEGORICAL_FEATURES, NUMERIC_FEATURES,
    annotate_candidates, canonical_surface_key, make_feature_frame,
)


def reaction(**changes):
    row = {
        "pubId": "MamunHighT2019",
        "surfaceComposition": "Pt3Ni",
        "facet": "111",
        "sites": {"O": "hollow-A"},
        "coverages": {"O": 0.25},
        "reactants": {"H2Ogas": 1, "H2gas": -1, "star": 1},
        "products": {"Ostar": 1},
        "reactionEnergy": -1.1,
    }
    row.update(changes)
    return row


class CatHubPreparationTests(unittest.TestCase):
    def test_exact_reaction_coverage_energy_and_minimum_site(self):
        rows = [
            reaction(),
            reaction(sites={"O": "bridge-B"}, reactionEnergy=-1.8),
            reaction(sites={"O": "hollow-A"}, reactionEnergy=-1.5),
            reaction(),  # repeated GraphQL edge must not add a site
            reaction(reactants={"O2gas": 0.5, "star": 1}, reactionEnergy=-100),
            reaction(products={"OHstar": 1}, reactionEnergy=-100),
            reaction(coverages={"O": 0.5}, reactionEnergy=-100),
            reaction(coverages={}, reactionEnergy=-100),
            reaction(reactionEnergy="NaN"),
            reaction(pubId="OtherPaper", reactionEnergy=-100),
            reaction(surfaceComposition="PtNi", facet="101", sites={"O": "top"},
                     coverages=json.dumps({"Ostar": 0.25}),
                     reactants=json.dumps({"H2Ogas": 1, "H2gas": -1, "star": 1}),
                     products=json.dumps({"Ostar": 1}), reactionEnergy=-0.8),
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "raw.json"
            output = Path(directory) / "surfaces.csv"
            source.write_text(json.dumps({"reactions": rows}), encoding="utf-8")
            prepared = prepare_reactions(source, output)
            self.assertEqual(len(prepared), 2)
            l12 = prepared.loc[prepared.surface_formula == "NiPt3"].iloc[0]
            self.assertEqual(l12.target_ev, -1.8)
            self.assertEqual(l12.n_sites, 2)
            self.assertEqual(l12.metal_pair, "Ni|Pt")
            self.assertEqual(l12.structure, "L12")
            self.assertEqual(pd.read_csv(output).shape[0], 2)
            summary = json.loads(output.with_suffix(".summary.json").read_text())
            self.assertEqual(summary["record_counts"]["other_reaction"], 2)
            self.assertEqual(summary["record_counts"]["duplicate_record"], 1)
            self.assertEqual(summary["site_count_total"], 3)

    def test_no_key_does_not_issue_request_or_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "raw.json"
            with patch.dict("os.environ", {"CATHUB_API_KEY": ""}), \
                    patch("oxygen_ml.data._post_graphql") as post:
                with self.assertRaisesRegex(ValueError, "CATHUB_API_KEY"):
                    fetch_reactions(output)
                post.assert_not_called()
            self.assertFalse(output.exists())

    def test_formula_token_order_aggregates_one_surface(self):
        rows = [
            reaction(surfaceComposition="Pt3Ni", sites={"O": "hollow"},
                     reactionEnergy=-1.2),
            reaction(surfaceComposition="NiPt3", sites={"O": "bridge"},
                     reactionEnergy=-1.7),
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "raw.json"
            output = Path(directory) / "surfaces.csv"
            source.write_text(json.dumps({"reactions": rows}), encoding="utf-8")
            prepared = prepare_reactions(source, output)
            self.assertEqual(len(prepared), 1)
            self.assertEqual(prepared.iloc[0].surface_formula, "NiPt3")
            self.assertEqual(prepared.iloc[0].target_ev, -1.7)
            self.assertEqual(prepared.iloc[0].n_sites, 2)

    def test_fetches_cursor_pages_and_saves_original_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "raw.json"
            first = {"edges": [{"node": reaction()}], "totalCount": 2,
                     "pageInfo": {"hasNextPage": True, "endCursor": "cursor-1"}}
            second = {"edges": [{"node": reaction(sites={"O": "top"})}],
                      "totalCount": 2,
                      "pageInfo": {"hasNextPage": False, "endCursor": "cursor-2"}}
            with patch("oxygen_ml.data._post_graphql", side_effect=[first, second]) as post:
                payload = fetch_reactions(output, api_key="fake-for-mock", delay_seconds=0)
            self.assertEqual(payload["pages"], 2)
            self.assertEqual(len(payload["reactions"]), 2)
            self.assertIn('first: 200', post.call_args_list[0].args[0])
            self.assertIn('after: "cursor-1"', post.call_args_list[1].args[0])
            self.assertNotIn("fake-for-mock", output.read_text())


class FeatureTests(unittest.TestCase):
    def test_stoichiometric_order_and_candidate_annotations(self):
        frame = annotate_candidates(pd.DataFrame({
            "surface_formula": ["Pt", "PtNi", "Pt3Ni", "Ni3Pt"],
            "facet": [None, None, None, "(211)"],
        }))
        self.assertEqual(frame.structure.tolist(), ["A1", "L10", "L12", "L12"])
        self.assertEqual(frame.facet.tolist(), ["111", "101", "111", "211"])
        self.assertEqual(frame.metal_pair.tolist(),
                         ["Pt|Pt", "Ni|Pt", "Ni|Pt", "Ni|Pt"])
        features = make_feature_frame(frame)
        self.assertEqual(features.columns.tolist(),
                         NUMERIC_FEATURES + CATEGORICAL_FEATURES)
        self.assertAlmostEqual(features.loc[2, "fraction_a"], 0.25)
        self.assertAlmostEqual(features.loc[3, "fraction_a"], 0.75)
        self.assertAlmostEqual(features.loc[2, "mean_atomic_number"],
                               0.25 * 28 + 0.75 * 78)

    def test_unsupported_formula_fails_loudly(self):
        for formula in ["Pt2Ni3", "Pt-Ni", "Xx3Ni", "Pt2NiO", "CeNi"]:
            with self.subTest(formula=formula):
                with self.assertRaises(ValueError):
                    annotate_candidates(pd.DataFrame({"surface_formula": [formula]}))

    def test_surface_identity_ignores_formula_order_not_stoichiometry(self):
        self.assertEqual(canonical_surface_key('Pt3Ni', '111'),
                         canonical_surface_key('NiPt3', '111'))
        self.assertNotEqual(canonical_surface_key('Pt3Ni', '111'),
                            canonical_surface_key('Ni3Pt', '111'))


if __name__ == "__main__":
    unittest.main()
