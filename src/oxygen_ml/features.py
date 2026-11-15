"""Composition and surface descriptors available before a DFT calculation.

These descriptors deliberately exclude adsorption sites, relaxed coordinates and
other outputs of the calculation whose adsorption energy is being predicted.
"""

from __future__ import annotations

import re

import pandas as pd


# Indexed by atomic number. The f-block elements are retained for formula
# validation, but rejected below because their group is not unambiguous.
_PERIODS = (
    "H He",
    "Li Be B C N O F Ne",
    "Na Mg Al Si P S Cl Ar",
    "K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr",
    "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe",
    "Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn",
    "Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og",
)
_GROUPS = (
    (1, 18),
    (1, 2, 13, 14, 15, 16, 17, 18),
    (1, 2, 13, 14, 15, 16, 17, 18),
    tuple(range(1, 19)),
    tuple(range(1, 19)),
    (1, 2, 3) + (None,) * 14 + tuple(range(4, 19)),
    (1, 2, 3) + (None,) * 14 + tuple(range(4, 19)),
)
_ELEMENTS: dict[str, tuple[int, int, int | None]] = {}
_number = 0
for _period, (_symbols, _groups) in enumerate(zip(_PERIODS, _GROUPS), start=1):
    for _symbol, _group in zip(_symbols.split(), _groups, strict=True):
        _number += 1
        _ELEMENTS[_symbol] = (_number, _period, _group)

_FORMULA_PART = re.compile(r"([A-Z][a-z]?)([1-9][0-9]*)?")

NUMERIC_FEATURES = [
    "atomic_number_a", "atomic_number_b",
    "group_a", "group_b", "period_a", "period_b",
    "fraction_a", "fraction_b", "mean_atomic_number",
    "absolute_atomic_number_difference",
]
CATEGORICAL_FEATURES = ["structure", "facet"]


def parse_surface_formula(formula: str) -> dict[str, int]:
    """Parse an element formula such as ``Pt3Ni`` (no parentheses or tags)."""
    if not isinstance(formula, str) or not formula:
        raise ValueError(f"Expected a nonempty surface formula, got {formula!r}")
    amounts: dict[str, int] = {}
    position = 0
    for match in _FORMULA_PART.finditer(formula):
        if match.start() != position:
            raise ValueError(f"Invalid surface formula: {formula!r}")
        symbol = match.group(1)
        if symbol not in _ELEMENTS:
            raise ValueError(f"Unknown element {symbol!r} in {formula!r}")
        amounts[symbol] = amounts.get(symbol, 0) + int(match.group(2) or 1)
        position = match.end()
    if position != len(formula) or not 1 <= len(amounts) <= 2:
        raise ValueError(f"Expected a pure element or binary alloy formula: {formula!r}")
    for symbol in amounts:
        if _ELEMENTS[symbol][2] is None:
            raise ValueError(f"No unambiguous periodic group for {symbol!r}")
    return amounts


def canonical_surface_formula(formula: str) -> str:
    """Normalize element order while retaining each element's amount."""
    amounts = parse_surface_formula(formula)
    return ''.join(
        symbol + (str(amounts[symbol]) if amounts[symbol] != 1 else '')
        for symbol in sorted(amounts)
    )


def canonical_surface_key(formula: str, facet: str) -> str:
    """Ignore formula token order while retaining which metal has 3/4 share."""
    return canonical_surface_formula(formula) + '|' + str(facet)


def _identity(formula: str) -> tuple[dict[str, int], str, str]:
    amounts = parse_surface_formula(formula)
    symbols = sorted(amounts)
    if len(symbols) == 1:
        return amounts, "A1", f"{symbols[0]}|{symbols[0]}"
    counts = sorted(amounts.values())
    structure = {(1, 1): "L10", (1, 3): "L12"}.get(tuple(counts))
    if structure is None:
        raise ValueError(
            f"Unsupported stoichiometry for {formula!r}; expected AB or A3B/AB3"
        )
    return amounts, structure, "|".join(symbols)


def _normalize_facet(value: object, structure: str) -> str:
    if pd.isna(value) or str(value).strip() == "":
        # The MamunHighT2019 conventional slab indices; callers can always
        # provide the precise facet explicitly when their cell convention differs.
        return "101" if structure == "L10" else "111"
    facet = str(value).strip().strip("()").replace(" ", "")
    if not re.fullmatch(r"[0-9]+", facet):
        raise ValueError(f"Invalid facet {value!r}; use a Miller-index string")
    return facet


def annotate_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Add inferred structure, unordered metal pair and, if missing, facet.

    ``surface_formula`` is mandatory. Supplying ``facet`` overrides the default
    conventional slab index (111 for A1/L12; 101 for L10).
    """
    if "surface_formula" not in df.columns:
        raise ValueError("Candidate table needs a surface_formula column")
    out = df.copy()
    identities = [_identity(value) for value in out["surface_formula"]]
    inferred_structure = [entry[1] for entry in identities]
    if "structure" in out.columns:
        for actual, expected, formula in zip(
            out["structure"], inferred_structure, out["surface_formula"]
        ):
            if not pd.isna(actual) and str(actual).strip() and str(actual).upper() != expected:
                raise ValueError(
                    f"Structure {actual!r} disagrees with stoichiometry of {formula!r} ({expected})"
                )
    out["structure"] = inferred_structure
    out["metal_pair"] = [entry[2] for entry in identities]
    facet_values = out["facet"] if "facet" in out.columns else [None] * len(out)
    out["facet"] = [
        _normalize_facet(facet, structure)
        for facet, structure in zip(facet_values, inferred_structure)
    ]
    return out


def make_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build a stable feature table for both fitting and inference.

    The alphabetical element order fixes the meaning of fractions for AB versus
    A3B/AB3, while ``metal_pair`` remains available outside the model for
    grouped validation.
    """
    annotated = annotate_candidates(df)
    rows = []
    for formula in annotated["surface_formula"]:
        amounts = parse_surface_formula(formula)
        names = sorted(amounts)
        if len(names) == 1:
            names.append(names[0])
        a, b = names
        z_a, period_a, group_a = _ELEMENTS[a]
        z_b, period_b, group_b = _ELEMENTS[b]
        total = sum(amounts.values())
        fraction_a = amounts[a] / total if a != b else 1.0
        fraction_b = amounts[b] / total if a != b else 0.0
        rows.append({
            "atomic_number_a": z_a, "atomic_number_b": z_b,
            "group_a": group_a, "group_b": group_b,
            "period_a": period_a, "period_b": period_b,
            "fraction_a": fraction_a, "fraction_b": fraction_b,
            "mean_atomic_number": fraction_a * z_a + fraction_b * z_b,
            "absolute_atomic_number_difference": abs(z_a - z_b),
        })
    features = pd.DataFrame(rows, index=annotated.index, columns=NUMERIC_FEATURES)
    for column in CATEGORICAL_FEATURES:
        features[column] = annotated[column].astype("string")
    return features[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
