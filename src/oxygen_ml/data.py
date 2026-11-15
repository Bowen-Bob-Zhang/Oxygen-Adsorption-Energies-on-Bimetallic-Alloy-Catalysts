"""Fetch and prepare the MamunHighT2019 oxygen adsorption reactions.

Run ``python -m oxygen_ml.data fetch --output data/raw/cathub_o.json`` after
setting ``CATHUB_API_KEY``. Processing is offline and never needs the API key:
``python -m oxygen_ml.data prepare --input data/raw/cathub_o.json
--output data/processed/surfaces.csv``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from oxygen_ml.features import annotate_candidates, canonical_surface_formula


API_URL = "https://api-catalysis-hub.slac.stanford.edu/graphql"
PUBLICATION_ID = "MamunHighT2019"
PAGE_SIZE = 200  # The official CatHub client notes a server cap of 200.
PREPARED_COLUMNS = [
    "surface_formula", "facet", "structure", "metal_pair", "target_ev", "n_sites"
]


def _query(after: str | None) -> str:
    cursor = f", after: {json.dumps(after)}" if after is not None else ""
    return """{
      reactions(first: %d%s, pubId: %s, products: "O") {
        totalCount
        pageInfo { hasNextPage endCursor }
        edges { node {
          surfaceComposition facet sites coverages reactants products
          reactionEnergy pubId
        } }
      }
    }""" % (PAGE_SIZE, cursor, json.dumps(PUBLICATION_ID))


def _post_graphql(query: str, key: str) -> dict:
    request = Request(
        API_URL,
        data=json.dumps({"query": query}).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-API-Key": key},
        method="POST",
    )
    for attempt in range(3):
        try:
            with urlopen(request, timeout=45) as response:
                data = json.load(response)
            break
        except HTTPError as error:
            if error.code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(20 * (attempt + 1))
                continue
            raise RuntimeError(f"CatHub GraphQL request failed: HTTP {error.code}") from error
        except URLError as error:
            raise RuntimeError(f"CatHub GraphQL request failed: {error}") from error
    if data.get("errors"):
        raise RuntimeError(f"CatHub GraphQL rejected the query: {data['errors']}")
    if not isinstance(data.get("data", {}).get("reactions"), dict):
        raise RuntimeError("CatHub returned no reactions connection")
    return data["data"]["reactions"]


def fetch_reactions(output_path: Path, api_key: str | None = None,
                    delay_seconds: float = 6.5) -> dict:
    """Page through publication O records and write their untouched fields.

    This is the only function that contacts CatHub. A key is required by the
    current official service; it is never written to disk or printed.
    """
    key = api_key or os.environ.get("CATHUB_API_KEY")
    if delay_seconds < 0:
        raise ValueError("delay_seconds must not be negative")
    if not key:
        raise ValueError(
            "CatHub requires an API key. Set CATHUB_API_KEY (get a key via "
            "https://api-catalysis-hub.slac.stanford.edu/auth/login)."
        )
    rows: list[dict] = []
    cursor = None
    seen_cursors = set()
    page_count = 0
    total_count = None
    while True:
        connection = _post_graphql(_query(cursor), key)
        edges = connection.get("edges") or []
        if not isinstance(edges, list):
            raise RuntimeError("Unexpected CatHub edges response")
        rows.extend(edge["node"] for edge in edges)
        page_count += 1
        total_count = connection.get("totalCount")
        info = connection.get("pageInfo") or {}
        if not info.get("hasNextPage"):
            break
        next_cursor = info.get("endCursor")
        if not edges or not next_cursor or next_cursor in seen_cursors:
            raise RuntimeError("CatHub pagination stopped making progress")
        seen_cursors.add(next_cursor)
        cursor = next_cursor
        time.sleep(delay_seconds)
    payload = {
        "source": API_URL,
        "publication_id": PUBLICATION_ID,
        "server_filter": {"products": "O"},
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "pages": page_count,
        "total_count": total_count,
        "reactions": rows,
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(target)
    return payload


def _mapping(value: object) -> dict:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def _same_stoichiometry(value: object, expected: dict[str, float]) -> bool:
    actual = _mapping(value)
    if set(actual) != set(expected):
        return False
    try:
        return all(
            not isinstance(actual[name], bool)
            and math.isclose(float(actual[name]), expected[name], rel_tol=0, abs_tol=1e-8)
            for name in expected
        )
    except (TypeError, ValueError):
        return False


def _matching_coverage(value: object, expected: float) -> bool:
    mapping = _mapping(value)
    # CatHub has used both O and Ostar as an adsorbate key. An additional
    # adsorbate in this mapping would change the physical target, so exclude it.
    if set(mapping) not in ({"O"}, {"Ostar"}):
        return False
    try:
        amount = next(iter(mapping.values()))
        return not isinstance(amount, bool) and math.isclose(
            float(amount), expected, rel_tol=0, abs_tol=1e-6
        )
    except (TypeError, ValueError):
        return False


def _raw_rows(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError("Expected a CatHub JSON object or a reaction list")
    data = payload.get("data", payload)
    rows = data.get("reactions")
    if isinstance(rows, dict):
        rows = [edge["node"] for edge in rows.get("edges", [])]
    if not isinstance(rows, list):
        raise ValueError("Input does not contain a reactions list")
    return rows


def prepare_reactions(
    input_path: Path, output_path: Path, coverage: float = 0.25
) -> pd.DataFrame:
    """Select one physical target and minimize energy across adsorption sites.

    Target: H2O(g) - H2(g) + * -> O* at specified O coverage, in eV. One
    surface/facet contributes exactly one label, the lowest valid site energy.
    ``n_sites`` counts distinct site names if available, otherwise records.
    """
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    groups: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(dict)
    counts = Counter()
    seen_records = set()
    for index, row in enumerate(_raw_rows(payload)):
        counts["input_records"] += 1
        if not isinstance(row, dict):
            counts["invalid_record"] += 1
            continue
        # The server's products="O" filter is intentionally not trusted as
        # the full target definition; different reference reactions coexist.
        if row.get("pubId", PUBLICATION_ID) != PUBLICATION_ID:
            counts["other_publication"] += 1
            continue
        if not _same_stoichiometry(
            row.get("reactants"), {"H2Ogas": 1, "H2gas": -1, "star": 1}
        ) or not _same_stoichiometry(row.get("products"), {"Ostar": 1}):
            counts["other_reaction"] += 1
            continue
        if not _matching_coverage(row.get("coverages"), coverage):
            counts["other_or_missing_coverage"] += 1
            continue
        try:
            energy = float(row["reactionEnergy"])
        except (KeyError, TypeError, ValueError):
            counts["missing_or_nonfinite_energy"] += 1
            continue
        if not math.isfinite(energy):
            counts["missing_or_nonfinite_energy"] += 1
            continue
        formula = row.get("surfaceComposition")
        if not formula:
            counts["unsupported_surface"] += 1
            continue
        try:
            annotated = annotate_candidates(
                pd.DataFrame([{"surface_formula": formula, "facet": row.get("facet")}])
            ).iloc[0]
        except ValueError:
            # Other surface compositions in the same publication cannot be
            # represented by the A1/L10/L12 model and are excluded.
            counts["unsupported_surface"] += 1
            continue
        key = (
            canonical_surface_formula(annotated["surface_formula"]), annotated["facet"],
            annotated["structure"], annotated["metal_pair"],
        )
        # A repeated edge/page is not an additional observation.
        fingerprint = json.dumps(row, sort_keys=True, ensure_ascii=False)
        if fingerprint in seen_records:
            counts["duplicate_record"] += 1
            continue
        seen_records.add(fingerprint)
        counts["valid_site_records"] += 1
        sites = _mapping(row.get("sites"))
        site_name = json.dumps(sites, sort_keys=True) if sites else f"record-{index}"
        previous = groups[key].get(site_name)
        groups[key][site_name] = energy if previous is None else min(previous, energy)
    result = pd.DataFrame(
        [
            (*key, min(site_energies.values()), len(site_energies))
            for key, site_energies in sorted(groups.items())
        ],
        columns=PREPARED_COLUMNS,
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(target, index=False)
    summary = {
        "publication_id": PUBLICATION_ID,
        "reference_reaction": "H2O(g) - H2(g) + * -> O*",
        "coverage": coverage,
        "surface_labels": len(result),
        "site_count_total": int(result["n_sites"].sum()),
        "record_counts": dict(counts),
    }
    target.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="Download CatHub records with CATHUB_API_KEY")
    fetch.add_argument("--output", required=True, type=Path)
    fetch.add_argument("--delay-seconds", type=float, default=6.5,
                       help="Pause between pages to avoid overloading the API")
    prepare = commands.add_parser("prepare", help="Prepare an offline surface target CSV")
    prepare.add_argument("--input", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--coverage", type=float, default=0.25)
    args = parser.parse_args()
    if args.command == "fetch":
        data = fetch_reactions(args.output, delay_seconds=args.delay_seconds)
        print(f"Fetched {len(data['reactions'])} records in {data['pages']} pages to {args.output}")
    else:
        frame = prepare_reactions(args.input, args.output, coverage=args.coverage)
        print(f"Prepared {len(frame)} surface labels in {args.output}")


if __name__ == "__main__":
    main()
