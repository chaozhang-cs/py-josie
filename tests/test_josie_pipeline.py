from __future__ import annotations

from pathlib import Path
import sys

import duckdb
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from josie_indexing import build_josie_index_polars
from josie_online import (
    TokenTable,
    rawTokenSet,
    searchMergeList,
    searchMergeProbeCostModelGreedy,
    searchProbeSetOptimized,
    set_total_number_of_sets,
)


@pytest.fixture(scope="module")
def sample_index() -> tuple[duckdb.DuckDBPyConnection, TokenTable]:
    """Create an in-memory index and token table shared across tests."""
    df1 = pd.DataFrame({"col1": ["banana", "apple", "cherry", "kiwi"]})
    df2 = pd.DataFrame({"col2": ["banana", "durian", "x", "apple", "pear", "kiwi"]})
    df3 = pd.DataFrame({"col3": ["x", "y", "z", "banana", "apple"]})
    df4 = pd.DataFrame({"col4": ["melon", "pear", "apple", "banana", "cherry"]})

    dataframes_with_id = [
        ("t1", df1),
        ("t2", df2),
        ("t3", df3),
        ("t4", df4),
    ]

    con = build_josie_index_polars(dataframes_with_id, ":memory:")
    tb = TokenTable(con, "inverted_lists", ignoreSelf=True)
    total_sets = con.execute("SELECT COUNT(*) FROM sets").fetchone()[0]
    set_total_number_of_sets(total_sets)
    yield con, tb
    con.close()


@pytest.mark.parametrize(
    "tokens",
    [
        [b"banana"],
        [b"apple"],
        [b"banana", b"apple"],
        [b"banana", b"cherry"],
        [b"pear", b"kiwi"],
        [b"x", b"y", b"z"],
        [b"melon", b"banana"],
    ],
)
def test_search_strategies_converge(
    sample_index: tuple[duckdb.DuckDBPyConnection, TokenTable], tokens: list[bytes]
) -> None:
    con, tb = sample_index
    query = rawTokenSet(ID=None, RawTokens=tokens, Tokens=None)
    probe_res, _ = searchProbeSetOptimized(
        con, "inverted_lists", "sets", tb, query, k=3, ignoreSelf=True
    )
    merge_res, _ = searchMergeList(
        con, "inverted_lists", tb, query, k=3, ignoreSelf=True
    )
    greedy_res, _ = searchMergeProbeCostModelGreedy(
        con, "inverted_lists", "sets", tb, query, k=3, ignoreSelf=True
    )

    assert {r.ID for r in probe_res} == {r.ID for r in merge_res} == {
        r.ID for r in greedy_res
    }
    assert {r.Overlap for r in probe_res} == {r.Overlap for r in merge_res} == {
        r.Overlap for r in greedy_res
    }
