"""Utilities for constructing JOSIE-compatible inverted indexes with DuckDB.

The implementation mirrors the preprocessing stages described in the JOSIE
paper, mapping sets of raw string tokens to integer identifiers that feed the
search algorithms implemented in ``josie_online``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from hashlib import blake2b

import duckdb
import pandas as pd
import polars as pl


__all__ = ["build_josie_index_polars"]


def build_josie_index_polars(
    dataframes_with_id: Iterable[tuple[str, pd.DataFrame | pl.DataFrame]],
    db_path: str,
    skip_tokens: set[str] | None = None,
) -> duckdb.DuckDBPyConnection:
    """Materialise the inverted index backing the online JOSIE operators.

    Parameters
    ----------
    dataframes_with_id
        Iterable of ``(table_id, dataframe)`` pairs that provide the raw
        string tokens. DataFrames can be either ``pandas`` or ``polars`` and
        each column becomes an independent set in the index.
    db_path
        Path to the DuckDB database file. Use ``":memory:"`` for an in-memory
        database during testing.
    skip_tokens
        Optional set of tokens that should be ignored (case insensitive).

    Returns
    -------
    duckdb.DuckDBPyConnection
        An open DuckDB connection with the ``inverted_lists``, ``sets``, and
        ``metadata`` tables populated and indexed.
    """
    if skip_tokens is None:
        skip_tokens = {
            "-",
            "--",
            "n/a",
            "total",
            "$",
            ":",
            "*",
            "+",
            "�",
            "@",
            "†",
            "▼",
        }

    skip_tokens_normalized = {str(token).lower() for token in skip_tokens}

    # Step 0: Extract RawTokens (RawToken, SetID) and metadata (table_id, col_name, set_id)
    raw_tokens = []
    metadata = []
    set_counter = 0
    for table_id, df in dataframes_with_id:
        df_pl = pl.from_pandas(df) if not isinstance(df, pl.DataFrame) else df
        for col in df_pl.columns:
            set_id = set_counter
            set_counter += 1
            tokens = df_pl[col].drop_nulls().cast(str).unique().to_list()
            for t in tokens:
                token_lower = t.lower()
                # if t.lower() not in skip_tokens and not t.isnumeric():
                if token_lower not in skip_tokens_normalized:
                    raw_tokens.append((t, set_id))
            metadata.append((table_id, col, set_id))

    raw_df = pl.DataFrame(raw_tokens, schema=["raw_token", "set_id"], orient="row")
    meta_df = pl.DataFrame(
        metadata, schema=["table_id", "column_name", "set_id"], orient="row"
    )

    # Step 1: Build posting lists (RawToken -> SetIDs)
    posting_lists = (
        raw_df.group_by("raw_token")
        .agg(pl.col("set_id").implode().list.unique().list.sort().alias("set_id"))
        .with_columns(
            pl.col("set_id")
            .map_elements(
                lambda values: "|".join(str(v) for v in values),
                return_dtype=pl.Utf8,
            )
            .alias("_sort_key")
        )
    )    
    # current schema of posting_lists: raw_token | set_id (list)

    # Step 2: Assign TokenID & GroupID
    posting_lists = posting_lists.with_columns(
        [
            pl.col("set_id").list.len().alias("frequency"),
            pl.col("set_id")
            .map_elements(
                lambda x: blake2b(str(x).encode(), digest_size=8).hexdigest(),
                return_dtype=pl.Utf8,
            )
            .alias("hash"),
        ]
    )
    # current schema of posting_lists: raw_token | set_id (list) | frequency | hash

    posting_lists_sorted = (
        posting_lists.sort(["frequency", "hash", "_sort_key"]).drop("_sort_key")
    ).with_row_index(name="token_id")
    # current schema of posting_lists_sorted: token_id | raw_token | set_id (list) | frequency | hash

    # Deduplicate by assigning group ids
    posting_lists_sorted = posting_lists_sorted.with_columns(
        (pl.col("set_id") != pl.col("set_id").shift(1))
        .fill_null(True)
        .alias("is_new_group")
    ).with_columns(pl.col("is_new_group").cast(pl.Int64).cum_sum().alias("group_id"))
    # current schema of posting_lists_sorted: token_id | raw_token | set_id (list) | frequency | hash | group_id

    # posting_lists_sorted = posting_lists_sorted.with_columns(
    #     posting_lists_sorted.group_by("group_id").agg(pl.len()).rename({"count": "group_count"})["group_count"].over("group_id")
    # )
    posting_lists_sorted = posting_lists_sorted.with_columns(
        pl.len().over("group_id").alias("group_count")
    )
    # current schema of posting_lists_sorted: token_id | raw_token | set_id (list) | frequency | hash | group_id | group_count

    # Step 3: Build Integer Sets (SetID -> TokenIDs)
    token_map = raw_df.join(
        posting_lists_sorted.select(["raw_token", "token_id", "group_id", "frequency"]),
        on="raw_token",
        how="left",
    )
    # schema of raw_df: raw_token | set_id
    # schema of token_map: raw_token | set_id | token_id | group_id | frequency

    token_map = token_map.with_columns(
        (pl.col("frequency") > 1).cast(pl.Int64).alias("is_non_singular")
    )
    # schema of token_map: raw_token | set_id | token_id | group_id | frequency | is_non_singular

    set_to_tokens = token_map.group_by("set_id").agg(
        pl.col("token_id").sort().alias("token_id"),
        pl.len().alias("size"),
        pl.sum("is_non_singular").alias("num_non_singular_token"),
    )

    # schema of set_to_tokens: set_id | token_id (list) | size | num_non_singular_token

    # Step 4: Build final Posting Lists (TokenID -> SetIDs, sizes, positions)
    inv_map = []
    for sid, tokens, size, *_ in set_to_tokens.iter_rows():
        for pos, tid in enumerate(tokens):
            inv_map.append((tid, sid, size, pos))
    inv_df = pl.DataFrame(
        inv_map, schema=["token_id", "set_id", "set_size", "position"], orient="row"
    )
    # schema of inv_df: token_id | set_id | set_size | position
    posting_lists_final = inv_df.group_by("token_id").agg(
        [
            pl.col("set_id").alias("set_ids"),
            pl.col("set_size").alias("set_sizes"),
            pl.col("position").alias("match_positions"),
        ]
    )
    # schema of posting_lists_final: token_id | set_ids (list) | set_sizes (list) | match_positions (list)
    posting_lists_final = posting_lists_final.join(
        posting_lists_sorted.select(
            ["token_id", "raw_token", "frequency", "group_id", "group_count"]
        ),
        on="token_id",
        how="left",
    )

    # Step 5: Save results into DuckDB
    con = duckdb.connect(database=db_path)
    con.execute("DROP TABLE IF EXISTS inverted_lists")
    con.execute("DROP TABLE IF EXISTS sets")
    con.execute("DROP TABLE IF EXISTS metadata")

    # Inverted lists schema
    con.execute(
        """
        CREATE TABLE inverted_lists (
            token INTEGER,
            frequency INTEGER,
            duplicate_group_id INTEGER,
            duplicate_group_count INTEGER,
            set_ids INTEGER[],
            set_sizes INTEGER[],
            match_positions INTEGER[],
            raw_token TEXT
        )
    """
    )

    # Sets schema
    con.execute(
        """
        CREATE TABLE sets (
            id INTEGER,
            size INTEGER,
            num_non_singular_token INTEGER,
            tokens INTEGER[]
        )
    """
    )

    # Metadata schema
    con.execute(
        """
        CREATE TABLE metadata (
            table_id TEXT,
            column_name TEXT,
            set_id INTEGER
        )
    """
    )

    con.register("pl", posting_lists_final.to_pandas())
    con.execute(
        """
        INSERT INTO inverted_lists
        SELECT token_id, frequency, group_id, group_count,
               set_ids, set_sizes, match_positions, raw_token
        FROM pl
    """
    )

    con.register("st", set_to_tokens.to_pandas())
    con.execute(
        """
        INSERT INTO sets
        SELECT set_id, size, num_non_singular_token, token_id
        FROM st
    """
    )

    con.register("mt", meta_df.to_pandas())
    con.execute(
        """
        INSERT INTO metadata
        SELECT table_id, column_name, set_id
        FROM mt
    """
    )
    
    # build indexes to speed up queries
    con.execute("CREATE INDEX idx_inverted_lists_token ON inverted_lists(token)")
    con.execute("CREATE INDEX idx_sets_id ON sets(id)")
    con.execute("CREATE INDEX idx_metadata_set_id ON metadata(set_id)")

    return con
