"""Python translation of the Go JOSIE sources tweaked for DuckDB.

The module is intentionally organised to mirror the original Go files so that
cross-referencing the paper/implementation remains straightforward.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import duckdb
import heapq
import math

__all__ = [
    "rawTokenSet",
    "TokenTable",
    "searchProbeSetSuffix",
    "searchProbeSetOptimized",
    "searchMergeList",
    "searchMergeDistinctList",
    "searchMergeProbeCostModelGreedy",
    "searchResult",
    "searchResultHeap",
    "InvertedList",
    "InvertedEntry",
    "set_total_number_of_sets",
]

############################################################
# ===================== common.go ==========================
############################################################

totalNumberOfSets: float = 1.0  # same as Go init default


def set_total_number_of_sets(n: int | float):
    """Persist the global corpus size that powers overlap heuristics."""
    global totalNumberOfSets
    totalNumberOfSets = float(n if n > 0 else 1.0)


def pruningPowerUb(freq: int, k: int) -> float:
    """Return the log upper-bound used to prune candidate inverted lists."""
    a = (min(k, freq) + 0.5) * (
        totalNumberOfSets - float(k) - float(freq) + float(min(k, freq)) + 0.5
    )
    b = (max(0, k - freq) + 0.5) * (max(freq - k, 0) + 0.5)
    if b <= 0:
        b = 1e-9
    return math.log(a / b)


def inverseSetFrequency(freq: int) -> float:
    """Compute the IDF-style weight for a token frequency."""
    if freq <= 0:
        freq = 1
    return math.log(totalNumberOfSets / float(freq))


def nextDistinctList(
    tokens: List[int], gids: List[int], currListIndex: int
) -> Tuple[int, int]:
    """Advance to the next token whose group differs, tracking skipped overlap. Returns (nextIndex, numSkipped)."""
    if currListIndex == len(tokens) - 1:
        return len(tokens), 0
    numSkipped = 0
    for i in range(currListIndex + 1, len(tokens)):
        if i < len(tokens) - 1 and gids[i + 1] == gids[i]:
            numSkipped += 1
            continue
        return i, numSkipped
    return len(tokens), numSkipped


def overlap(setTokens: List[int], queryTokens: List[int]) -> int:
    """Return the size of the intersection of two sorted token lists."""
    i = j = ov = 0
    while i < len(queryTokens) and j < len(setTokens):
        d = queryTokens[i] - setTokens[j]
        if d == 0:
            ov += 1
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    return ov


def overlapAndUpdateCounts(
    setTokens: List[int], queryTokens: List[int], counts: List[int]
) -> int:
    """Compute overlap and decrement per-token counters when matches occur."""
    i = j = ov = 0
    while i < len(queryTokens) and j < len(setTokens):
        d = queryTokens[i] - setTokens[j]
        if d == 0:
            counts[i] -= 1
            ov += 1
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    return ov


############################################################
# ===================== heap.go ============================
############################################################


@dataclass
class searchResult:
    """Stores the ID of a candidate set and its overlap with the query."""

    ID: int
    Overlap: int


class searchResultHeap:
    """Min-heap keyed by overlap that keeps track of the current top-k sets."""

    def __init__(self):
        self._h: List[Tuple[int, int]] = []  # (Overlap, ID) as min-heap

    def __len__(self):
        return len(self._h)

    def Len(self) -> int:
        return len(self._h)

    def Push(self, x: searchResult):
        """Insert a new candidate using (overlap, id) for min-heap ordering."""
        heapq.heappush(self._h, (x.Overlap, x.ID))

    def Pop(self) -> searchResult:
        """Remove and return the lowest-overlap candidate."""
        ov, id_ = heapq.heappop(self._h)
        return searchResult(ID=id_, Overlap=ov)


def kthOverlap(h: searchResultHeap, k: int) -> int:
    """Return the overlap needed to qualify for the current top-k."""
    if h.Len() < k:
        return 0
    return h._h[0][0]


def pushCandidate(h: searchResultHeap, k: int, id_: int, overlap_val: int) -> bool:
    """Push a candidate if it improves the heap; return True when accepted."""
    if h.Len() == k:
        if h._h[0][0] >= overlap_val:
            return False
        heapq.heappop(h._h)
    heapq.heappush(h._h, (overlap_val, id_))
    return True


def orderedResults(h: searchResultHeap) -> List[searchResult]:
    """Return heap contents ordered from highest to lowest overlap."""
    tmp = list(h._h)
    tmp.sort(reverse=True)  # descending
    return [searchResult(ID=id_, Overlap=ov) for (ov, id_) in tmp]


def kthOverlapAfterPush(h: searchResultHeap, k: int, overlap_val: int) -> int:
    """Predict the kth overlap if a value were inserted without mutating h."""
    # Mirrors Go logic but guards indices.
    if h.Len() < k - 1:
        return 0
    kth = h._h[0][0] if h.Len() >= 1 else 0
    if overlap_val <= kth:
        return kth
    if k == 1:
        return overlap_val
    if k == 2:
        if h.Len() >= 2:
            jth = h._h[1][0]
        else:
            jth = overlap_val
        return min(jth, overlap_val)
    # k >= 3
    if h.Len() >= 3:
        jth = min(h._h[1][0], h._h[2][0])
    elif h.Len() == 2:
        jth = h._h[1][0]
    else:
        jth = overlap_val
    return min(jth, overlap_val)


def copyHeap(h: searchResultHeap) -> searchResultHeap:
    """Return a shallow copy of the heap useful for what-if analysis."""
    h2 = searchResultHeap()
    h2._h = list(h._h)
    return h2


############################################################
# ===================== tokentable.go ======================
############################################################


@dataclass
class rawTokenSet:
    """Container for a set in its raw-token representation."""

    ID: Optional[int]
    RawTokens: Optional[List[bytes]]
    Tokens: Optional[List[int]]  # Unused in the in-memory variant


def _qi(name: str) -> str:
    """Safe identifier quoting for DuckDB."""
    # Manual safe quoting for DuckDB identifiers
    return '"' + name.replace('"', '""') + '"'


# The in-memory token table variant.
class TokenTable:
    """
    In-memory token table (simplified):
      - Maps raw_token -> (token, group_id).
      - Keeps frequencies indexed by duplicate_group_id.
      - process(set): consumes set.RawTokens; applies ignoreSelf; returns
        (tokens, counts, gids) sorted by token ASC.
    """

    def __init__(
        self, con: duckdb.DuckDBPyConnection, list_table: str, ignoreSelf: bool
    ):
        self.con = con
        self.list_table = list_table
        self.ignoreSelf = ignoreSelf

        # Size frequency array by max duplicate_group_id so gid lookups stay O(1).
        row = self.con.execute(
            f"SELECT MAX(duplicate_group_id) FROM {_qi(self.list_table)}"
        ).fetchone()
        max_gid = int(row[0]) if row and row[0] is not None else -1
        self.frequencies: List[int] = [0] * (max_gid + 1)

        # Load raw_token, token, frequency, duplicate_group_id
        df = self.con.execute(
            f"""
            SELECT raw_token, token, frequency, duplicate_group_id
            FROM {_qi(self.list_table)}
            """
        ).fetchdf()

        # Map: raw_token (bytes) -> (token, group_id)
        self.token_map: Dict[bytes, Tuple[int, int]] = {}

        for raw_token, token, freq, gid in zip(
            df["raw_token"], df["token"], df["frequency"], df["duplicate_group_id"]
        ):
            if not isinstance(raw_token, (bytes, bytearray)):
                raw_token = str(raw_token).encode("utf-8")

            t = int(token)
            g = int(gid)
            f = int(freq)

            if g >= len(self.frequencies):
                self.frequencies.extend([0] * (g - len(self.frequencies) + 1))
            self.frequencies[g] = f

            self.token_map[raw_token] = (t, g)

    def process(self, set_: rawTokenSet) -> Tuple[List[int], List[int], List[int]]:
        """Map raw bytes to token ids plus frequencies, sorted by token id."""
        tokens: List[int] = []
        counts: List[int] = []
        gids: List[int] = []

        if not set_.RawTokens:
            return tokens, counts, gids

        for raw_token in set_.RawTokens:
            if not isinstance(raw_token, (bytes, bytearray)):
                raw_token = str(raw_token).encode("utf-8")

            entry = self.token_map.get(raw_token)
            if entry is None:
                continue

            token, gid = entry
            freq = self.frequencies[gid]

            # NOTE: since all tokens are originally from the database, if frequency is 1
            # it means the token only exists in the query set
            if self.ignoreSelf and freq < 2:
                continue

            tokens.append(int(token))
            counts.append(int(freq - 1))
            gids.append(int(gid))

        # Sort by token ASC, keep arrays aligned
        order = sorted(range(len(tokens)), key=lambda i: tokens[i])
        tokens = [tokens[i] for i in order]
        counts = [counts[i] for i in order]
        gids = [gids[i] for i in order]
        return tokens, counts, gids


############################################################
# ================ helpers for inverted lists ==============
############################################################


@dataclass
class InvertedEntry:
    """Represents a single posting from the inverted index."""

    ID: int
    Size: int
    MatchPosition: int


def InvertedList(
    con: duckdb.DuckDBPyConnection, list_table: str, token: int
) -> List[InvertedEntry]:
    """Materialise the postings list for a token from DuckDB."""
    # Robust: concatenate rows if multiple (should be one per token).
    df = con.execute(
        f"SELECT set_ids, set_sizes, match_positions FROM {list_table} WHERE token = ?",
        [token],
    ).fetchdf()
    if df.empty:
        return []
    entries: List[InvertedEntry] = []
    for _, row in df.iterrows():
        set_ids = row["set_ids"]
        set_sizes = row["set_sizes"]
        match_positions = row["match_positions"]
        for sid, sz, mp in zip(set_ids, set_sizes, match_positions):
            entries.append(
                InvertedEntry(ID=int(sid), Size=int(sz), MatchPosition=int(mp))
            )
    return entries


def setTokensSuffix(
    con: duckdb.DuckDBPyConnection, set_table: str, set_id: int, start_pos: int
) -> List[int]:
    """Fetch the tail of the token sequence starting at start_pos (inclusive)."""
    row = con.execute(
        f"SELECT tokens FROM {set_table} WHERE id = ?", [set_id]
    ).fetchone()
    if not row:
        return []
    tokens = row[0]
    start_pos = max(0, min(start_pos, len(tokens)))
    return tokens[start_pos:]


############################################################
# ===================== probe_sets.go ======================
############################################################


def searchProbeSetSuffix(
    con: duckdb.DuckDBPyConnection,
    list_table: str,
    set_table: str,
    tb: TokenTable,
    query: rawTokenSet,
    k: int,
    ignoreSelf: bool,
) -> Tuple[List[searchResult], dict]:
    """Baseline suffix probing that scans each posting list independently."""
    tokens, _, _ = tb.process(query)
    h = searchResultHeap()
    ignores: Dict[int, bool] = (
        {query.ID: True} if (ignoreSelf and query.ID is not None) else {}
    )
    for i, token in enumerate(tokens):
        # Early exit once remaining tokens cannot beat the current kth overlap.
        if kthOverlap(h, k) >= len(tokens) - i:
            break
        entries = InvertedList(con, list_table, token)
        for entry in entries:
            if ignores.get(entry.ID, False):
                continue
            ignores[entry.ID] = True
            if kthOverlap(h, k) >= min(
                len(tokens) - i, entry.Size - entry.MatchPosition
            ):
                continue
            s = setTokensSuffix(con, set_table, entry.ID, entry.MatchPosition)
            o = overlap(s, tokens[i:])
            pushCandidate(h, k, entry.ID, o)
    return orderedResults(h), {}


def searchProbeSetOptimized(
    con: duckdb.DuckDBPyConnection,
    list_table: str,
    set_table: str,
    tb: TokenTable,
    query: rawTokenSet,
    k: int,
    ignoreSelf: bool,
    gids_from_tokens: Optional[List[int]] = None,
) -> Tuple[List[searchResult], dict]:
    """Optimised probe that batches duplicate-group tokens to reuse work."""
    tokens, _, gids = tb.process(query) # query tokens sorted by the global token order
    if gids_from_tokens is not None:
        gids = gids_from_tokens
    h = searchResultHeap()
    ignores: Dict[int, bool] = (
        {query.ID: True} if (ignoreSelf and query.ID is not None) else {}
    )
    numSkipped = 0
    qsize = len(tokens) 
    i = 0
    while i < qsize:
        token = tokens[i]
        skippedOverlap = numSkipped
        # Consider both remaining tokens and skipped overlap when pruning.
        if kthOverlap(h, k) >= len(tokens) - i + skippedOverlap: #prefix filtering
            break
        entries = InvertedList(con, list_table, token) #get posting list for the token
        for entry in entries: # should be just one entry per token
            if ignores.get(entry.ID, False):
                continue
            ignores[entry.ID] = True
            if kthOverlap(h, k) >= min(
                len(tokens) - i + skippedOverlap,
                entry.Size - entry.MatchPosition + skippedOverlap,
            ): 
                continue
            s = setTokensSuffix(con, set_table, entry.ID, entry.MatchPosition)
            o = overlap(s, tokens[i:]) + skippedOverlap
            pushCandidate(h, k, entry.ID, o)
        ni, numSkipped = nextDistinctList(tokens, gids, i)
        i = ni
    return orderedResults(h), {}


############################################################
# ===================== merge_lists.go =====================
############################################################


def searchMergeList(
    con: duckdb.DuckDBPyConnection,
    list_table: str,
    tb: TokenTable,
    query: rawTokenSet,
    k: int,
    ignoreSelf: bool,
) -> Tuple[List[searchResult], dict]:
    """Merge-list strategy that counts overlaps across all postings."""
    tokens, _, _ = tb.process(query)
    counter: Dict[int, int] = {}
    for token in tokens:
        for entry in InvertedList(con, list_table, token):
            if ignoreSelf and query.ID is not None and entry.ID == query.ID:
                continue
            counter[entry.ID] = counter.get(entry.ID, 0) + 1
    h = searchResultHeap()
    for sid, ov in counter.items():
        pushCandidate(h, k, sid, ov)
    return orderedResults(h), {}


def searchMergeDistinctList(
    con: duckdb.DuckDBPyConnection,
    list_table: str,
    tb: TokenTable,
    query: rawTokenSet,
    k: int,
    ignoreSelf: bool,
    gids_from_tokens: Optional[List[int]] = None,
) -> Tuple[List[searchResult], dict]:
    """Merge-list variant that skips duplicate groups to tighten bounds."""
    tokens, _, gids = tb.process(query)
    if gids_from_tokens is not None:
        gids = gids_from_tokens
    counter: Dict[int, int] = {}
    numSkipped = 0
    qsize = len(tokens)
    i = 0
    while i < qsize:
        token = tokens[i]
        skippedOverlap = numSkipped
        for entry in InvertedList(con, list_table, token):
            if ignoreSelf and query.ID is not None and entry.ID == query.ID:
                continue
            counter[entry.ID] = counter.get(entry.ID, 0) + skippedOverlap + 1
        ni, numSkipped = nextDistinctList(tokens, gids, i)
        i = ni
    h = searchResultHeap()
    for sid, ov in counter.items():
        pushCandidate(h, k, sid, ov)
    return orderedResults(h), {}


############################################################
# ===================== josie_util.go ======================
############################################################


@dataclass
class candidateEntry:
    """Aggregates state for a candidate set while the algorithm probes it."""

    id: int
    size: int
    firstMatchPosition: int
    latestMatchPosition: int
    queryFirstMatchPosition: int
    partialOverlap: int
    maximumOverlap: int = 0
    estimatedOverlap: int = 0
    estimatedCost: float = 0.0
    estimatedNextUpperbound: int = 0
    estimatedNextTruncation: int = 0
    read: bool = False


def newCandidateEntry(
    id_: int,
    size: int,
    candidateCurrentPosition: int,
    queryCurrentPosition: int,
    skippedOverlap: int,
) -> candidateEntry:
    """Initialise a candidate when we observe its first matching token."""
    return candidateEntry(
        id=id_,
        size=size,
        firstMatchPosition=candidateCurrentPosition,
        latestMatchPosition=candidateCurrentPosition,
        queryFirstMatchPosition=queryCurrentPosition,
        partialOverlap=skippedOverlap + 1, #overlapping token skipped + the token at the current position
    )


def candidate_update(
    ce: candidateEntry, candidateCurrentPosition: int, skippedOverlap: int
):
    """Extend the candidate overlap when subsequent tokens match."""
    ce.latestMatchPosition = candidateCurrentPosition
    ce.partialOverlap = ce.partialOverlap + skippedOverlap + 1


def candidate_upperboundOverlap(
    ce: candidateEntry, querySize: int, queryCurrentPosition: int
) -> int:
    """Tight upper bound on how much overlap the candidate can still gain."""
    ce.maximumOverlap = ce.partialOverlap + min(
        querySize - queryCurrentPosition - 1, ce.size - ce.latestMatchPosition - 1
    )
    return ce.maximumOverlap


# Estimate the total overlap, this assumes update has been called if
# the queryCurrentPosition has a matching token
# This is based on equation (4) in the paper
def candidate_estOverlap(
    ce: candidateEntry, querySize: int, queryCurrentPosition: int
) -> int:
    """Extrapolate final overlap based on prefix matches seen so far."""
    prefix_len = queryCurrentPosition + 1 - ce.queryFirstMatchPosition
    if prefix_len <= 0:
        prefix_len = 1
    est = int(
        float(ce.partialOverlap)
        / float(prefix_len)
        * float(querySize - ce.queryFirstMatchPosition)
    )
    ce.estimatedOverlap = min(
        est, candidate_upperboundOverlap(ce, querySize, queryCurrentPosition)
    )
    return ce.estimatedOverlap


# I/O cost model placeholders (simple version)
def readSetCost(length: int) -> float:
    """Toy cost model: proportional to the remaining suffix length."""
    return float(max(0, length))


def readListCost(freq_plus_one: int) -> float:
    """Toy cost model: proportional to the postings list length."""
    return float(max(0, freq_plus_one))


def candidate_estCost(ce: candidateEntry) -> float:
    """Estimate how expensive it is to read the remainder of the set."""
    ce.estimatedCost = readSetCost(candidate_suffixLength(ce))
    return ce.estimatedCost

# Estimate the number tokens truncated from the suffix after reading the posting lists
# from queryCurrentPosition+1 to queryNextPosition
# This is based on equation (10) in the paper
def candidate_estTruncation(
    ce: candidateEntry,
    querySize: int,
    queryCurrentPosition: int,
    queryNextPosition: int,
) -> int:
    """Predict how many suffix elements can be skipped in the next batch."""
    denom = max(1, (querySize - ce.queryFirstMatchPosition))
    ce.estimatedNextTruncation = int(
        float(queryNextPosition - queryCurrentPosition)
        / float(denom)
        * float(max(0, ce.size - ce.firstMatchPosition))
    )
    return ce.estimatedNextTruncation


# Estimate the next overlap upper bound after reading the posting lists
# from queryCurrentPosition+1 to queryNextPosition
# This is based on equation (11) in the paper
def candidate_estNextOverlapUpperbound(
    ce: candidateEntry,
    querySize: int,
    queryCurrentPosition: int,
    queryNextPosition: int,
) -> int:
    """Upper bound the overlap achievable after the next probing step."""
    queryJumpLength = queryNextPosition - queryCurrentPosition
    queryPrefixLength = max(1, (queryCurrentPosition + 1 - ce.queryFirstMatchPosition))
    additionalOverlap = int(
        float(ce.partialOverlap) / float(queryPrefixLength) * float(queryJumpLength)
    )
    denom = max(1, (querySize - ce.queryFirstMatchPosition))
    nextLatestMatchingPosition = (
        int(
            float(queryJumpLength)
            / float(denom)
            * float(max(0, ce.size - ce.firstMatchPosition))
        )
        + ce.latestMatchPosition
    )
    ce.estimatedNextUpperbound = (
        ce.partialOverlap
        + additionalOverlap
        + min(
            querySize - queryNextPosition - 1, ce.size - nextLatestMatchingPosition - 1
        )
    )
    return ce.estimatedNextUpperbound


def candidate_suffixLength(ce: candidateEntry) -> int:
    """Length of the suffix that has not yet been compared."""
    return max(0, ce.size - ce.latestMatchPosition - 1)


def candidate_checkMinSampleSize(
    ce: candidateEntry, queryCurrentPosition: int, batchSize: int
) -> bool:
    """Guard to ensure we have seen enough prefix before probing further."""
    return (queryCurrentPosition - ce.queryFirstMatchPosition + 1) > batchSize


def sort_byEstimatedOverlap(candidates: List[candidateEntry]) -> List[candidateEntry]:
    """Sort candidates by optimistic overlap then cost (descending)."""
    return sorted(candidates, key=lambda c: (-c.estimatedOverlap, c.estimatedCost))


def sort_byMaximumOverlap_increasing(
    candidates: List[candidateEntry],
) -> List[candidateEntry]:
    """Order candidates by the tightest known maximum overlap."""
    return sorted(candidates, key=lambda c: c.maximumOverlap)


def sort_byFutureMaxOverlap_increasing(
    candidates: List[candidateEntry],
) -> List[candidateEntry]:
    """Order candidates by their predicted overlap after the next batch."""
    return sorted(candidates, key=lambda c: c.estimatedNextUpperbound)


def upperboundOverlapUknownCandidate(
    querySize: int, queryCurrentPosition: int, prefixOverlap: int
) -> int:
    """Loose bound assuming unseen candidates could match every remaining token."""
    return querySize - queryCurrentPosition + prefixOverlap


def nextBatchDistinctLists(
    tokens: List[int], gids: List[int], currIndex: int, batchSize: int
) -> int:
    """Compute the index that marks the end of the next batch of lists."""
    n = 0
    end = currIndex
    next_i, _ = nextDistinctList(tokens, gids, currIndex)
    while next_i < len(tokens):
        end = next_i
        n += 1
        if n == batchSize:
            break
        next_i, _ = nextDistinctList(tokens, gids, end)
    return end


def prefixLength(querySize: int, kthOverlapVal: int) -> int:
    """Length of the query prefix that drove the current kth overlap."""
    if kthOverlapVal == 0:
        return querySize
    return querySize - kthOverlapVal + 1


def readListsBenefitForCandidate(ce: candidateEntry, kthOverlapVal: int) -> float:
    """Estimate benefit gained from reading more of this candidate's list."""
    if kthOverlapVal >= ce.estimatedNextUpperbound:
        return ce.estimatedCost
    return ce.estimatedCost - readSetCost(
        candidate_suffixLength(ce) - ce.estimatedNextTruncation
    )

#  Process unread candidates from the counter to obtain the sorted list of
#  qualified candidates, and compute the benefit of reading the next batch
#  of lists.
def processCandidatesInit(
    querySize: int,
    queryCurrentPosition: int,
    nextBatchEndIndex: int,
    kthOverlapVal: int,
    minSampleSize: int,
    candidates: Dict[int, candidateEntry],
    ignores: Dict[int, bool],
) -> Tuple[float, int, List[candidateEntry]]:
    """Score existing candidates before probing the next batch of lists."""
    readListsBenefit = 0.0
    numWithBenefit = 0
    qualified: List[candidateEntry] = []
    for cid, ce in list(candidates.items()):
        candidate_upperboundOverlap(ce, querySize, queryCurrentPosition)
        # Disqualify candidates and remove it for future reads
        if kthOverlapVal >= ce.maximumOverlap:
            del candidates[cid]
            ignores[cid] = True
            continue
        # Candidate does not qualify if the estimation std err is too high; not enough samples for estimation
        if not candidate_checkMinSampleSize(ce, queryCurrentPosition, minSampleSize):
            continue
        # Compute estimation
        candidate_estCost(ce)
        candidate_estOverlap(ce, querySize, queryCurrentPosition)
        candidate_estTruncation(ce, querySize, queryCurrentPosition, nextBatchEndIndex)
        candidate_estNextOverlapUpperbound(
            ce, querySize, queryCurrentPosition, nextBatchEndIndex
        )
        readListsBenefit += readListsBenefitForCandidate(ce, kthOverlapVal)
        qualified.append(ce)
        if ce.estimatedOverlap > kthOverlapVal:
            numWithBenefit += 1
    return readListsBenefit, numWithBenefit, qualified # qualified candidates are compared to see whether they will be read


def processCandidatesUpdate(
    kthOverlapVal: int,
    candidates_arr: List[candidateEntry],
    counter: Dict[int, candidateEntry],
    ignores: Dict[int, bool],
) -> float:
    """Refresh candidate gains after the heap's kth overlap changed."""
    readListsBenefit = 0.0
    for j, ce in enumerate(candidates_arr):
        if ce is None or ce.read:
            continue
        if ce.maximumOverlap <= kthOverlapVal:
            candidates_arr[j] = None  # type: ignore
            if ce.id in counter:
                del counter[ce.id]
            ignores[ce.id] = True
        readListsBenefit += readListsBenefitForCandidate(ce, kthOverlapVal)
    return readListsBenefit


def readSetBenefit(
    querySize: int,
    kthOverlapVal: int,
    kthOverlapAfterPushVal: int,
    candidates_arr: List[candidateEntry],
    readListCosts: List[float],
    fast: bool,
) -> float:
    """Estimate the benefit of probing one more candidate set."""
    b = 0.0
    if kthOverlapAfterPushVal <= kthOverlapVal:
        return b
    p0 = prefixLength(querySize, kthOverlapVal)
    p1 = prefixLength(querySize, kthOverlapAfterPushVal)
    b += readListCosts[p0 - 1] - readListCosts[p1 - 1]
    if fast:
        return b
    for ce in candidates_arr:
        if ce is None or ce.read:
            continue
        if ce.maximumOverlap <= kthOverlapAfterPushVal:
            b += ce.estimatedCost
    return b


############################################################
# ===================== josie.go ===========================
############################################################

batchSize = 8
expensiveEstimationBudget = 2000


def searchMergeProbeCostModelGreedy(
    con: duckdb.DuckDBPyConnection,
    listTable: str,
    setTable: str,
    tb: TokenTable,
    query: rawTokenSet,
    k: int,
    ignoreSelf: bool,
) -> Tuple[List[searchResult], dict]:
    """Hybrid merge/probe strategy that balances cost using a greedy heuristic."""

    tokens, freqs, gids = tb.process(query)

    # cumulative cost of reading posting lists up to i, size of posting list i is freqs[i] + 1
    readListCosts: List[float] = [0.0 for _ in range(len(freqs))]
    for i in range(len(freqs)):        
        if i == 0:
            readListCosts[i] = readListCost(freqs[i] + 1)
        else:
            readListCosts[i] = readListCosts[i - 1] + readListCost(freqs[i] + 1)

    querySize = len(tokens)
    counter: Dict[int, candidateEntry] = {}  # candidate ID -> candidateEntry
    ignores: Dict[int, bool] = (
        {query.ID: True} if (ignoreSelf and query.ID is not None) else {}
    ) # IDs of candidates we can skip

    h = searchResultHeap()
    numSkipped = 0  # overlap contribution of skipped duplicate-group tokens
    currBatchLists = batchSize

    i = 0
    while i < querySize:
        token = tokens[i]
        skippedOverlap = numSkipped
        
        # is this prefix filtering
        maxOverlapUnseenCandidate = upperboundOverlapUknownCandidate(
            querySize, i, skippedOverlap
        )

        # Early terminates once the threshold index has reached and
		# there is no remaining sets in the counter
        if kthOverlap(h, k) >= maxOverlapUnseenCandidate and len(counter) == 0:
            break

        # Read the list for the current token
        entries = InvertedList(con, listTable, token)

        # Merge this list and compute counter entries
		# Skip sets that has been computed for exact overlap previously
        for entry in entries: # should be just one entry per token            
            if ignores.get(entry.ID, False):
                continue
            # Process seen candidates
            if entry.ID in counter:
                # We have seen this candidate before; just update overlap stats.
                candidate_update(counter[entry.ID], entry.MatchPosition, skippedOverlap)
                continue            
            # No need to process unseen candidate if we have reached this point
            if kthOverlap(h, k) >= maxOverlapUnseenCandidate:
                continue
            
            # Process new candidate
            counter[entry.ID] = newCandidateEntry(
                entry.ID, entry.Size, entry.MatchPosition, i, skippedOverlap
            )

        if i == querySize - 1:
            break

        # Get the next list
        if (
            (len(counter) == 0) # Continue reading the next list when there is no candidates
            or ((len(counter) < k) and (h.Len() < k)) # Do not start reading sets until we have seen at least k
            or (currBatchLists > 0) # Continue reading the next list when we are still in the current batch
        ):
            currBatchLists -= 1
            ni, numSkipped = nextDistinctList(tokens, gids, i)
            i = ni
            continue

        # Reset counter
        currBatchLists = batchSize
        
        # Find the end index of the next batch of posting lists
        nextBatchEndIndex = nextBatchDistinctLists(tokens, gids, i, batchSize)
        
        # Compute the cost of reading the next batch of posting lists
        mergeListsCost = readListCosts[nextBatchEndIndex] - readListCosts[i]
        
        # Process candidates to estimate benefit of reading the next batch of posting lists and obtain qualified candidates
        mergeListsBenefit, numWithBenefit, candidates_arr = processCandidatesInit(
            querySize,
            i,
            nextBatchEndIndex,
            kthOverlap(h, k),
            batchSize,
            counter,
            ignores,
        )

        if numWithBenefit == 0 or len(candidates_arr) == 0:
            ni, numSkipped = nextDistinctList(tokens, gids, i)
            i = ni
            continue

        # Sort candidates by estimated overlap (descending) and estimated cost (ascending)
        candidates_arr = sort_byEstimatedOverlap(candidates_arr)

        prevKthOverlap = kthOverlap(h, k)
        numCandidateExpensive = 0
        fastEstimate = False
        fastEstimateKthOverlap = 0

        for cand in candidates_arr:
            if cand is None:
                continue
            kth = kthOverlap(h, k)
            if cand.estimatedOverlap <= kth: # descending order, so we can stop here
                break

            if h.Len() >= k:
                # Decide whether to continue probing by comparing marginal gains.
                numCandidateExpensive += 1
                if (not fastEstimate) and (
                    numCandidateExpensive * len(candidates_arr)
                    > expensiveEstimationBudget
                ): # switch to fast estimation based on the budget
                    fastEstimate = True
                    fastEstimateKthOverlap = prevKthOverlap
                if not fastEstimate:
                    mergeListsBenefit = processCandidatesUpdate(
                        kth, candidates_arr, counter, ignores
                    ) # mergeListBenefit
                probeSetBenefit = readSetBenefit(
                    querySize,
                    kth,
                    kthOverlapAfterPush(h, k, cand.estimatedOverlap),
                    candidates_arr,
                    readListCosts,
                    fastEstimate,
                )
                probeSetCost = cand.estimatedCost
                if (probeSetBenefit - probeSetCost) < (
                    mergeListsBenefit - mergeListsCost
                ):
                    break # if probing benefit is less than merge-list benefit, stop probing further

            if fastEstimate or (
                (numCandidateExpensive + 1) * len(candidates_arr)
                > expensiveEstimationBudget
            ):
                # Under fast estimation we approximate the remaining benefits.
                mergeListsBenefit -= readListsBenefitForCandidate(
                    cand, fastEstimateKthOverlap
                )

            cand.read = True
            ignores[cand.id] = True
            if cand.id in counter:
                del counter[cand.id]

            if cand.maximumOverlap <= kth:
                continue

            if candidate_suffixLength(cand) > 0:
                s = setTokensSuffix(
                    con, setTable, cand.id, cand.latestMatchPosition + 1
                )
                suffixOverlap = overlap(s, tokens[i + 1 :])
                totalOverlap = suffixOverlap + cand.partialOverlap
            else:
                totalOverlap = cand.partialOverlap

            prevKthOverlap = kth
            # Push the now fully-evaluated candidate onto the heap.
            pushCandidate(h, k, cand.id, totalOverlap)

        ni, numSkipped = nextDistinctList(tokens, gids, i)
        i = ni

    for ce in counter.values():
        pushCandidate(h, k, ce.id, ce.partialOverlap)

    return orderedResults(h), {}
