# py-josie

`py-josie` is a Python implementation of the JOSIE algorithms, originally introduced in the SIGMOD 2019 paper ([link](https://dl.acm.org/doi/10.1145/3299869.3300065)). The original JOSIE system was implemented in Go with PostgreSQL ([link](https://github.com/ekzhu/josie)).

The `py-josie` package provides an end-to-end pipeline in Python: it builds an inverted index from tabular data columns using DuckDB and Polars, and then performs fast top-k set overlap search.

JOSIE itself is a highly optimized search algorithm over inverted indexes. Its key idea is a cost model that dynamically decides whether it is more efficient to read a posting list or to probe a candidate set. While JOSIE was originally proposed for joinable table discovery in large-scale data lakes, it is equally applicable to general set overlap search.

## Getting Started

### Prerequisites
- Python 3.10+
- DuckDB, Polars, Pandas (listed in `requirements.txt`)

Install the dependencies into a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For development or running the test suite you will also need `pytest`:

```bash
pip install -r requirements-dev.txt
```

### Install in editable mode

```bash
pip install -e .
```

### Building an index

```python
import pandas as pd
from josie_indexing import build_josie_index_polars

frames = [
    ("demo_a", pd.DataFrame({"title": ["apple", "banana", "cherry"]})),
    ("demo_b", pd.DataFrame({"title": ["banana", "durian", "kiwi", "apple"]})),
]

con = build_josie_index_polars(frames, ":memory:")
print(con.execute("SELECT COUNT(*) FROM sets").fetchone())
```

### Running queries

```python
from josie_online import (
    TokenTable,
    rawTokenSet,
    searchProbeSetOptimized,
    set_total_number_of_sets,
)

# Prepare lookup structures
set_total_number_of_sets(con.execute("SELECT COUNT(*) FROM sets").fetchone()[0])
token_table = TokenTable(con, "inverted_lists", ignoreSelf=True)

query = rawTokenSet(ID=None, RawTokens=[b"apple", b"banana"], Tokens=None)
results, _ = searchMergeProbeCostModelGreedy(
    con,
    list_table="inverted_lists",
    set_table="sets",
    tb=token_table,
    query=query,
    k=3,
    ignoreSelf=True,
)
print([(r.ID, r.Overlap) for r in results])
```

### Running the test suite

```bash
pytest
```

## Project layout

```
py-josie/
├── josie_indexing.py        # Index construction utilities
├── josie_online.py          # Online search strategies and helpers
├── tests/                   # Pytest-based contract tests
│   └── test_josie_pipeline.py
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
└── README.md
```


## Acknowledgements
The algorithms and concepts implemented here originate from the JOSIE authors. This repository focuses on providing a Python adaptation. 

This work is supported by the Canada Excellence Research Chair in Data Intelligence project, under the supervision of Prof. Renée J. Miller.

## Contact
Chao Zhang, University of Waterloo: chao.zhang@uwaterloo.ca