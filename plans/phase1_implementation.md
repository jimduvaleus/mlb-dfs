# Phase 1: Skeleton & Data Ingestion Implementation Plan

## 1. Project Structure

The project will follow a standard modular Python layout to ensure scalability for future phases.

```text
mlb-dfs/
├── data/
│   ├── raw/
│   │   ├── dk_salaries/          # Store incoming DK Salary CSVs
│   │   └── retrosheet/           # Downloaded Retrosheet .EVN/.EVA files
│   └── processed/
│       ├── historical_logs.parquet # Processed historical player performance
│       └── copula_store/          # To be used in Phase 2
├── scripts/
│   ├── fetch_retrosheet.sh        # Shell script to download/unzip data
│   └── process_historical.py      # Script to invoke Chadwick and generate parquet
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── dk_slate.py           # DK Salary CSV parsing
│   │   └── retrosheet_parser.py   # Chadwick wrapper and event processing
│   ├── models/
│   │   ├── __init__.py
│   │   └── marginals.py          # Gaussian/Mixture distribution classes
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py               # ProjectionProvider interface
│   │   └── static_csv.py         # Static CSV implementation
│   └── utils/
│       ├── __init__.py
│       └── scoring.py            # DK scoring logic constants
├── tests/
│   ├── conftest.py
│   ├── test_ingestion.py
│   ├── test_marginals.py
│   └── test_providers.py
├── pyproject.toml                 # Dependencies (numpy, scipy, pandas, pyarrow, pytest)
└── README.md
```

## 2. Slate Ingestor (`src/ingestion/dk_slate.py`)

**Goal**: Parse the standard DraftKings Salary CSV for "Classic" MLB slates.

- **Input**: Path to DK Salary CSV.
- **Fields to Extract**:
  - `Position`: (P, C, 1B, 2B, 3B, SS, OF)
  - `Name + ID`: (Primary key for uniqueness)
  - `Name`: (Display name)
  - `ID`: (Numeric ID)
  - `Roster Position`: (Used for validity)
  - `Salary`: (Numeric)
  - `TeamAbbrev`: (Team)
- **Design**:
  - A `DKSlate` class that loads the CSV into a `pandas.DataFrame`.
  - Method `get_players()`: Returns a list of standardized `Player` objects or a cleaned DataFrame.
  - Validation for missing salaries or unexpected position strings.

## 3. Historical Ingestor (`src/ingestion/retrosheet_parser.py`)

**Goal**: Convert Retrosheet event files into a clean player-game performance log.

- **Workflow**:
  1.  **Fetcher**: Script `scripts/fetch_retrosheet.sh` uses `curl` to download year-by-year event zip files from `retrosheet.org/game.htm`.
  2.  **Chadwick Wrapper**: Python code to invoke `cwevent` on unzipped `.EVN` files.
      - Command: `cwevent -y [year] -f 0-96 [files...] > events.csv` (using standard output format).
  3.  **Event Processor**:
      - Read the raw event CSV.
      - Map play codes to DraftKings scoring actions (e.g., event code 20 = Single = 3pts).
      - Aggregate by `game_id` and `player_id`.
      - **Pitcher Logic**: Sum innings, strikeouts, earned runs, and wins.
  4.  **Storage**: Save the final aggregated table as a **Parquet** file for Phase 2.

## 4. Gaussian Marginal Distributions (`src/models/marginals.py`)

**Goal**: Provide a simple interface for player performance distributions.

- **Class `GaussianMarginal`**:
  - `__init__(self, mu, sigma)`: Initialize with projected mean and std dev.
  - `ppf(self, u)`: Inverse Cumulative Distribution Function.
    - Implementation: `scipy.stats.norm.ppf(u, loc=self.mu, scale=self.sigma)`
- **Future-proofing**: Ensure the interface is compatible with the `MixtureDistribution` to be implemented in Phase 4.

## 5. Projection Abstraction (`src/providers/`)

**Goal**: Decouple the simulation engine from the source of player projections.

- **Interface `ProjectionProvider` (`src/providers/base.py`)**:
  - Abstract base class with method `get_projection(player_id) -> Tuple[float, float]`.
- **Implementation `StaticCSVProvider` (`src/providers/static_csv.py`)**:
  - Reads a CSV file with `player_id`, `projected_mean`, `projected_std`.
  - Useful for testing and manual entry before the prop-scraping module is built.

## 6. Testing Strategy

- **Test Suite**: `pytest`
- **Unit Tests**:
  - `test_ingestion.py`:
    - Mock DK CSV parsing.
    - Verify scoring logic for a simulated event log.
  - `test_marginals.py`:
    - Verify `ppf(0.5) == mu`.
    - Verify `ppf(0.1) < ppf(0.9)`.
  - `test_providers.py`:
    - Verify `StaticCSVProvider` correctly handles missing IDs.
- **Integration Tests**:
  - Small "End-to-End" test: Load a 1-game Retrosheet sample -> Process -> Verify total points match expectations.

## 7. Dependencies

- `pandas`: Data manipulation.
- `numpy`: Numerical operations.
- `scipy`: Statistical distributions.
- `pyarrow`: Parquet storage support.
- `pytest`: Testing framework.
- `chadwick`: Required binary tools (must be installed on host).
