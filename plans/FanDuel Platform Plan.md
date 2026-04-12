# FanDuel Platform Support — Phased Implementation Plan

## Context

The optimizer currently supports only DraftKings. This plan adds FanDuel as a second platform, covering the full stack: scoring rules, roster constraints, slate ingestion, projection fetching, upload file format, and UI. Each phase keeps DK working unchanged; FD support is additive.

**FanDuel Scoring**
- Batters: 1B=3, 2B=6, 3B=9, HR=12, RBI=3.5, R=3.2, BB=3, SB=6, HBP=3
- Pitchers: W=6, QS=4, ER=-3, SO=3, IP=3 (no CG/NH/hit-against bonuses)

**FanDuel Roster**: 9 players — P×1, C/1B×1, 2B×1, 3B×1, SS×1, OF×3, UTIL×1 (any non-P)
**Salary cap**: $35,000 | **Max hitters per team**: 4

---

## Phase 0 — Safety Locks (no behavior change)

**Goal**: Prevent accidental historical data corruption before any refactoring.

The scoring constants in `src/utils/scoring.py` are imported directly by `src/ingestion/retrosheet_parser.py` to compute DK points stored in `historical_logs.parquet`, which trains the copula model. Changing them silently corrupts the copula. FD scoring must live in separate files and must never touch `retrosheet_parser.py` until Phase 9.

**Actions:**
1. Add a `# HISTORICAL PIPELINE WARNING` docstring to `src/utils/scoring.py` explaining the dependency
2. Create `tests/test_scoring_contract.py` — asserts all DK scoring constants by exact value (acts as regression wire if anyone accidentally changes them)

**Files:** `src/utils/scoring.py` (docstring only), `tests/test_scoring_contract.py` (new)

---

## Phase 1 — Platform Abstraction Layer

**Goal**: Define the canonical data structures for platform-specific rules. No call sites change yet.

Create `src/platforms/` package:

| File | Purpose |
|---|---|
| `src/platforms/base.py` | `Platform` enum (`draftkings`\|`fanduel`), `ScoringRules` frozen dataclass (batter/pitcher weights + `.batter_points()` / `.pitcher_points()` helpers), `RosterRules` frozen dataclass (requirements dict, salary_cap, max_hitters_per_team, min_games, roster_size) |
| `src/platforms/draftkings.py` | `DK_SCORING`, `DK_ROSTER` — exact numeric parity with current `scoring.py` and `lineup.py` constants |
| `src/platforms/fanduel.py` | `FD_SCORING`, `FD_ROSTER`, `FD_SLOT_ELIGIBILITY` — maps slot label → eligible player positions (e.g. `'C/1B'→{'C','1B'}`, `'UTIL'→{'C','1B','2B','3B','SS','OF'}`) |
| `src/platforms/registry.py` | `get_scoring(platform)`, `get_roster(platform)`, `get_slot_eligibility(platform)` factory functions |

Create `tests/test_platforms.py` — verify DK constants match legacy `scoring.py`, FD batter/pitcher point calculations, roster slot counts.

**Files created:** `src/platforms/__init__.py`, `base.py`, `draftkings.py`, `fanduel.py`, `registry.py`, `tests/test_platforms.py`

---

## Phase 2 — Config Model

**Goal**: Add `platform` to `AppConfig` so it flows from `config.yaml` through Pydantic to every consumer. Existing configs without the key default to `draftkings`.

**`src/api/models.py`:**
```python
from src.platforms.base import Platform

class AppConfig(BaseModel):
    platform: Platform = Platform.DRAFTKINGS   # new top-level field
    paths: PathsConfig = ...
    ...

class PathsConfig(BaseModel):
    dk_slate: str = ""
    fd_slate: str = ""   # new — path to FD salary CSV
    ...
```

**`src/api/config_io.py`:** add `fd_slate` to the None→`""` write path; ensure `read_config` reads `platform` key from YAML with `"draftkings"` default.

Create `tests/test_config_io.py` — round-trip a config with `platform: fanduel` through write/read.

**Files modified:** `src/api/models.py`, `src/api/config_io.py` | **Created:** `tests/test_config_io.py`

---

## Phase 3 — Slate Ingestion Abstraction

**Goal**: Extract `BaseSlateIngestor` ABC; add `FanDuelSlateIngestor`; create factory.

**`src/ingestion/dk_slate.py`:** Add `BaseSlateIngestor(ABC)` with abstract `get_slate_dataframe() → pd.DataFrame` and `get_players() → List[Player]`. `DraftKingsSlateIngestor` becomes a subclass — no logic change.

**`src/ingestion/fd_slate.py`:** `FanDuelSlateIngestor` — parses FD salary CSV columns (`Id`, `Nickname`, `Position`, `Salary`, `Team`, `Game`, `Opponent`, `Roster Position`). Outputs same standardized DataFrame schema as DK ingestor. Mark column-name mapping with `# VERIFY: against actual FD salary CSV export` until confirmed with a real file.

**`src/ingestion/factory.py`:**
```python
def get_ingestor(platform: Platform, slate_path: str) -> BaseSlateIngestor:
    if platform == Platform.DRAFTKINGS: return DraftKingsSlateIngestor(slate_path)
    if platform == Platform.FANDUEL: return FanDuelSlateIngestor(slate_path)
```

Create `tests/test_fd_ingestion.py` — synthetic FD CSV fixture, covers column renaming, salary validation, position parsing, game extraction.

**Files modified:** `src/ingestion/dk_slate.py` | **Created:** `fd_slate.py`, `factory.py`, `tests/test_fd_ingestion.py`

---

## Phase 4 — Platform-Aware Lineup Validation

**Goal**: Make `Lineup.is_valid()` accept platform rules rather than reading module-level constants. The existing module-level constants (`SLOTS`, `SALARY_CAP`, etc.) are preserved for backward compatibility.

**`src/optimization/lineup.py` key changes:**

```python
def is_valid(
    self,
    player_meta: PlayerMeta,
    salary_floor: Optional[float] = None,
    rules: Optional[RosterRules] = None,
    slot_eligibility: Optional[dict] = None,
) -> bool:
    r = rules or DK_ROSTER
    se = slot_eligibility or get_slot_eligibility(Platform.DRAFTKINGS)
    slots = r.slots  # e.g. ['P','C/1B','2B','3B','SS','OF','OF','OF','UTIL'] for FD
    ...
```

The bipartite matching must use `slot_eligibility` to resolve compound slot labels. Current check is `slot_pos in elig` (works only for exact-match DK slots). FD requires:
```python
# slot_pos may be 'C/1B' or 'UTIL' — expand via se
slot_positions = se.get(slot_pos, {slot_pos})
if elig & slot_positions and j not in visited:
```

The "both pitchers not same team" constraint (lines 109–112) guards with `len(pitcher_teams) == 2` — automatically skipped for FD's 1-pitcher roster.

Hitter-per-team check uses `r.max_hitters_per_team` (4 for FD, 5 for DK).
Salary cap uses `r.salary_cap`.

Create `tests/test_fd_lineup.py` — valid/invalid FD 9-player lineups, C/1B slot filling with both C and 1B players, UTIL slot flexibility, salary cap at $35K, max-4 hitter team constraint.

**Files modified:** `src/optimization/lineup.py` | **Created:** `tests/test_fd_lineup.py`

---

## Phase 5 — Upload & Entry Files

**Goal**: Add FD upload/entry file handlers alongside existing DK handlers.

**`src/api/fd_entries.py`** (new): FD upload CSV header:
```
Entry ID, Contest Name, Contest ID, Entry Fee, P, C/1B, 2B, 3B, SS, OF, OF, OF, UTIL
```
FD format uses `"Name (Id)"` player column values, not bare integer IDs. `assign_players_to_fd_slots()` uses `FD_SLOT_ELIGIBILITY` for bipartite matching. `parse_fd_entry_file()` — mark with `# TODO: verify with real FD entry file` until confirmed.

**`src/api/entries_factory.py`** (new): `get_entry_handlers(platform)` returns dict of `{scan, parse, assign, write}` callables for the selected platform.

Create `tests/test_fd_entries.py` — FD upload CSV header, `Name (Id)` column format, slot assignment bipartite matching.

**Files created:** `src/api/fd_entries.py`, `src/api/entries_factory.py`, `tests/test_fd_entries.py`

---

## Phase 6 — Projection Fetch Scripts

**Goal**: Add `--platform` flag to `fetch_rotowire_projections.py` and `fetch_dff_projections.py`. DK behavior unchanged when flag is absent.

**`scripts/fetch_rotowire_projections.py`:**
- Add `FANDUEL_SITE_ID = 2` constant (mark `# TODO: confirm from RotoWire JS bundle`)
- Add `--platform {draftkings,fanduel}` CLI arg; pass appropriate `siteID` to API calls
- Refactor `build_projections_csv()` to accept a pre-loaded `slate_df` (from factory ingestor) rather than a raw CSV path — this removes the internal DK column-name assumptions

**`scripts/fetch_dff_projections.py`:**
- Add `FD_URL_SEGMENT = "fanduel"` constant (vs current `"draftkings"`)
- Add `--platform` arg; navigate to `/mlb/projections/fanduel` for FD
- Mark FD DFF column parsing with `# TODO: verify FD DFF column structure`

**`src/api/projections_meta.py`:** `fetch_and_cache_slates()` accepts `site_id` param; `compute_freshness()` uses platform-appropriate slate path.

**`src/api/server.py`** (projections endpoints only): pass `--platform {cfg.platform.value}` and the platform-appropriate slate path to script subprocess calls.

**Files modified:** `scripts/fetch_rotowire_projections.py`, `scripts/fetch_dff_projections.py`, `src/api/projections_meta.py`, `src/api/server.py` (projections endpoint block)

---

## Phase 7 — Pipeline Wiring

**Goal**: Wire all factories through `PipelineRunner`; update server's `_load_slate_df()`.

**`src/api/pipeline.py`** key changes (around lines 84–106):
```python
platform = Platform(cfg.get("platform", "draftkings"))
roster_rules = get_roster(platform)
slot_eligibility = get_slot_eligibility(platform)
slate_path = paths["dk_slate"] if platform == Platform.DRAFTKINGS else paths["fd_slate"]
ingestor = get_ingestor(platform, slate_path)
slate_df = ingestor.get_slate_dataframe()
entry_handlers = get_entry_handlers(platform)
```

Pass `rules=roster_rules, slot_eligibility=slot_eligibility` through to `Lineup.is_valid()` calls. This requires `PortfolioConstructor` in `src/optimization/portfolio.py` to accept and forward these params.

**`src/api/server.py`** — `_load_slate_df()`: switch on `read_config().platform` to select ingestor via factory.

**`src/optimization/portfolio.py`**: `PortfolioConstructor.__init__` accepts `rules: Optional[RosterRules] = None, slot_eligibility: Optional[dict] = None`; forwards to all `Lineup.is_valid()` calls.

**Files modified:** `src/api/pipeline.py`, `src/api/server.py` (`_load_slate_df` only), `src/optimization/portfolio.py`

---

## Phase 8 — UI Updates

**Goal**: Platform selector in ConfigForm; propagate platform to SlatePanel (with cache reset) and PortfolioTable (FD slot order); ProjectionsPanel stays largely unchanged.

**`ui/src/types.ts`:**
```typescript
export type PlatformType = 'draftkings' | 'fanduel'
interface AppConfig { platform: PlatformType; ... }
interface PathsConfig { dk_slate: string; fd_slate: string; ... }
```

**`ui/src/components/ConfigForm.tsx`:**
- Add platform `<select>` at top of form (before Projections section)
- Conditionally show `dk_slate` or `fd_slate` path input based on current platform
- On platform change, auto-adjust `optimizer.salary_floor` default ($48,500 DK → $30,000 FD) as a UX hint (field remains editable)

**`ui/src/components/SlatePanel.tsx`:**
- Add `platform: PlatformType` prop
- Change `useEffect([], [])` to `useEffect([platform])` — clear local state and re-fetch when platform changes (the server returns a new `slate_id` from the new slate CSV, so exclusion state naturally resets)

**`ui/src/components/PortfolioTable.tsx`:**
- Add `platform?: PlatformType` prop
- `sortPlayersByPosition` uses FD position order when `platform === 'fanduel'`: `['C/1B','2B','3B','SS','OF','OF','OF','UTIL']` (1 pitcher already sorted first)

**`ui/src/App.tsx`:** Pass `config.platform` as prop to `SlatePanel` and `PortfolioTable`.

**`ui/src/components/ProjectionsPanel.tsx`:** No functional change required; fetch log lines from the scripts will show platform context. Optionally add platform badge next to status.

**Files modified:** `ui/src/types.ts`, `ui/src/components/ConfigForm.tsx`, `ui/src/components/SlatePanel.tsx`, `ui/src/components/PortfolioTable.tsx`, `ui/src/App.tsx`

---

## Phase 9 — Historical Pipeline (Deferred)

The copula model and PCA batter model are trained on DK scoring distributions from `historical_logs.parquet`. FD has meaningfully different scoring (e.g. HR=12 vs 10, IP=3 vs 2.25) — the rank-correlation structure differs.

**Do not change `src/ingestion/retrosheet_parser.py` until this phase.** It imports DK constants from `scoring.py` directly; this is the production copula's dependency anchor.

When FD historical data is needed:
1. Add `--platform` to `scripts/process_historical.py` — pass a `ScoringRules` instance to `RetrosheetParser` rather than hard-importing constants; output to `historical_logs_fd.parquet`
2. Run `build_copula.py --platform fanduel` → `copula_fd.parquet`
3. Run `fit_batter_pca.py --platform fanduel` → `batter_pca_model_fd.npz`, `batter_score_grid_fd.npy`
4. Users set `paths.copula`, `paths.batter_pca_model`, `paths.batter_score_grid` to FD files in config

**Files modified (when executed):** `scripts/process_historical.py`, `src/ingestion/retrosheet_parser.py`, `scripts/build_copula.py`, `scripts/fit_batter_pca.py`

---

## Phase 10 — Integration Test & Config Documentation

Create `tests/test_fd_pipeline_integration.py`: synthetic FD CSV fixture → `FanDuelSlateIngestor` → `Lineup.is_valid()` with FD rules → `write_fd_upload_files()` produces correct header. Does not require copula files.

Update `config.yaml` with commented FD config example:
```yaml
# platform: draftkings or fanduel
platform: draftkings
paths:
  dk_slate: data/raw/DKSalaries.csv
  fd_slate: ""       # path to FD salary CSV when platform: fanduel
  # For FanDuel use FD-specific copula (requires Phase 9 rebuild):
  # copula: data/processed/copula_fd.parquet
```

---

## Dependency Order

```
Phase 0 (safety)
    ↓
Phase 1 (platform dataclasses)
    ↓
Phase 2 (config model)          Phase 3 (ingestion)    Phase 4 (lineup)
                    \                   |                    /
                     ↓                 ↓                   ↓
                          Phase 5 (upload files)
                                  ↓
                          Phase 6 (proj scripts)
                                  ↓
                          Phase 7 (pipeline wiring)
                                  ↓
                          Phase 8 (UI)
                                  ↓
          Phase 9 (historical — deferred)   Phase 10 (integration tests)
```

Phases 3, 4 can be parallelized after Phase 1. Phases 5, 6 can be parallelized after 3+4.

---

## Risk Register

| Risk | Phase | Mitigation |
|---|---|---|
| FD CSV column names differ from assumed | 3 | `# VERIFY` marker; confirm before Phase 7 merge |
| FD RotoWire siteID unconfirmed | 6 | `# TODO` marker; check RotoWire JS bundle |
| FD DFF page structure differs | 6 | Mark as best-effort; RotoWire is primary fetch source |
| FD entry file format unknown | 5 | `# TODO` in `parse_fd_entry_file`; defer until real file obtained |
| UTIL slot bipartite matching edge case | 4 | Unit tests: C fills C/1B and UTIL; 1B fills C/1B and UTIL; multiple C players forces correct assignment |
| `PortfolioConstructor` not forwarding rules | 7 | Phase 4 test suite catches this — FD lineup validation fails without correct rules |
| Historical copula silently invalid for FD | 9 | Phase 0 test locks DK constants; `retrosheet_parser.py` explicitly excluded until Phase 9 |
| UI slate state stale after platform switch | 8 | `useEffect([platform])` dependency + `setSlate(null)` on entry |

---

## Files Summary

| Phase | Modified | Created |
|---|---|---|
| 0 | `src/utils/scoring.py` | `tests/test_scoring_contract.py` |
| 1 | — | `src/platforms/{__init__,base,draftkings,fanduel,registry}.py`, `tests/test_platforms.py` |
| 2 | `src/api/models.py`, `src/api/config_io.py` | `tests/test_config_io.py` |
| 3 | `src/ingestion/dk_slate.py` | `src/ingestion/fd_slate.py`, `factory.py`, `tests/test_fd_ingestion.py` |
| 4 | `src/optimization/lineup.py` | `tests/test_fd_lineup.py` |
| 5 | — | `src/api/fd_entries.py`, `entries_factory.py`, `tests/test_fd_entries.py` |
| 6 | `scripts/fetch_rotowire_projections.py`, `scripts/fetch_dff_projections.py`, `src/api/projections_meta.py`, `src/api/server.py` (proj endpoints) | — |
| 7 | `src/api/pipeline.py`, `src/api/server.py` (`_load_slate_df`), `src/optimization/portfolio.py` | — |
| 8 | `ui/src/types.ts`, `ConfigForm.tsx`, `SlatePanel.tsx`, `PortfolioTable.tsx`, `App.tsx` | — |
| 9 (deferred) | `scripts/process_historical.py`, `retrosheet_parser.py`, `build_copula.py`, `fit_batter_pca.py` | — |
| 10 | `config.yaml` | `tests/test_fd_pipeline_integration.py` |
