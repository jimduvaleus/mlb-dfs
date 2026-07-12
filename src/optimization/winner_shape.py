"""Sim-free "winner shape" model (ceiling-first redesign, Phase 2d).

An L2 logistic regression on lineup composition features, fitted on real DK
contest entries (label: finished in that contest's top 1%) across the slate
archive by scripts/fit_winner_shape.py. Because it is fitted on what real
top-1% lineups *look like* — not on what the simulator thinks a good lineup
is — it is the one ranking currency immune to the recurring "model p99
worlds != real p99 worlds" failure mode.

Features are z-scored within the population being scored (per slate at fit
time, per candidate pool at inference). That per-population standardization
is also the guard against the known ownership shift: the model is fitted on
real %Drafted but scored on model ownership, and only the *relative* spread
within a slate carries signal across that boundary.

Artifact: data/processed/winner_shape_model.json —
    {"feature_names": [...],
     "models": {cutoff_date "MMDDYYYY": {"coef": [...], "intercept": x,
                "n_slates": k, "n_entries": m}},
     "latest": cutoff_date}
Walk-forward contract: the model keyed by cutoff date D was fitted only on
slates strictly before D. Replays of slate D must use that key (or the
newest key before D); the live pipeline uses "latest".
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("data/processed/winner_shape_model.json")

FEATURE_NAMES = [
    "primary", "secondary", "prim5", "sec2",
    "salary_k", "at_cap", "own_sum", "own_min", "bringback",
]


def lineup_features(records: "list[dict] | object") -> np.ndarray:
    """(N, len(FEATURE_NAMES)) raw feature matrix from composition records
    (dicts or a DataFrame with the _real_entry_records/_pool_records schema:
    primary, secondary, salary, own_sum, own_min, bringback)."""
    import pandas as pd
    df = pd.DataFrame(records)
    X = np.column_stack([
        df["primary"].astype(float),
        df["secondary"].astype(float),
        (df["primary"] >= 5).astype(float),
        (df["secondary"] >= 2).astype(float),
        df["salary"].astype(float) / 1000.0,
        (df["salary"].astype(float) >= 50_000).astype(float),
        df["own_sum"].astype(float),
        df["own_min"].astype(float),
        df["bringback"].astype(float),
    ])
    return X


def standardize(X: np.ndarray) -> np.ndarray:
    """Z-score each column within the population being scored."""
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (X - mu) / sd


def fit_logistic_irls(
    X: np.ndarray, y: np.ndarray, l2: float = 1.0, max_iter: int = 50, tol: float = 1e-8,
) -> tuple[np.ndarray, float]:
    """L2-regularized logistic regression via IRLS (numpy only — sklearn is
    deliberately not a dependency of this repo). Returns (coef, intercept);
    the intercept is unpenalized."""
    n, d = X.shape
    Xb = np.column_stack([X, np.ones(n)])
    w = np.zeros(d + 1)
    reg = np.eye(d + 1) * float(l2)
    reg[d, d] = 0.0  # unpenalized intercept
    for _ in range(max_iter):
        z = Xb @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))
        W = np.clip(p * (1 - p), 1e-9, None)
        grad = Xb.T @ (y - p) - reg @ w
        H = (Xb * W[:, None]).T @ Xb + reg
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        w = w + step
        if float(np.abs(step).max()) < tol:
            break
    return w[:d], float(w[d])


def load_model(
    model_path: Path = DEFAULT_MODEL_PATH, slate_date: Optional[str] = None,
) -> Optional[dict]:
    """Return {"coef", "intercept", "feature_names"} for a slate.

    slate_date "MMDDYYYY": walk-forward selection — the newest model whose
    cutoff is <= that date (a cutoff-D model saw only slates before D).
    None: the "latest" model (live pipeline).
    """
    if not Path(model_path).exists():
        return None
    try:
        art = json.loads(Path(model_path).read_text())
    except Exception as exc:
        logger.warning("winner_shape: could not read %s (%s)", model_path, exc)
        return None
    models = art.get("models", {})
    if not models:
        return None
    if slate_date is None:
        key = art.get("latest") or max(models, key=lambda k: datetime.strptime(k, "%m%d%Y"))
    else:
        target = datetime.strptime(slate_date[:8], "%m%d%Y")
        eligible = [k for k in models if datetime.strptime(k, "%m%d%Y") <= target]
        if not eligible:
            return None
        key = max(eligible, key=lambda k: datetime.strptime(k, "%m%d%Y"))
    m = models[key]
    return {
        "coef": np.asarray(m["coef"], dtype=np.float64),
        "intercept": float(m["intercept"]),
        "feature_names": art.get("feature_names", FEATURE_NAMES),
        "cutoff": key,
    }


def score_lineups(
    lineups: list,
    players_df,
    ownership_vec: np.ndarray,
    model: dict,
) -> np.ndarray:
    """(M,) winner-shape logits for candidate Lineup objects.

    Ownership comes from the caller's field-ownership vector (the model
    ownership estimate of real %Drafted); the within-pool z-scoring absorbs
    the level difference vs the real %Drafted the model was fitted on.
    """
    pid_team = dict(zip(players_df["player_id"].astype(int), players_df["team"]))
    pid_pos = dict(zip(players_df["player_id"].astype(int), players_df["position"]))
    pid_salary = dict(zip(players_df["player_id"].astype(int), players_df["salary"].astype(float)))
    pid_opp = dict(zip(players_df["player_id"].astype(int), players_df.get("opponent", "")))
    pid_own = dict(zip(players_df["player_id"].astype(int), np.asarray(ownership_vec, dtype=float)))

    records = []
    for lu in lineups:
        team_counts: dict[str, int] = {}
        pitcher_opps = []
        owns = []
        salary = 0.0
        for pid in lu.player_ids:
            pid = int(pid)
            salary += pid_salary.get(pid, 0.0)
            owns.append(pid_own.get(pid, 0.0))
            if pid_pos.get(pid, "") == "P":
                pitcher_opps.append(pid_opp.get(pid, ""))
            else:
                t = pid_team.get(pid, "")
                if t:
                    team_counts[t] = team_counts.get(t, 0) + 1
        counts = sorted(team_counts.values(), reverse=True)
        records.append({
            "primary": counts[0] if counts else 0,
            "secondary": counts[1] if len(counts) > 1 else 0,
            "salary": salary,
            "own_sum": float(sum(owns)),
            "own_min": float(min(owns)) if owns else 0.0,
            "bringback": bool(any(o and o in team_counts for o in pitcher_opps)),
        })
    X = standardize(lineup_features(records))
    return X @ model["coef"] + model["intercept"]
