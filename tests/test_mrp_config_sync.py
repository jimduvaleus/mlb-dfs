"""The three-file config sync for the MRP allocator.

CLAUDE.md: "any field exposed in the UI must also be declared in
src/api/models.py GppConfig. Pydantic silently drops unknown fields on
POST /api/config, so omitting a field there means Save Config is a no-op for
that field. The three places that must stay in sync: config.yaml (default
value), src/api/models.py (Pydantic model + default), ui/src/types.ts."

That failure is silent by construction -- the UI shows the edit, the save
appears to succeed, and the value is gone -- so it gets a test rather than
vigilance.
"""
import re
from pathlib import Path

import pytest
import yaml

from src.api.models import AppConfig, MarginalRewardConfig

ROOT = Path(__file__).resolve().parents[1]
FIELDS = set(MarginalRewardConfig.model_fields)


def test_defaults_match_the_documented_values():
    mr = AppConfig().marginal_reward
    assert mr.gamma_in == 7, "paper's C-3 for a 10-player roster"
    assert mr.gamma_out == 8, "no-op against a pool through the 9/10 cull"
    assert mr.allow_cross_contest_duplicates is False
    assert mr.smooth_tau_scale == 0.0, "exact estimator by default"


def test_posted_values_survive_the_pydantic_model():
    """The exact silent-drop failure CLAUDE.md warns about."""
    posted = {
        "gpp": {"external_pool_ev_type": "marginal_reward"},
        "marginal_reward": {"gamma_in": 5, "gamma_out": 6, "smooth_tau_scale": 1.0,
                            "allow_cross_contest_duplicates": True,
                            "field_pool_size": 9999, "max_sims_per_contest": 8888},
    }
    rt = AppConfig(**posted)
    assert rt.gpp.external_pool_ev_type == "marginal_reward"
    assert rt.marginal_reward.gamma_in == 5
    assert rt.marginal_reward.gamma_out == 6
    assert rt.marginal_reward.smooth_tau_scale == 1.0
    assert rt.marginal_reward.allow_cross_contest_duplicates is True
    assert rt.marginal_reward.field_pool_size == 9999
    assert rt.marginal_reward.max_sims_per_contest == 8888


def test_typescript_interface_matches_the_pydantic_model():
    ts = (ROOT / "ui" / "src" / "types.ts").read_text()
    block = re.search(r"export interface MarginalRewardConfig \{(.*?)\n\}", ts, re.S)
    assert block, "MarginalRewardConfig missing from ui/src/types.ts"
    ts_fields = set(re.findall(r"^\s*(\w+)\??:", block.group(1), re.M))
    assert ts_fields == FIELDS, (
        f"types.ts and models.py disagree; only in TS: {ts_fields - FIELDS}, "
        f"only in Python: {FIELDS - ts_fields}"
    )


def test_typescript_appconfig_carries_the_section():
    ts = (ROOT / "ui" / "src" / "types.ts").read_text()
    block = re.search(r"export interface AppConfig \{(.*?)\n\}", ts, re.S)
    assert block and "marginal_reward" in block.group(1), (
        "AppConfig must carry marginal_reward or the form cannot read or save it"
    )


def test_config_example_documents_every_field():
    cfg = yaml.safe_load((ROOT / "config.example.yaml").read_text())
    assert "marginal_reward" in cfg, "config.example.yaml is the tracked known-good snapshot"
    assert set(cfg["marginal_reward"]) >= FIELDS, (
        f"undocumented in config.example.yaml: {FIELDS - set(cfg['marginal_reward'])}"
    )


def test_pipeline_accepts_marginal_reward_as_an_ev_type():
    """If the validation tuple omits it, the pipeline silently falls back to
    'roi' with only a log line -- the run would succeed and quietly not be MRP."""
    src = (ROOT / "src" / "api" / "pipeline.py").read_text()
    guard = re.search(r"if _ev_type not in \((.*?)\):", src, re.S)
    assert guard and "marginal_reward" in guard.group(1)
    assert 'elif _ev_type == "marginal_reward":' in src, "no allocator branch"


@pytest.mark.parametrize("field", sorted(FIELDS))
def test_every_field_is_read_by_the_pipeline_branch(field):
    """A field that exists in all three places but is never read is a dead
    control that silently does nothing when the user changes it."""
    src = (ROOT / "src" / "api" / "pipeline.py").read_text()
    branch = src.split('elif _ev_type == "marginal_reward":', 1)[1].split("\n        else:", 1)[0]
    assert f'"{field}"' in branch, f"pipeline never reads marginal_reward.{field}"
