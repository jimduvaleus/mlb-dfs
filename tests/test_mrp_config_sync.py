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


# ---------------------------------------------------------------------------
# Run-dialog availability: the mode must be STARTABLE, not just selectable
# ---------------------------------------------------------------------------

def test_only_roi_requires_saber_roi_blocks():
    """RunOptionsDialog disables the "External candidate pool (SaberSim)"
    checkbox when /api/run/cache-status reports the pool unavailable, and it
    reports unavailable when ROI blocks are required but absent. Exports
    commonly have none, so every currency that does not consult Saber's ROI
    must say so -- otherwise the mode is selectable in config but cannot be
    started."""
    from src.api.server import external_roi_blocks_required

    assert external_roi_blocks_required("roi") is True
    for ev in ("prj_own", "p_win", "proj_top", "self_play",
               "topn_coverage", "marginal_reward"):
        assert external_roi_blocks_required(ev) is False, f"{ev} does not consult Saber ROI"


def test_roi_requirement_is_case_and_whitespace_insensitive():
    from src.api.server import external_roi_blocks_required

    assert external_roi_blocks_required("  MARGINAL_REWARD ") is False
    assert external_roi_blocks_required("  ROI ") is True
    assert external_roi_blocks_required("") is True, "empty falls back to the roi default"
    assert external_roi_blocks_required(None) is True


def test_availability_formula_admits_a_pool_with_no_roi_blocks():
    """The concrete regression: 10,066 lineups, 0 ROI blocks -- the shape of
    the live export -- must leave marginal_reward startable."""
    from src.api.server import external_roi_blocks_required

    n_lineups, n_contests = 10_066, 0

    def available(ev):
        return n_lineups > 0 and (n_contests > 0 or not external_roi_blocks_required(ev))

    assert available("marginal_reward") is True
    assert available("topn_coverage") is True
    assert available("roi") is False, "roi genuinely does need the blocks"


def test_roi_requirement_is_not_written_as_an_exclusion_list():
    """Guard the shape, not just today's answer: an exclusion list is what
    went stale, twice."""
    src = (ROOT / "src" / "api" / "server.py").read_text()
    assert "_roi_required = external_roi_blocks_required(" in src
    assert '_ev_type not in ("prj_own"' not in src, "exclusion list reintroduced"
