"""Read and write config.yaml."""
import yaml
from pathlib import Path
from .models import AppConfig

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def read_config() -> AppConfig:
    """Load config.yaml into AppConfig.

    Sections are taken from `AppConfig.model_fields` rather than listed by
    hand. The hand-written list silently dropped any section added to the model
    later: `write_config` dumps the whole model, so a new section WAS written to
    disk, but the next read discarded it and returned the field defaults. The
    UI then showed the default, and its next save -- which builds its payload
    from a fresh GET -- wrote that default back over the real value. A setting
    could not survive a round trip, and a hand-edit to config.yaml was reverted
    by the next Save while the pipeline (which reads the raw YAML directly) was
    meanwhile honouring it. Enumerating the model keeps that from recurring.

    A section present but empty (`gpp:` with nothing under it) parses as None,
    which Pydantic rejects, so those fall back to the field default.
    """
    if not CONFIG_PATH.exists():
        return AppConfig()
    with open(CONFIG_PATH) as f:
        raw = yaml.safe_load(f) or {}
    kwargs = {
        name: raw[name]
        for name in AppConfig.model_fields
        if name in raw and raw[name] is not None
    }
    kwargs.setdefault("platform", "draftkings")
    return AppConfig(**kwargs)


def write_config(cfg: AppConfig) -> None:
    data = cfg.model_dump(exclude_none=False)
    # Serialize Platform enum to its string value for YAML round-trips.
    data["platform"] = cfg.platform.value
    # Represent None values as empty strings for paths, omit optional nones
    paths = data["paths"]
    for key in ("projections", "fd_projections", "batter_pca_model", "batter_score_grid",
                "batter_pca_model_fd", "batter_score_grid_fd"):
        if paths[key] is None:
            paths[key] = ""
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
