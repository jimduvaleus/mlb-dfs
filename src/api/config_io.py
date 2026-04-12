"""Read and write config.yaml."""
import yaml
from pathlib import Path
from .models import AppConfig

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def read_config() -> AppConfig:
    if not CONFIG_PATH.exists():
        return AppConfig()
    with open(CONFIG_PATH) as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig(
        platform=raw.get("platform", "draftkings"),
        paths=raw.get("paths", {}),
        simulation=raw.get("simulation", {}),
        optimizer=raw.get("optimizer", {}),
        portfolio=raw.get("portfolio", {}),
    )


def write_config(cfg: AppConfig) -> None:
    data = cfg.model_dump(exclude_none=False)
    # Serialize Platform enum to its string value for YAML round-trips.
    data["platform"] = cfg.platform.value
    # Represent None values as empty strings for paths, omit optional nones
    paths = data["paths"]
    for key in ("projections", "fd_projections", "batter_pca_model", "batter_score_grid"):
        if paths[key] is None:
            paths[key] = ""
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
