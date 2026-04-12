"""
Tests for src/api/config_io.py (Phase 2).

Coverage:
- Round-trip write/read preserves platform: fanduel
- Round-trip write/read preserves platform: draftkings
- Existing configs without a 'platform' key default to draftkings
- fd_slate is written and read back correctly
- Default AppConfig has platform=draftkings and fd_slate=""
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.api.config_io import read_config, write_config
from src.api.models import AppConfig, PathsConfig
from src.platforms.base import Platform


def _patched_config_path(tmp_path: Path):
    """Context manager: redirect CONFIG_PATH to a temp file."""
    tmp_file = tmp_path / "config.yaml"
    return patch("src.api.config_io.CONFIG_PATH", tmp_file)


class TestDefaults:
    def test_default_platform_is_draftkings(self):
        cfg = AppConfig()
        assert cfg.platform == Platform.DRAFTKINGS

    def test_default_fd_slate_is_empty(self):
        cfg = AppConfig()
        assert cfg.paths.fd_slate == ""

    def test_read_config_no_file_returns_default(self, tmp_path):
        with _patched_config_path(tmp_path):
            cfg = read_config()
        assert cfg.platform == Platform.DRAFTKINGS
        assert cfg.paths.fd_slate == ""


class TestPlatformRoundTrip:
    def test_fanduel_round_trip(self, tmp_path):
        """Write platform=fanduel, read back, assert it survives."""
        cfg = AppConfig(platform=Platform.FANDUEL)
        with _patched_config_path(tmp_path):
            write_config(cfg)
            result = read_config()
        assert result.platform == Platform.FANDUEL

    def test_draftkings_round_trip(self, tmp_path):
        """Write platform=draftkings, read back."""
        cfg = AppConfig(platform=Platform.DRAFTKINGS)
        with _patched_config_path(tmp_path):
            write_config(cfg)
            result = read_config()
        assert result.platform == Platform.DRAFTKINGS

    def test_platform_written_as_plain_string(self, tmp_path):
        """YAML must store the value as a plain string, not an enum repr."""
        import yaml

        cfg = AppConfig(platform=Platform.FANDUEL)
        tmp_file = tmp_path / "config.yaml"
        with patch("src.api.config_io.CONFIG_PATH", tmp_file):
            write_config(cfg)
        with open(tmp_file) as f:
            raw = yaml.safe_load(f)
        assert raw["platform"] == "fanduel"
        assert isinstance(raw["platform"], str)


class TestFdSlateRoundTrip:
    def test_fd_slate_path_survives_round_trip(self, tmp_path):
        cfg = AppConfig(paths=PathsConfig(fd_slate="data/raw/FDSalaries.csv"))
        with _patched_config_path(tmp_path):
            write_config(cfg)
            result = read_config()
        assert result.paths.fd_slate == "data/raw/FDSalaries.csv"

    def test_fd_slate_empty_string_survives_round_trip(self, tmp_path):
        cfg = AppConfig(paths=PathsConfig(fd_slate=""))
        with _patched_config_path(tmp_path):
            write_config(cfg)
            result = read_config()
        assert result.paths.fd_slate == ""


class TestMissingPlatformKey:
    def test_yaml_without_platform_key_defaults_to_draftkings(self, tmp_path):
        """Simulate a config.yaml written before Phase 2 (no platform key)."""
        import yaml

        tmp_file = tmp_path / "config.yaml"
        legacy_data = {
            "paths": {"dk_slate": "data/raw/DKSalaries.csv", "copula": ""},
            "simulation": {"n_sims": 10000},
            "optimizer": {},
            "portfolio": {},
        }
        with open(tmp_file, "w") as f:
            yaml.dump(legacy_data, f)

        with patch("src.api.config_io.CONFIG_PATH", tmp_file):
            cfg = read_config()

        assert cfg.platform == Platform.DRAFTKINGS

    def test_yaml_without_fd_slate_key_defaults_to_empty(self, tmp_path):
        """Simulate a config.yaml written before fd_slate was added."""
        import yaml

        tmp_file = tmp_path / "config.yaml"
        legacy_data = {
            "platform": "draftkings",
            "paths": {"dk_slate": "data/raw/DKSalaries.csv"},
            "simulation": {},
            "optimizer": {},
            "portfolio": {},
        }
        with open(tmp_file, "w") as f:
            yaml.dump(legacy_data, f)

        with patch("src.api.config_io.CONFIG_PATH", tmp_file):
            cfg = read_config()

        assert cfg.paths.fd_slate == ""


class TestFullRoundTrip:
    def test_combined_fanduel_config(self, tmp_path):
        """Full round-trip: platform=fanduel + fd_slate path + other fields intact."""
        cfg = AppConfig(
            platform=Platform.FANDUEL,
            paths=PathsConfig(
                dk_slate="data/raw/DKSalaries.csv",
                fd_slate="data/raw/FDSalaries.csv",
                copula="data/processed/empirical_copula.parquet",
            ),
        )
        with _patched_config_path(tmp_path):
            write_config(cfg)
            result = read_config()

        assert result.platform == Platform.FANDUEL
        assert result.paths.fd_slate == "data/raw/FDSalaries.csv"
        assert result.paths.dk_slate == "data/raw/DKSalaries.csv"
        assert result.paths.copula == "data/processed/empirical_copula.parquet"
