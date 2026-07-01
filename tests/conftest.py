"""Shared fixtures for BioVenture tests."""

from pathlib import Path

import pytest

from bioventure.config import BioVentureConfig
from bioventure.rng import RNGFactory

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


@pytest.fixture
def base_config() -> BioVentureConfig:
    return BioVentureConfig.from_yaml(CONFIGS_DIR / "base.yaml")


@pytest.fixture
def rng_factory() -> RNGFactory:
    return RNGFactory(seed=12345)
