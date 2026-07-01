"""Load and validate YAML configuration into typed, immutable dataclasses."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimulationConfig:
    n_iterations: int
    seed: int
    start_year: int
    end_year: int
    dt: float = 1.0

    def __post_init__(self):
        if self.n_iterations < 1:
            raise ValueError(f"n_iterations must be >= 1, got {self.n_iterations}")
        if self.end_year <= self.start_year:
            raise ValueError(
                f"end_year ({self.end_year}) must be > start_year ({self.start_year})"
            )
        if self.dt <= 0:
            raise ValueError(f"dt must be > 0, got {self.dt}")

    @property
    def n_steps(self) -> int:
        return int((self.end_year - self.start_year) / self.dt)

    @property
    def years(self) -> list[float]:
        return [self.start_year + i * self.dt for i in range(self.n_steps + 1)]


# ---------------------------------------------------------------------------
# Clinical pipeline
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhaseTransition:
    name: str
    probability: float

    def __post_init__(self):
        if not 0.0 < self.probability < 1.0:
            raise ValueError(
                f"Phase '{self.name}' probability must be in (0, 1), got {self.probability}"
            )


@dataclass(frozen=True)
class ClinicalConfig:
    phases: tuple[PhaseTransition, ...]

    @property
    def cumulative_probability(self) -> float:
        p = 1.0
        for phase in self.phases:
            p *= phase.probability
        return p


# ---------------------------------------------------------------------------
# R&D compression
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RDCompressionConfig:
    baseline_timeline_years: float
    baseline_cost_billion_usd: float
    ai_compression_factor: float
    cost_reduction_factor: float

    def __post_init__(self):
        if not 0.0 < self.ai_compression_factor < 1.0:
            raise ValueError(
                f"ai_compression_factor must be in (0, 1), got {self.ai_compression_factor}"
            )
        if not 0.0 < self.cost_reduction_factor < 1.0:
            raise ValueError(
                f"cost_reduction_factor must be in (0, 1), got {self.cost_reduction_factor}"
            )


# ---------------------------------------------------------------------------
# Adoption
# ---------------------------------------------------------------------------

_ADOPTION_MODELS = ("bass",)


@dataclass(frozen=True)
class AdoptionConfig:
    model: str
    bass_p: float
    bass_q: float
    ceiling: float
    start_fraction: float

    def __post_init__(self):
        if self.model not in _ADOPTION_MODELS:
            raise ValueError(f"Unknown adoption model '{self.model}', expected {_ADOPTION_MODELS}")
        if not 0.0 < self.ceiling <= 1.0:
            raise ValueError(f"ceiling must be in (0, 1], got {self.ceiling}")
        if not 0.0 <= self.start_fraction < self.ceiling:
            raise ValueError(
                f"start_fraction must be in [0, ceiling={self.ceiling}), "
                f"got {self.start_fraction}"
            )


# ---------------------------------------------------------------------------
# Market
# ---------------------------------------------------------------------------

_PROCESS_TYPES = ("gbm", "jump_diffusion", "regime_switching")


@dataclass(frozen=True)
class MarketConfig:
    anchor_2025_billion_usd: float
    process: str
    process_params: dict[str, Any]

    def __post_init__(self):
        if self.anchor_2025_billion_usd <= 0:
            raise ValueError("anchor_2025_billion_usd must be > 0")
        if self.process not in _PROCESS_TYPES:
            raise ValueError(f"Unknown process '{self.process}', expected {_PROCESS_TYPES}")


# ---------------------------------------------------------------------------
# Valuation
# ---------------------------------------------------------------------------

_PAYOFF_DISTRIBUTIONS = ("lognormal", "pareto")


@dataclass(frozen=True)
class PayoffConfig:
    distribution: str
    params: dict[str, float]

    def __post_init__(self):
        if self.distribution not in _PAYOFF_DISTRIBUTIONS:
            raise ValueError(
                f"Unknown payoff distribution '{self.distribution}', "
                f"expected {_PAYOFF_DISTRIBUTIONS}"
            )


@dataclass(frozen=True)
class PortfolioConfig:
    n_bets: int
    bet_size_million_usd: float

    def __post_init__(self):
        if self.n_bets < 1:
            raise ValueError(f"n_bets must be >= 1, got {self.n_bets}")
        if self.bet_size_million_usd <= 0:
            raise ValueError(f"bet_size_million_usd must be > 0, got {self.bet_size_million_usd}")


@dataclass(frozen=True)
class DiscountConfig:
    risk_free_rate: float
    risk_premium: float

    @property
    def discount_rate(self) -> float:
        return self.risk_free_rate + self.risk_premium


@dataclass(frozen=True)
class ValuationConfig:
    payoff: PayoffConfig
    portfolio: PortfolioConfig
    discount: DiscountConfig


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BioVentureConfig:
    simulation: SimulationConfig
    clinical: ClinicalConfig
    rd_compression: RDCompressionConfig
    adoption: AdoptionConfig
    market: MarketConfig
    valuation: ValuationConfig

    def config_hash(self) -> str:
        raw = json.dumps(dataclasses.asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @classmethod
    def from_yaml(cls, path: str | Path) -> BioVentureConfig:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, d: dict) -> BioVentureConfig:
        sim = d["simulation"]
        clin = d["clinical"]
        rd = d["rd_compression"]
        adopt = d["adoption"]
        mkt = d["market"]
        val = d["valuation"]

        return cls(
            simulation=SimulationConfig(
                n_iterations=sim["n_iterations"],
                seed=sim["seed"],
                start_year=sim["start_year"],
                end_year=sim["end_year"],
                dt=sim.get("dt", 1.0),
            ),
            clinical=ClinicalConfig(
                phases=tuple(
                    PhaseTransition(name=p["name"], probability=p["probability"])
                    for p in clin["phases"]
                ),
            ),
            rd_compression=RDCompressionConfig(
                baseline_timeline_years=rd["baseline_timeline_years"],
                baseline_cost_billion_usd=rd["baseline_cost_billion_usd"],
                ai_compression_factor=rd["ai_compression_factor"],
                cost_reduction_factor=rd["cost_reduction_factor"],
            ),
            adoption=AdoptionConfig(
                model=adopt["model"],
                bass_p=adopt["bass_p"],
                bass_q=adopt["bass_q"],
                ceiling=adopt["ceiling"],
                start_fraction=adopt["start_fraction"],
            ),
            market=MarketConfig(
                anchor_2025_billion_usd=mkt["anchor_2025_billion_usd"],
                process=mkt["process"],
                process_params=dict(mkt.get("process_params", {})),
            ),
            valuation=ValuationConfig(
                payoff=PayoffConfig(
                    distribution=val["payoff"]["distribution"],
                    params=dict(val["payoff"]["params"]),
                ),
                portfolio=PortfolioConfig(
                    n_bets=val["portfolio"]["n_bets"],
                    bet_size_million_usd=val["portfolio"]["bet_size_million_usd"],
                ),
                discount=DiscountConfig(
                    risk_free_rate=val["discount"]["risk_free_rate"],
                    risk_premium=val["discount"]["risk_premium"],
                ),
            ),
        )
