"""Monte Carlo simulation engine.

Orchestrates the full BioVenture loop per iteration:

    params  = sampler.draw(rng)
    cum_pos = ∏ clinical phase probabilities
    compr   = rd_model.compress(ai_cf, cost_rf)
    curve   = adoption_model.compute_curve(p, q, n_steps)
    market  = market_model.evolve(process, curve, t, dt, iter_rng)
    portf   = portfolio_model.simulate(iter_rng, cum_pos)
    recorder.record(i, params, cum_pos, compr, market, portf)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from bioventure.config import BioVentureConfig
from bioventure.models.adoption import AdoptionModel
from bioventure.models.market import MarketModel
from bioventure.models.rd_compression import RDCompressionModel
from bioventure.processes.registry import get_process
from bioventure.rng import RNGFactory
from bioventure.simulation.recorder import Recorder
from bioventure.simulation.sampler import ParameterSampler
from bioventure.valuation.payoff import LognormalPayoff, build_payoff
from bioventure.valuation.portfolio import PortfolioModel


_CLINICAL_KEYS = (
    "clinical.phase_1_to_2",
    "clinical.phase_2_to_3",
    "clinical.phase_3_to_approval",
    "clinical.approval_to_launch",
)


class SimulationEngine:
    """Orchestrate the BioVenture Monte Carlo simulation.

    Parameters
    ----------
    config : BioVentureConfig
        Frozen configuration from ``base.yaml``.
    sampler : ParameterSampler
        Draws correlated parameter vectors per iteration.
    """

    def __init__(
        self,
        config: BioVentureConfig,
        sampler: ParameterSampler,
    ) -> None:
        self._cfg = config
        self._sampler = sampler

        sim = config.simulation
        rd = config.rd_compression
        adopt = config.adoption
        mkt = config.market
        val = config.valuation

        self._n = sim.n_iterations
        self._n_steps = sim.n_steps
        self._t = float(self._n_steps)
        self._dt = sim.dt

        self._rd_model = RDCompressionModel(
            rd.baseline_timeline_years,
            rd.baseline_cost_billion_usd,
        )
        self._adoption_model = AdoptionModel(
            adopt.ceiling,
            adopt.start_fraction,
        )
        self._market_model = MarketModel(mkt.anchor_2025_billion_usd)

        self._process_name = mkt.process
        self._process_params = dict(mkt.process_params)

        self._n_bets = val.portfolio.n_bets
        self._bet_size = val.portfolio.bet_size_million_usd
        self._payoff_dist = val.payoff.distribution
        self._payoff_params = dict(val.payoff.params)

    @property
    def config(self) -> BioVentureConfig:
        return self._cfg

    @property
    def n_iterations(self) -> int:
        return self._n

    def run(
        self,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Recorder:
        """Execute the full Monte Carlo simulation.

        Parameters
        ----------
        progress_callback : callable or None
            Called as ``progress_callback(i, total)`` after each
            iteration. Useful for progress bars; set to ``None``
            (default) to disable.

        Returns
        -------
        Recorder
            Fully populated with all iteration results.
        """
        factory = RNGFactory(self._cfg.simulation.seed)
        sampler_rng, seed_rng = factory.spawn(2)

        recorder = Recorder(
            n_iterations=self._n,
            n_steps=self._n_steps,
            param_names=self._sampler.parameter_names,
        )

        for i in range(self._n):
            iter_seed = int(seed_rng.integers(0, 2**31))
            iter_rng = np.random.default_rng(iter_seed)

            params = self._sampler.draw(sampler_rng)

            cum_pos = self._cumulative_pos(params)

            compression = self._rd_model.compress(
                ai_compression_factor=params[
                    "rd_compression.ai_compression_factor"
                ],
                cost_reduction_factor=params[
                    "rd_compression.cost_reduction_factor"
                ],
            )

            adoption_curve = self._adoption_model.compute_curve(
                p=params["adoption.bass_p"],
                q=params["adoption.bass_q"],
                n_years=self._n_steps,
            )

            process = self._build_process(params)
            market = self._market_model.evolve(
                process=process,
                adoption_curve=adoption_curve,
                t=self._t,
                dt=self._dt,
                rng=iter_rng,
            )

            payoff = self._build_payoff(params)
            portfolio_model = PortfolioModel(
                n_bets=self._n_bets,
                bet_size_million_usd=self._bet_size,
                payoff=payoff,
            )
            portfolio = portfolio_model.simulate(
                iter_rng, success_probability=cum_pos
            )

            recorder.record(
                i=i,
                params=params,
                cumulative_pos=cum_pos,
                compression=compression,
                market=market,
                portfolio=portfolio,
            )

            if progress_callback is not None:
                progress_callback(i, self._n)

        return recorder

    def _cumulative_pos(self, params: dict[str, float]) -> float:
        """Product of the four drawn clinical phase probabilities."""
        cum = 1.0
        for key in _CLINICAL_KEYS:
            cum *= params[key]
        return cum

    def _build_process(self, params: dict[str, float]):
        """Build the stochastic market process for one iteration.

        GBM uses per-iteration sampled drift and volatility.
        Other processes use the static params from base.yaml.
        """
        name = self._process_name
        if name == "gbm":
            return get_process("gbm", {
                "drift": params.get(
                    "market.drift",
                    self._process_params.get("drift", 0.12),
                ),
                "volatility": params.get(
                    "market.volatility",
                    self._process_params.get("volatility", 0.18),
                ),
            })
        return get_process(name, self._process_params)

    def _build_payoff(self, params: dict[str, float]):
        """Build the payoff distribution for one iteration.

        Lognormal uses per-iteration sampled mu and sigma.
        Other payoff types use the static params from base.yaml.
        """
        dist = self._payoff_dist
        if dist == "lognormal":
            return LognormalPayoff(
                mu=params.get(
                    "valuation.exit_multiple_mu",
                    self._payoff_params.get("mu", 0.7),
                ),
                sigma=params.get(
                    "valuation.exit_multiple_sigma",
                    self._payoff_params.get("sigma", 1.2),
                ),
            )
        return build_payoff(dist, self._payoff_params)

    @classmethod
    def from_configs(
        cls,
        base_yaml: str | Path,
        priors_yaml: str | Path,
        correlations_yaml: str | Path,
        sampler: ParameterSampler | None = None,
    ) -> SimulationEngine:
        """Build an engine from YAML config files.

        Parameters
        ----------
        base_yaml : path-like
            Path to ``configs/base.yaml``.
        priors_yaml : path-like
            Path to ``configs/priors.yaml``.
        correlations_yaml : path-like
            Path to ``configs/correlations.yaml``.
        sampler : ParameterSampler or None
            Pre-built sampler; built from YAML files if ``None``.
        """
        config = BioVentureConfig.from_yaml(base_yaml)
        if sampler is None:
            sampler = ParameterSampler.from_configs(priors_yaml, correlations_yaml)
        return cls(config, sampler)
    