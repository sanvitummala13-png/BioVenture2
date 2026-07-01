"""Clinical pipeline model: phase-transition probabilities and pipeline yield.

Each drug candidate passes through sequential phases (I → II → III → Approval
→ Launch). Each transition is drawn from a Beta distribution whose
hyperparameters encode historical success rates and sample-size confidence.

The model outputs:
- Per-phase success draws (independent Bernoulli trials per candidate).
- Cumulative probability of success (product of phase draws).
- Pipeline yield: expected number of launches from a starting cohort.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator


@dataclass(frozen=True)
class PhaseResult:
    """Results of a single Monte Carlo draw across the clinical pipeline.

    Attributes
    ----------
    phase_probabilities : np.ndarray
        Shape ``(n_phases,)`` — drawn success probability per phase.
    cumulative_pos : float
        Product of all phase probabilities (probability of full success).
    pipeline_yield : float
        Expected launches from a cohort: ``cohort_size * cumulative_pos``.
    """

    phase_probabilities: np.ndarray
    cumulative_pos: float
    pipeline_yield: float


class ClinicalPipeline:
    """Stochastic clinical pipeline with Beta-distributed phase transitions.

    Parameters
    ----------
    phase_names : list[str]
        Human-readable label for each phase transition.
    alpha : np.ndarray
        Beta distribution α parameters (one per phase).
    beta_ : np.ndarray
        Beta distribution β parameters (one per phase).
    cohort_size : int
        Number of candidates entering Phase I per simulation draw.
    """

    def __init__(
        self,
        phase_names: list[str],
        alpha: np.ndarray,
        beta_: np.ndarray,
        cohort_size: int = 100,
    ) -> None:
        alpha = np.asarray(alpha, dtype=np.float64)
        beta_ = np.asarray(beta_, dtype=np.float64)

        if alpha.ndim != 1 or beta_.ndim != 1:
            raise ValueError("alpha and beta must be 1-D arrays")
        if len(alpha) != len(beta_):
            raise ValueError(
                f"alpha and beta must have same length, "
                f"got {len(alpha)} and {len(beta_)}"
            )
        if len(phase_names) != len(alpha):
            raise ValueError(
                f"phase_names length ({len(phase_names)}) must match "
                f"alpha length ({len(alpha)})"
            )
        if np.any(alpha <= 0) or np.any(beta_ <= 0):
            raise ValueError("All alpha and beta values must be > 0")
        if cohort_size < 1:
            raise ValueError(f"cohort_size must be >= 1, got {cohort_size}")

        self._phase_names = list(phase_names)
        self._alpha = alpha
        self._beta = beta_
        self._n_phases = len(alpha)
        self._cohort_size = cohort_size

    @property
    def n_phases(self) -> int:
        return self._n_phases

    @property
    def phase_names(self) -> list[str]:
        return list(self._phase_names)

    @property
    def cohort_size(self) -> int:
        return self._cohort_size

    @property
    def expected_probabilities(self) -> np.ndarray:
        """Prior mean of each phase: α / (α + β)."""
        return self._alpha / (self._alpha + self._beta)

    @property
    def expected_cumulative_pos(self) -> float:
        """Product of prior means — point estimate of full-pipeline success."""
        return float(np.prod(self.expected_probabilities))

    def draw(self, rng: Generator) -> PhaseResult:
        """Single Monte Carlo draw of phase-transition probabilities.

        Parameters
        ----------
        rng : Generator
            NumPy random generator.

        Returns
        -------
        PhaseResult
            One realisation of the pipeline.
        """
        probs = rng.beta(self._alpha, self._beta)
        cum = float(np.prod(probs))
        return PhaseResult(
            phase_probabilities=probs,
            cumulative_pos=cum,
            pipeline_yield=self._cohort_size * cum,
        )

    def draw_batch(self, n: int, rng: Generator) -> dict[str, np.ndarray]:
        """Vectorised batch draw.

        Parameters
        ----------
        n : int
            Number of Monte Carlo iterations.
        rng : Generator
            NumPy random generator.

        Returns
        -------
        dict[str, np.ndarray]
            ``"phase_probabilities"`` : shape ``(n, n_phases)``
            ``"cumulative_pos"``      : shape ``(n,)``
            ``"pipeline_yield"``      : shape ``(n,)``
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        probs = rng.beta(self._alpha, self._beta, size=(n, self._n_phases))
        cum = np.prod(probs, axis=1)
        return {
            "phase_probabilities": probs,
            "cumulative_pos": cum,
            "pipeline_yield": self._cohort_size * cum,
        }

    @classmethod
    def from_config(
        cls,
        phases: list[dict],
        prior_params: dict[str, dict],
        cohort_size: int = 100,
    ) -> ClinicalPipeline:
        """Build from config structures.

        Parameters
        ----------
        phases : list[dict]
            From ``base.yaml`` — each dict has ``"name"`` and ``"probability"``.
        prior_params : dict[str, dict]
            From ``priors.yaml`` — keyed by dotted name (e.g.
            ``"clinical.phase_1_to_2"``), each value has ``"a"`` and ``"b"``.
        cohort_size : int
            Candidates entering the pipeline.
        """
        names = [p["name"] for p in phases]
        prior_keys = sorted(k for k in prior_params if k.startswith("clinical."))
        if len(prior_keys) != len(phases):
            raise ValueError(
                f"Expected {len(phases)} clinical priors, found {len(prior_keys)}: "
                f"{prior_keys}"
            )
        alpha = np.array([prior_params[k]["a"] for k in prior_keys])
        beta_ = np.array([prior_params[k]["b"] for k in prior_keys])
        return cls(names, alpha, beta_, cohort_size)
    