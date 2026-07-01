# BioVentureSim

A quantitative research framework for modeling AI-enabled drug discovery markets using stochastic processes, Bayesian inference, Monte Carlo simulation, and machine learning.

---

## Overview

BioVentureSim is a modular Python framework designed to model the long-term dynamics of AI-enabled drug discovery markets under uncertainty.

Rather than producing a single deterministic forecast, the framework generates thousands of simulated market trajectories to quantify commercialization risk, market volatility, and long-term growth. The project combines methods from quantitative finance, computational biology, statistics, and machine learning to explore how scientific, regulatory, and economic factors influence emerging biotechnology markets.

---

## Research Question

**How can stochastic modeling and Monte Carlo simulation improve our understanding of uncertainty in AI-enabled drug discovery markets?**

---

## Methodology

The framework integrates techniques from multiple disciplines, including:

- Monte Carlo simulation
- Bayesian inference
- Probability theory
- Stochastic processes
- Quantitative finance
- Machine learning
- Statistical risk analysis

Planned modeling components include:

- Probability distributions
- Gaussian copulas
- Geometric Brownian Motion
- Jump-diffusion processes
- Hidden Markov Models
- Bayesian parameter calibration
- Sensitivity analysis
- Portfolio optimization
- Principal Component Analysis (PCA)

---

## Project Structure

```text
bioventure/
├── README.md
├── pyproject.toml
├── requirements.txt
├── LICENSE
├── CITATION.cff
├── .gitignore
│
├── configs/
│   ├── base.yaml                    # canonical baseline (2026–2035)
│   ├── scenarios/
│   │   ├── conservative.yaml
│   │   ├── aggressive_ai.yaml
│   │   └── bear_market.yaml
│   ├── priors.yaml                  # distributional priors per parameter
│   ├── correlations.yaml            # copula correlation matrix + copula type selector
│   ├── processes.yaml               # which stochastic process per model component
│   └── calibration.yaml             # calibration targets, method, MCMC settings
│
├── data/
│   ├── raw/
│   │   ├── clinical_transition_rates.csv
│   │   ├── market_baseline.csv
│   │   ├── ai_adoption_anchors.csv
│   │   └── historical_returns.csv   # NEW — venture fund returns for payoff calibration
│   ├── interim/
│   ├── processed/
│   └── external/
│
├── src/bioventure/
│   ├── __init__.py
│   ├── config.py                    # YAML → typed dataclasses, validation, hashing
│   ├── rng.py                       # seeded Generator factory, substream spawning
│   │
│   ├── distributions/
│   │   ├── __init__.py
│   │   ├── priors.py                # factory: priors.yaml → frozen scipy distributions
│   │   ├── copula_base.py           # CopulaBase protocol: sample(n) → correlated uniforms
│   │   ├── gaussian_copula.py       # Gaussian copula implementation
│   │   └── student_copula.py        # Student-t copula (tail dependence)
│   │
│   ├── processes/                   # ── NEW: pluggable stochastic process layer ──
│   │   ├── __init__.py
│   │   ├── base.py                  # StochasticProcess protocol: evolve(state, dt, rng) → path
│   │   ├── gbm.py                   # geometric Brownian motion
│   │   ├── jump_diffusion.py        # Merton jump-diffusion
│   │   ├── regime_switching.py      # Markov regime-switching drift+vol
│   │   └── registry.py              # name → class lookup, populated from processes.yaml
│   │
│   ├── models/                      # ── domain models (NO financial valuation here) ──
│   │   ├── __init__.py
│   │   ├── clinical.py              # pipeline phase transitions (Beta-distributed)
│   │   ├── rd_compression.py        # AI-driven timeline/cost compression
│   │   ├── adoption.py              # Bass diffusion of AI-enabled R&D
│   │   └── market.py                # market-size trajectory (delegates to a StochasticProcess)
│   │
│   ├── valuation/                   # ── NEW: financial valuation, separated from domain models ──
│   │   ├── __init__.py
│   │   ├── payoff.py                # venture exit-multiple distributions (power-law / lognormal)
│   │   ├── portfolio.py             # portfolio-level aggregation of bets
│   │   ├── discount.py              # risk-adjusted discounting (rNPV, optional WACC)
│   │   └── metrics.py               # IRR, MOIC, TVPI, DPI, probability of loss
│   │
│   ├── calibration/                 # ── NEW: Bayesian calibration layer ──
│   │   ├── __init__.py
│   │   ├── likelihoods.py           # likelihood functions per data source
│   │   ├── bayesian.py              # posterior estimation (MLE → MAP → optional MCMC via emcee)
│   │   └── prior_update.py          # merge posteriors back into priors.yaml format for simulation
│   │
│   ├── validation/                  # ── NEW: model validation ──
│   │   ├── __init__.py
│   │   ├── backtests.py             # hindcast: simulate historical period, compare to realized data
│   │   ├── stylized_facts.py        # statistical checks: tail shape, autocorrelation, moments
│   │   └── cross_validation.py      # k-fold or rolling-window validation of calibrated parameters
│   │
│   ├── simulation/                  # ── REFACTORED: four distinct responsibilities ──
│   │   ├── __init__.py
│   │   ├── sampler.py               # draw correlated parameter vectors via copula
│   │   ├── engine.py                # orchestrate: sampler → models → valuation → recorder
│   │   ├── recorder.py              # collect, buffer, and flush raw per-iteration results
│   │   └── aggregator.py            # reduce raw results → quantiles, tail stats, distributions
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── sensitivity.py           # Sobol / Morris global sensitivity (SALib)
│   │   └── convergence.py           # MC diagnostics: running mean, std-error vs N
│   │
│   ├── viz/
│   │   ├── __init__.py
│   │   ├── distributions.py         # fan charts, histograms, KDEs
│   │   ├── tornado.py               # sensitivity tornado + Sobol bar charts
│   │   ├── calibration_plots.py     # NEW: prior/posterior comparison, trace plots
│   │   └── style.py                 # publication matplotlib theme
│   │
│   └── io/
│       ├── __init__.py
│       ├── loaders.py               # schema-checked CSV/YAML readers
│       └── writers.py               # Parquet, figures, run manifests
│
├── scripts/
│   ├── run_calibration.py           # NEW: fit posteriors from historical data
│   ├── run_simulation.py            # main MC run
│   ├── run_sensitivity.py           # Sobol/Morris analysis
│   ├── run_validation.py            # NEW: backtests + stylized-fact checks
│   └── build_report.py             # regenerate all figures/tables
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_calibration_review.ipynb  # NEW: inspect posteriors before simulation
│   ├── 03_model_sanity_checks.ipynb
│   └── 04_results_narrative.ipynb
│
├── results/
│   ├── calibration/                 # NEW: posterior samples + diagnostics
│   ├── runs/<run_id>/
│   │   ├── manifest.json
│   │   ├── raw_paths.parquet
│   │   ├── summary.parquet
│   │   └── sensitivity.parquet
│   ├── validation/                  # NEW: backtest reports
│   ├── figures/
│   └── tables/
│
├── paper/
│   ├── manuscript.md
│   └── figures/
│
└── tests/
    ├── conftest.py
    ├── test_config.py
    ├── test_distributions.py
    ├── test_copulas.py              # both Gaussian + Student-t: correlation recovery, tail dep.
    ├── test_processes.py            # GBM, jump-diff, regime-switch: known moments, path properties
    ├── test_models.py
    ├── test_valuation.py            # NEW: payoff, portfolio, discount, metrics
    ├── test_calibration.py          # NEW: posterior recovery on synthetic data
    ├── test_sampler.py              # NEW: marginal + joint properties of sampled vectors
    ├── test_engine.py
    └── test_validation.py           # NEW
```

---

## Current Development

Phase 1
- Configuration management
- YAML-based parameter system
- Deterministic random number generation
- Unit testing framework

Phase 2
- Probability distributions
- Gaussian copulas
- Prior sampling engine
- Statistical validation tests

Upcoming phases include stochastic market processes, Monte Carlo simulation, Bayesian calibration, quantitative valuation models, portfolio optimization, machine learning, and publication-quality visualization.

---

## Data Sources

The framework is designed to integrate publicly available biotechnology datasets, including:

- ClinicalTrials.gov
- Open Targets Platform
- FDA AI in Drug Development resources

Simulation assumptions are stored as configurable YAML files to support reproducible experimentation.

---

## Technologies

- Python
- NumPy
- SciPy
- Pandas
- Matplotlib
- Plotly
- PyYAML
- pytest
- Git

---

## Running Tests

```bash
py -m pytest tests/ -v
```

---

## Repository Goals

- Develop a reproducible quantitative research framework
- Model AI-enabled drug discovery market dynamics
- Apply quantitative finance methods to biotechnology
- Produce publication-quality analyses and visualizations
- Maintain a modular, test-driven Python codebase

---

## Disclaimer

BioVentureSim is an independent research project intended for educational and scientific purposes. All simulations are based on probabilistic models and configurable assumptions and should not be interpreted as financial, investment, regulatory, or medical advice.

---

## Author

Sanvi Tummala
Independent Computational Biology & Quantitative Research
