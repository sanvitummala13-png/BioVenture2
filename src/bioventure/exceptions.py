"""Custom exception hierarchy for BioVenture.

All library-level errors subclass ``BioVentureError`` so callers can
catch the entire domain with a single ``except`` clause.
"""

from __future__ import annotations


class BioVentureError(Exception):
    """Base exception for all BioVenture library errors."""


class ConfigError(BioVentureError):
    """Invalid or missing configuration value."""


class SimulationError(BioVentureError):
    """Error raised during Monte Carlo simulation."""


class CalibrationError(BioVentureError):
    """Error raised during Bayesian calibration."""


class ValidationError(BioVentureError):
    """Input failed a domain validation check."""


class NumericalError(BioVentureError):
    """Non-finite or unstable numerical output detected."""
    