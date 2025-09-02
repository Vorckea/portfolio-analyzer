"""Core types for objective protocol used by the optimizer.

This module defines the `ObjectiveProtocol` used to decouple the optimizer
from concrete objective implementations and `ObjectiveError` for error
reporting during adaptation.
"""

from __future__ import annotations

from typing import Callable, Optional, Protocol

import numpy as np


class ObjectiveError(Exception):
    """Raised for invalid inputs or failures in objective evaluation."""


class ObjectiveProtocol(Protocol):
    """Protocol describing a pluggable objective.

    Implementations must expose a method that returns a weights-only callable
    (Callable[[np.ndarray], float]). Optionally implementations can expose
    a `name` attribute and a `gradient` callable.
    """

    name: str

    def to_callable(self) -> Callable[[np.ndarray], float]:
        """Return a weights-only callable that accepts a numpy array of weights."""

    def gradient(self) -> Optional[Callable[[np.ndarray], np.ndarray]]:
        """Optionally return a gradient callable or None if not available."""
