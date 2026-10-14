"""Shared fit-accounting protocol at the authoritative dispatch fit boundary.

The bounded synthetic path uses a persistent ``FitCounter``. The full sequential
campaign uses a journal-backed adapter whose debit is durable before estimator
construction but which must NOT masquerade as a bounded ``FitBudget``. Both expose
the same minimal structural surface, so ``fit_arm`` routes them identically.

Runtime-checkable Protocol membership only requires the listed members; it is a
routing contract, not a capacity guarantee. A fabricated bounded ``FitBudget``
must never be used to satisfy it for the opted-out full run.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class FitAccounting(Protocol):
    """Minimal per-fit debit surface consumed by the dispatch routing."""

    def before_fit(self) -> None:
        """Durably charge one model fit before construction, or raise."""

    @property
    def reserved_model_fits(self) -> int:
        """Cumulative model fits reserved for this campaign/owner."""

    @property
    def used_model_fits(self) -> int:
        """Cumulative model fits already charged."""
