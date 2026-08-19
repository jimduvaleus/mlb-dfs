"""Marginal-Reward Portfolio (MRP) — Haugh & Singal 2019 formulation (2).

A parallel portfolio-construction track that prices SELF-COMPETITION: our own
entries sit inside the order statistic, so a second near-identical entry is
penalised mechanically (it cannot also take first place) rather than by a
correlation heuristic.

Production is untouched. See plans/ for the build plan and EVIDENCE_LOG.md for
the pre-registration governing any archive-facing comparison.
"""
