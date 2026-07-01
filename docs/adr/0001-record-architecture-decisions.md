# 1. Record architecture decisions

- **Status:** Accepted
- **Date:** 2026-07-01

## Context

Dong Feng makes several load-bearing architecture choices early (the flagship
model paradigm, the protocol layer, the rules backend). These choices ripple
through the whole codebase and are expensive to reverse. Future contributors —
human and AI agents alike — need to know not just *what* was decided but *why*,
so they neither re-litigate settled questions nor accidentally violate a decision
whose rationale is invisible in the code.

## Decision

We record architecturally significant decisions as **Architecture Decision
Records (ADRs)** in `docs/adr/`, numbered sequentially (`NNNN-title.md`). Each ADR
is short and follows the same shape: **Context**, **Decision**, **Consequences**,
and a **Status** (Proposed / Accepted / Deprecated / Superseded).

An ADR is warranted when a choice affects module boundaries, public contracts,
the model paradigm, external protocols, or major dependencies. Small, local, or
easily-reversible choices do not need one.

ADRs are immutable once accepted: to change a decision, add a new ADR that
supersedes the old one (and update the old one's Status to *Superseded by NNNN*).

## Consequences

- New contributors and agents can read `docs/adr/` to understand the *why* behind
  the structure without archaeology through git history.
- There is a small ongoing cost to write an ADR for significant changes; this is
  intentional friction that forces the decision to be articulated.
- The `CLAUDE.md` "to find X, read Y" table points here for the question "why was
  this done this way?".
