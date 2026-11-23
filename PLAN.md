# NeutralAtomBench MVP

## Goal

Make one reproducible comparison inspired by
`ghz_sequential_vs_feedforward.ipynb`:

1. prepare an N-qubit GHZ state with sequential entangling gates;
2. prepare the same state from a linear cluster, X-basis measurements, and
   feed-forward corrections;
3. compare GHZ fidelity as N grows under the same simple noise model.

The deliverable is one script, one data table, and one plot. It should be easy
to read, edit, and run from the repository root.

## Decisions made for this MVP

| Decision | Choice |
| --- | --- |
| Code shape | One module: `ghz_mvp.py` |
| Simulator | Qiskit Aer, already installed |
| Protocols | Sequential GHZ and cluster + feed-forward GHZ |
| Noise | CZ phase errors and an explicit idle/dephasing proxy |
| Main metric | Fidelity with the ideal GHZ state |
| Sweep | Qubit count (default 2 through 6) |
| Deferred | Stim, a generic protocol framework, hardware presets, loss, fusion, routing, CLI, and tests beyond direct correctness checks |

The two protocols use the same noise parameters. The comparison is exploratory:
the first plot answers how these two concrete constructions behave under the
chosen assumptions; it does not make a broad hardware claim.

## Tickets

- [x] MVP-1 — Create a working Python environment with Qiskit Aer.
- [x] MVP-2 — Add `ghz_mvp.py` with both protocol builders and an ideal-state correctness check.
- [x] MVP-3 — Apply one shared CZ and idle noise model, then return fidelity for either protocol.
- [x] MVP-4 — Sweep GHZ size and save a labelled comparison plot.
- [ ] MVP-5 — Review the first plot together and choose the next experiment: alter noise assumptions, improve scheduling, or add a task.

## Working agreement

We only add a new module when the current one has become difficult to use. Each
ticket should leave a runnable result before we take on the next one.
