# Identifiability of the correction term: Worst-Case Leakage Is Governed by the Principal Angles of the Dictionary
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23075791.svg)](https://doi.org/10.5281/zenodo.23075791)
Repository for the manuscript `manuscript/main.tex` (49 pages, PDF included). Prepared for
submission to JAIGP (AI Generated Papers).

**Authors.** Xiangrui Meng (School of Mathematics and Physics, Shanghai Normal University; ORCID
0009-0006-8902-0519) — lead (human) author, prompter and editor. DeepSeek V4.1 Flash (Agent A) —
experiments, certificate construction, falsification of the manuscript's own claims. Qwen 3.8 Flash
(Agent B) — derivation, theorems, numerics, literature verification.

**The question.** A scientific-ML model is written as a prior operator plus a learned correction,
`A* = sum_i theta_i* E_i + sum_j psi_j* F_j + Delta_hidden`. Least-squares fitting of the correction
block from `n` input-output pairs answers a question about *attribution*, not accuracy: how much of an
unmodelled operator `Delta_hidden` is forced into the fitted block? The answer is exact and geometric —
it is governed by the principal angles of the dictionary, with worst-case leakage `1/sin(alpha_min)`.

## Layout

| Path | Contents |
|---|---|
| `manuscript/` | `main.tex` (pdflatex, standard `article` class, no external figures), `main.pdf`, and `cover.png` (1200x800, the two closed-form factors of the certificate) |
| `scripts/agent_b/` | 69 numbered measurement/derivation scripts from Agent B (`b1`–`b65`, plus `enso_lib.py`, `lit25_check.py`) |
| `scripts/agent_a/` | 18 experiment scripts from Agent A (`d1`–`d16`) |
| `evidence/agent_b/`, `evidence/agent_a/` | the saved stdout of those runs (`.txt`, `.log`), named after the script that produced them |
| `logs/` | `community.md` — the full A/B collaboration log (claims registered, challenged, retracted) — and `literature-notes/` (163 internal reading notes, Chinese) |
| `tools/` | literature-fetching and bibliographic verification tools, plus the LaTeX surgery script used on the manuscript |

## Reproducing a claim

Each script prints the table that appears in the corresponding section of the paper; the saved output is
next to it under `evidence/`. From a script's own directory:

```
python b27_adversarial_margin.py     # (star) certificate, 4320 adversarial trials, n = 30..1800, alpha = 2..85 deg
python b25_pqn_law.py                # sample-size law for the remainder, 21 cells
python b38_exact_scalar_formula.py   # closed form for lambda_min(G): the sphere minimisation collapses
```

`scripts/agent_b/enso_lib.py` reads the ENSO index tables from `scripts/agent_b/data/` (resolved relative
to the file, so the scripts run from any working directory). Only the two Agent A scripts listed in
`requirements.txt` need `torch`; neither supports a claim in the paper.

Build the paper with `pdflatex main.tex` twice (no bibtex; the reference list is inline and every entry
was verified against its DOI or the publisher's own metadata).

## Data provenance

`scripts/agent_b/data/` holds public climate index tables (ONI, MEI v2, Nino-4, ERSST.v5 Nino regions,
SOI, CENSO) as downloaded from the NOAA/CPC and PSL index pages; the URLs recorded during the work are in
`logs/community.md`. The real-data leg they were used for was retired by its own measurements and is
reported as such in the manuscript, not as a successful application.

## What this repository deliberately does not hide

`logs/community.md` and the manuscript's "Limitations, retracted claims, and open items" section record
the claims that were falsified by our own runs, including ones the paper had already asserted. Where a
cited paper was checked only at abstract level, the manuscript says so. One third-party email address
that appeared inside an arXiv source's leftover review comments has been redacted from a literature note.

## License

MIT — see `LICENSE`. Copyright (c) 2026 Xiangrui Meng.
