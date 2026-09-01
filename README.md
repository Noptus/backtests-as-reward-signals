# Backtests as Reward Signals

[![CI](https://github.com/Noptus/backtests-as-reward-signals/actions/workflows/ci.yml/badge.svg)](https://github.com/Noptus/backtests-as-reward-signals/actions/workflows/ci.yml)
[![Paper](https://img.shields.io/badge/paper-PDF-b31b1b.svg)](output/pdf/backtests_as_reward_signals.pdf)
[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)

This repository accompanies *Backtests as Reward Signals*. It is a controlled
evaluation harness—not a trading signal—for one failure mode in autonomous
research: when a proposer can read a backtest score and condition its next idea
on that score, the backtest has become a reward signal and the tested family is
data-adaptive.

The experiment measures that channel under exact nulls. Target-feedback and
sham-feedback researchers start from the same candidates, use the same grammar
and budget, and differ only in whether later proposals see the scored target
path. The selected rule is then evaluated once on a temporally disjoint sample
that the proposer never saw.

**Read the [eight-page paper](output/pdf/backtests_as_reward_signals.pdf) first.**

## What happened

At the maximum budget, target feedback raised selected validation Sharpe without
raising sealed-sample Sharpe:

| Experiment | Primary units | Validation: target minus sham | Sealed confirmation: target minus sham |
|---|---:|---:|---:|
| Synthetic exact null | 2,000 markets | +0.255 [0.246, 0.265] | +0.000 [-0.020, 0.021] |
| Sign-randomized French panels | 15 dataset–era cells | +0.222 [0.192, 0.253] | -0.010 [-0.072, 0.052] |
| Recorded Claude proposer | 15 dataset–era cells | +0.181 [0.093, 0.268] | +0.027 [-0.078, 0.131] |

Intervals are 95% intervals at the pre-specified aggregation level. The
synthetic target-feedback arm promoted 97.2% of null markets under a naive
one-trial rule. Promotion was 42.0% under the participation-ratio DSR heuristic,
9.7% under Bonferroni, 35.2% under BH, and 5.3% on the single-use sealed sample.
Raw-count DSR was conservative in this benchmark. The paper reports a planted-
signal power study rather than treating low false-positive rates as sufficient.

The Claude experiment is deliberately not a model leaderboard. It freezes one
recorded Sonnet snapshot and has wider uncertainty than the 2,000-market
simulation. Its purpose is to show that the same feedback intervention can be
run on an actual language-model proposer. The synthetic and conditional-null
experiments carry the main statistical claim.

## Reproduce the frozen evidence

With [uv](https://docs.astral.sh/uv/) and
[Tectonic](https://tectonic-typesetting.github.io/) installed:

```bash
uv run --isolated --frozen --extra dev python -m alpha_firewall.repro_check
```

This command makes no network requests and no hosted-model calls. It checks all
174 frozen inputs against SHA-256 hashes, runs Ruff and 61 tests, regenerates
tables and figures in a new staging directory, recomputes 23 claim-critical
components through a separately coded verifier, and compiles the eight-page
paper. It fails closed on a missing input, hash mismatch, test failure, numeric
disagreement, or malformed paper.

This is artifact reproduction, not an independent replication and not a full
rerun of 2,293,500 candidate evaluations. See
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the distinction and the full-run
commands. The CI badge covers lint and the 61-test suite; the complete artifact
gate also needs Tectonic and is run with the command above.

## Where to look

1. [The paper](output/pdf/backtests_as_reward_signals.pdf) states the estimands,
   evidence, and limits.
2. [configs/deep_paper.toml](configs/deep_paper.toml) freezes seeds, budgets,
   panels, eras, and agent settings.
3. [experiment.py](src/alpha_firewall/experiment.py) and
   [search.py](src/alpha_firewall/search.py) implement the matched feedback arms
   and post-selection confirmation.
4. [test_search.py](tests/test_search.py) and
   [test_backtest.py](tests/test_backtest.py) test matched initialization,
   canonicalization, and timing integrity.
5. [deep_summary.json](artifacts/paper/deep_summary.json) contains the numbers
   behind the manuscript macros and tables.
6. [independent_headlines.json](artifacts/reproduction/independent_headlines.json)
   records the same-data verifier and its tolerance checks.
7. [The research-firewall specification](paper/p1_research_firewall_spec.md)
   translates the evidence into an operational control design while labeling
   what the paper did and did not validate.

The 120 agent transcripts, candidate-level audit files, model identifiers,
costs, failures, fallbacks, and result hashes are included under `artifacts/`.
The frozen metadata retains historical absolute paths as provenance; the live
CLI is portable and resolves `claude` from `PATH` by default.

## Research integrity

- The confirmation sample is not available to any proposal policy and is opened
  once, after selection.
- Synthetic confirmation starts at fresh seeds; the earlier 200-market run was
  a design pilot and is not pooled with the 2,000-market evidence.
- Five Kenneth French daily portfolio families are converted to finite-sample
  exact conditional nulls by independent sign randomization. The source
  archives and their hashes are recorded but not redistributed.
- Smaller candidate budgets are prefixes of one maximum-budget path.
- Invalid or duplicate Claude proposals trigger logged fallbacks. They are not
  silently removed from the estimand.
- Raw-count DSR and a participation-ratio sensitivity are separate results.
- The strategy language is intentionally narrow: lagged cross-sectional
  momentum/reversal rules with a fixed execution model. Generalization to a
  richer research language remains untested.

Claude was both a measured proposal policy and a critical coauthoring tool. The
human author fixed the protocol, checked the code and evidence, and accepts
responsibility for the paper. [MODEL_DISCLOSURE.md](MODEL_DISCLOSURE.md) gives
the complete boundary.

## Status and licenses

This is a working paper and has not yet completed peer review. Code is released
under the [MIT License](LICENSE). The manuscript and original documentation are
released under [CC BY 4.0](LICENSE-PAPER.md). External datasets and third-party
materials remain subject to their own terms.
