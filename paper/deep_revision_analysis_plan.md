# Prospective deep-revision analysis plan

Date fixed: 2026-08-25, before running the deep-revision synthetic,
empirical-null, or Claude-agent experiments.

Status: this is a prospective internal analysis plan, not an externally
registered preregistration. It was written after inspecting the earlier
200-market development experiment. The new confirmatory runs use fresh seeds
starting at 50,000 and corrected code. The earlier experiment is treated as a
design pilot and will not be pooled into confirmatory estimates.

Amendment log: the four-arm Claude design and planted-signal Experiment D were
added while Experiment A was still running, before its output existed and before
Experiments B--D began. The coarsened-channel wording was made implementation-
exact before the confirmatory Claude run. A one-quartet smoke test and one
completed plus one interrupted call from an incomplete timing run were discarded
without confirmatory strategy evaluation; the confirmatory run restarted all 120
calls from call zero after independent arm calls were parallelized. This changed
wall-clock execution only, not prompts, treatments, candidate budgets, or
estimands.

## Corrections made before the new runs

1. Equal-weight strategies now canonicalize the irrelevant volatility window,
   eliminating behaviorally duplicate specifications.
2. Canonical DSR uses the raw number of evaluated trials. The earlier
   participation-ratio substitution is retained only as a separately labeled
   heuristic sensitivity. No result will be described as a general failure of
   DSR.
3. The no-lookahead test perturbs an asset-specific future cross-section and
   requires both exact pre-cutoff invariance and post-cutoff sensitivity.
4. New result manifests record data and output hashes.

## Experiment A: parametric global null

Run 2,000 independent Student-t stochastic-volatility markets, each with 30
assets, 1,260 scored validation sessions, a disjoint 252-session gap, and 1,260
scored confirmation sessions. Search budgets are nested prefixes of 1, 10, 25,
50, 100, and 250. Policies are blind batch, adaptive sham feedback, and adaptive
target feedback. Each market-policy trajectory is evaluated once at maximum
budget and smaller budgets reuse its prefix.

Primary estimands at budget 250:

- paired target-minus-sham selected validation Sharpe;
- paired target-minus-sham confirmation Sharpe;
- paired target-minus-sham optimism gap;
- false-promotion probability for raw-budget Bonferroni and sealed confirmation.

Inference uses the independent market as the unit. Continuous contrasts receive
paired t intervals and paired sign-randomization p-values. Promotion rates
receive Wilson intervals. A confirmation difference whose 90% interval lies
inside [-0.10, 0.10] Sharpe is treated as evidence of practical equivalence,
not as proof of exact equality.

Canonical raw-count DSR, participation-ratio DSR, BH, naive HAC, Bonferroni,
sealed confirmation, and dual gates are all reported. Only the raw-count version
is labeled DSR without qualification.

## Experiment B: empirical exact nulls on canonical data

Download five daily value-weighted panels from the Kenneth French Data Library:

- 49 industry portfolios;
- 25 size/book-to-market portfolios;
- 100 size/book-to-market portfolios;
- 25 size/momentum portfolios;
- 25 size/short-term-reversal portfolios.

Raw files are downloaded during reproduction, not redistributed, and their
SHA-256 values are recorded. Three nonoverlapping calendar blocks begin in
1990, 2002, and 2014. Each contains one warm-up year, five validation years, a
one-year sealed gap/warm-up, and five confirmation years.

For every panel-block cell, generate 20 exact conditional-null paths by drawing
an independent Rademacher sign for each date and multiplying the entire
cross-section by that sign. This preserves each realized cross-sectional outer
product, absolute return, missingness pattern, jump magnitude, and calendar
while enforcing zero conditional expected return for lagged strategies. Sham
feedback uses an independently signed version of the same raw panel.

The benchmark contains 300 panel-block-null environments and 225,000 candidate
backtests at maximum budget. Primary inference averages null paths within each
of the 15 panel-block cells, then reports the distribution and t interval across
cells. Path-level intervals are secondary Monte Carlo summaries and are not
treated as 300 independent historical markets.

## Experiment C: Claude research agent

Claude Sonnet, invoked through Claude Code CLI 2.1.241, replaces the hand-coded
proposal operator while the evaluator and frozen strategy grammar remain
unchanged. The exact canonical model identifier returned by the service is
recorded with every call.

Each of the 15 empirical panel-block cells receives four independently signed
matched quartets. Every quartet has exact-target, independent-sham, blind, and
coarsened-target arms with five identical random initial candidates. Five
subsequent rounds request four new strategies, producing a budget of 25
candidates per arm. The coarsened arm releases only a running-best Sharpe rounded
down to 0.10 and updates it only when a candidate crosses the next 0.10 bin; the
blind arm receives prior specifications without numeric scores. Arms from the
same quartet are sent in separate calls so the model cannot match their
identical initial strategies and compare feedback. Episode identifiers, order,
and call grouping reveal neither arm nor dataset.

Every prompt, response wrapper, structured proposal, model identifier, latency,
token/cost report, parse failure, duplicate, and random fallback is archived.
The model never receives asset names, dates, raw returns, confirmation results,
or the target/sham label. It receives only the fixed grammar and its episode's
prior specifications with scalar feedback Sharpe.

Primary agent estimand: paired Claude exact-target-minus-sham validation Sharpe
at budget 25. The coarsened-minus-exact contrast is the primary mitigation
estimand. Secondary outcomes are target-minus-blind, confirmation differences,
optimism-gap differences, fallback rate, proposal diversity, and the difference
between the Claude feedback effect and the deterministic adaptive-search effect
on matched empirical-null cells. Inference aggregates the four quartets within
each of 15 panel-block cells before testing; quartet-level results are also shown
transparently.

Because the hosted model is proprietary and not seed-reproducible, claims apply
only to the recorded model snapshot and protocol. No claim about Claude or LLMs
in general will be made.

## Reporting commitments

- Report null or adverse outcomes, including a zero Claude feedback effect.
- Report canonical raw-count DSR even if it reverses the development result.
- Report every panel separately before any pooled summary.
- Do not interpret chronological or sign-randomized panels as deployable alpha.
- Label all post-run additions exploratory and record them in a deviations log.

## Experiment D: planted-signal power calibration

Run a smaller synthetic alternative experiment with the same 30-asset risk
process and a cross-sectionally standardized 21-session momentum signal planted
in the next return. Daily signal amplitudes are 0.5, 1.0, and 2.0 basis points;
zero amplitude is verified to reproduce the global-null generator exactly.
For each nonzero amplitude, run 250 fresh markets with blind, sham-adaptive, and
target-adaptive search at the same nested budgets. Report power curves for the
sealed holdout, raw-count DSR, Bonferroni, and dual gates. This experiment is
secondary: it characterizes the size--power cost of the firewall and does not
change the primary null hypotheses.
