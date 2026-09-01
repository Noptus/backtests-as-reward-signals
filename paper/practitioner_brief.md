# Hedge-fund research brief

## Thirty-second version

I built a controlled evaluation environment for autonomous quant researchers.
It measures how much validation alpha is manufactured when a proposal engine
receives backtest scores as rewards, using exact null markets where the correct
answer is known. The confirmatory evidence covers 2,000 synthetic markets, five
canonical Kenneth French portfolio families, a recorded Claude research agent,
and more than two million candidate backtests. The practical output is a
research-firewall protocol: separate development, limited validation feedback,
single-use confirmation, and immutable search traces.

## What the work demonstrates

- Experimental design: matched target-versus-sham feedback isolates the causal
  information channel from ordinary multiple testing and search coverage.
- Statistical judgment: raw-count DSR and an effective-trial heuristic are kept
  separate; size is never confused with power; historical panels are converted
  into exact conditional nulls instead of being presented as profitable alpha.
- Research engineering: one frozen strategy language, no-look-ahead tests,
  semantic deduplication, nested candidate budgets, parallel simulation, hashed
  datasets and outputs, and complete agent transcripts.
- Agent evaluation: exact, sham, blind, and coarsened feedback are randomized
  across matched Claude episodes, with schema failures and fallbacks included in
  the estimand rather than silently discarded.

## What the work does not claim

It does not claim a deployable strategy, universal failure of DSR or
Bonferroni, or stable superiority of one hosted model. The paper is about the
validity of a research process. That restraint is part of the contribution.

## Ten-minute interview demo

1. Show the frozen TOML protocol and the no-look-ahead/semantic-duplicate tests.
2. Open one matched target/sham candidate trace and show where proposals diverge
   after identical initialization.
3. Reproduce the 2,000-market validation-versus-confirmation contrast from the
   result CSV.
4. Show the same effect in the five French sign-randomized panels.
5. Open one blinded Claude transcript and the model/cost/fallback manifest.
6. Finish with the promotion-size and planted-signal power curves, emphasizing
   the governance trade-off.

## Publication sequence

1. Release the working paper, code, audit logs, and immutable hashes.
2. After an independent statistics review and a live preprint-policy check,
   submit the practitioner-facing journal version to the *Journal of Financial
   Data Science*; use *Quantitative Finance* as the formal-method backup.
3. Develop the powered cross-model benchmark for ACM ICAIF or the NeurIPS
   Evaluations & Datasets track when the next official calls open.
4. Extend the formal method and size--power analysis for AISTATS/ICML or a
   quantitative-finance journal only if the pre-stated go/no-go test is met.
5. Develop the trace-complexity paper as the most fund-specific follow-on: can
   hidden agentic researcher degrees of freedom predict sealed-sample decay?
