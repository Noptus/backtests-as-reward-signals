# Generator-to-archive provenance note

This note is a reviewer aid, not a new experiment. It answers one narrow
question: can the released generator and search code recreate representative
rows in the frozen synthetic archives from their recorded seeds?

## What is rerun

The audit fixes three markets before execution: the first, midpoint, and last
of the consecutive 2,000-market range, with seeds 50,000, 50,999, and 51,999.
For each market it recreates the target path, independent sham path, and sealed
confirmation path. It then reruns the blind, sham-feedback, and target-feedback
policies through budget 250.

That produces 750 candidate rows per market, or 2,250 rows in total. The audit
also reconstructs results at the six frozen prefixes 1, 10, 25, 50, 100, and
250 for all three policies and all three markets, giving 54 result rows.

## What is compared

Seeds, policies, trial indices, strategy fields, canonical strategy IDs,
selected winners, and every promotion flag must match exactly. Sharpe ratios,
probabilities, effective-trial values, and optimism gaps are compared with
absolute and relative tolerances of `1e-12`. The audit separately checks that
each smaller budget is a literal prefix and that the archived winner is the
first validation-Sharpe maximum in that prefix.

The current run matches every discrete field. The largest floating-point
difference is `2.220446049250313e-16`, roughly four thousand times smaller than
the fixed tolerance. The machine-readable report records the input hashes,
field lists, row counts, tolerances, and maximum differences.

## What this establishes

For these three markets, the released code-to-archive path is deterministic to
the stated tolerance. This is useful because the ordinary artifact check starts
from archived trials: it would not, by itself, detect a disagreement between
the generator/search pipeline and those archives.

## What it does not establish

Three markets are not a rerun of all 2,000 markets. The audit is not an
independent implementation, does not validate the statistical method, and
cannot detect a defect shared by the generator and its archived output. It does
not replay hosted-model calls or touch the sealed policy boundary. The separate
23-component verifier addresses aggregation from frozen CSVs; these two checks
cover different links in the evidence chain and neither is an independent
replication.

Run the audit with:

```bash
PYTHONPATH=src uv run python -m alpha_firewall.generator_audit
```

Its canonical result is
`artifacts/reproduction/synthetic_generator_spotcheck.json`. The same command
runs in public CI, while the full isolated artifact command also compares the
fresh report byte-for-value against the manifest-hashed canonical report.
