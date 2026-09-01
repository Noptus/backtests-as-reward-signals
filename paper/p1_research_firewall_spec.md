# Research-firewall implementation specification

Companion to the frozen P1 release. Prepared 1 September 2026. This document
changes no manuscript result, protocol, seed, table, figure, or 61-test release
claim. It translates the experiment into a reviewer-auditable control design;
it is not evidence that a production firewall has been deployed.

## Evidence boundary

P1 directly tests an information-flow claim: iterative access to a target
validation score increases selected validation performance under exact nulls,
whereas one strategy evaluated once on a temporally disjoint confirmation
sample remains near the nominal promotion rate in the tested grammar. P1 also
executes a frozen budget, target-versus-sham feedback intervention, selection
after search, and grammar-specific canonicalization.

The following are **engineering recommendations, not independently validated
products**: process-level capability isolation; an immutable hash chain for
every query; a service that physically prevents a second confirmation query;
automatic secret isolation; incident alerting; and recovery across machines or
providers. Coarsening and deduplication reduce feedback bandwidth only within a
specified grammar; neither is a universal validity guarantee.

## Trust boundaries and state machine

| Boundary | Permitted information flow | Prohibited flow | Evidence status |
|---|---|---|---|
| Development | Raw development data, code, prompts, and unrestricted iteration | Protected validation or confirmation data | recommended isolation |
| Protected validation | A frozen score or predeclared coarsened release for a canonical strategy fingerprint | Raw returns, hidden labels, another arm's trace, or an unlogged score | P1 tests exact/sham/blind/coarsened feedback effects; service enforcement is recommended |
| Confirmation | One frozen strategy fingerprint enters; one promotion decision leaves | Scores or returns returned to the proposer; any adaptive second query | P1's protocol evaluates confirmation once after selection; a production write-once service is recommended, not tested |

Confirmation state is monotone:

`SEALED -> QUERY_RECORDED -> PROMOTED | REJECTED`.

Any unauthorized read, missing ledger segment, strategy-fingerprint mismatch,
or attempted second query instead gives `BURNED`. A burned sample cannot be used
to retune a candidate, threshold, grammar, or budget.

## Auditable pseudocode

```text
freeze(protocol_hash, grammar, budget, score_release, selection_rule,
       promotion_gate, failure_policy)
state = SEALED

for proposal in proposer:
    spec = canonicalize(proposal)
    if fingerprint(spec) in executed: continue
    released = validation_service(spec, score_release)
    append_ledger(prompt_hash, fingerprint(spec), released, retries, fallbacks)
    if len(executed) == budget: break

selected = argmax(executed, key=frozen_validation_sharpe)
assert state == SEALED and confirmation_query_count == 0
append_ledger("confirmation_open", fingerprint(selected), protocol_hash)
state = QUERY_RECORDED
decision = confirmation_service.evaluate_once(selected, promotion_gate)
state = PROMOTED if decision else REJECTED

on unauthorized_access | missing_log | fingerprint_mismatch | second_query:
    state = BURNED
    abort_promotion(); quarantine_candidate(); preserve_incident_record()
```

The ledger append precedes the protected evaluation. Recovery uses a new sealed
sample and a newly frozen protocol; it never treats a failed confirmation result
as development feedback.

## Invariant-to-artifact map

| Invariant | Current evidence | Boundary of the evidence |
|---|---|---|
| Confirmation is evaluated after validation selection | `src/alpha_firewall/experiment.py:evaluate_trials` | code follows the order but does not itself enforce a cross-process write-once capability |
| One confirmation evaluation per recorded result row | `evaluate_trials` contains one confirmation backtest; frozen result CSVs and 30 August reproduction report | does not prove a malicious caller cannot invoke the function twice |
| Budget and gates are frozen | `configs/deep_paper.toml`, `paper/deep_revision_analysis_plan.md` | applies to the released experiments, not every future deployment |
| Behaviorally duplicate specs do not consume budget | `StrategySpec` canonicalization, `run_search` seen-set, `tests/test_search.py` | only equivalences encoded by the current strategy grammar |
| Target and sham begin from matched candidates | `tests/test_search.py:test_target_and_sham_arms_share_initial_candidates` | later proposal paths may diverge because their feedback differs, which is the intervention |
| Frozen inputs fail on hash mismatch | `verify_manifest` and `tests/test_repro_check.py` | a manifest is not a live per-query hash chain |
| Frozen-winner resampling can be invalid | prospective P4 exact counterexample | separate P4 method evidence; not part of the frozen 61-test P1 release |

## Reviewer checks

1. Can the proposer address the confirmation capability before selection?
2. Is every released score linked to a canonical executable fingerprint and the
   protocol hash?
3. Can a retry, fallback, repair, or duplicate escape the frozen budget?
4. Does any confirmation-derived value return to proposal generation?
5. Does every failure burn the sample without retuning on its result?

A “yes” answer to any of checks 1--4 or a “no” answer to check 5's fail-closed
behavior blocks a production validity claim. It does not retroactively alter P1's bounded
experimental result; it shows that a proposed deployment does not implement the
control boundary the paper recommends.
