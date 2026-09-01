# Model-use disclosure

Claude had two distinct roles in this project.

## Measured proposal policy

The experiment invoked Claude Sonnet through Claude Code CLI 2.1.241. Sixty
matched quartets produced 240 arm-level research episodes. Ten episodes were
batched per request across five rounds and four independent call groups, for
120 archived model calls and a recorded service cost of USD 17.54.

The four arms received exact target feedback, independent sham feedback, no
numeric feedback, or a coarsened running best. They never received returns,
asset labels, dates, confirmation outcomes, or another arm's history. Opaque
identifiers and randomized call ordering hid arm, panel, and era. Schema errors,
duplicates, and fallbacks remain in the intent-to-treat estimand.

Every prompt and response wrapper is stored under
`artifacts/agentic/transcripts/`. Model aliases, canonical identifiers, token
use, latency, costs, file hashes, and fallback rates are stored with the frozen
results.

## Coauthoring and review tool

Claude was also used to criticize experimental design, code, statistical
language, and release readiness. Model suggestions were treated as review
comments, not authority. Quantitative statements were checked against the
producing code and archived outputs. Recent literature citations were checked
against primary proceedings or arXiv records. A withdrawn preprint identified
during the release audit was removed.

The human author chose the question and protocol, resolved conflicts, decided
what to report, and accepts responsibility for the manuscript and repository.
No prose detector can establish authorship or research integrity; the relevant
evidence is the disclosed workflow, frozen protocol, executable tests, complete
traces, and claim-to-artifact audit trail.
