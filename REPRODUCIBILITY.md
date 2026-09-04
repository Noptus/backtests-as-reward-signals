# Reproducibility guide

The repository supports two different operations. They should not be described
as the same thing.

## 1. Verify the frozen artifact

```bash
uv run --isolated --frozen --extra dev python -m alpha_firewall.repro_check
```

The verifier first hashes every item in
`artifacts/results/deep_artifact_manifest.json`. It then runs static checks and
64 tests, regenerates three complete synthetic searches from fixed seeds and
compares 2,250 candidate rows plus 54 result rows with the archive, regenerates
all numeric and text outputs from the archived trials, executes a separately
coded headline verifier, and compiles the paper in a new staging directory.
Hosted-model generations are not replayed. No data are downloaded.

A successful run ends with a JSON object whose `status` is `PASS`. The detailed
report is written inside the timestamped `tmp/reproduction/` directory. The
staging directory is intentionally ignored by Git.

The `uv.lock` file, rather than the lower-bound compatibility declarations in
`pyproject.toml`, is the exact environment record used by `--frozen` commands.

The independent headline verifier does not import the manuscript analysis. It
recomputes 23 claim-critical components directly from the frozen trial CSVs and
compares them at explicit tolerances. This catches divergent coding and
aggregation, but it is still a same-data check. It cannot reveal a defect shared
by the archived trials and the code that generated them.

The generator spot check addresses a different provenance question. It reruns
the first, midpoint, and last of the 2,000 synthetic markets from their seeds,
including all three search policies and all six budget prefixes. This verifies
the code-to-archive path for those three fixed markets only. It is neither a
full experimental rerun nor an independent replication of the method.
The field-by-field scope and limitations are recorded in
[`paper/generator_provenance_note.md`](paper/generator_provenance_note.md).

## 2. Rerun the experiments

```bash
uv sync --frozen --extra dev

PYTHONPATH=src uv run python -m alpha_firewall.cli synthetic \
  --config configs/deep_paper.toml

PYTHONPATH=src uv run python -m alpha_firewall.cli empirical-null \
  --config configs/deep_paper.toml

PYTHONPATH=src uv run python -m alpha_firewall.cli power \
  --config configs/deep_paper.toml

PYTHONPATH=src uv run python -m alpha_firewall.cli agentic \
  --config configs/deep_paper.toml --claude /path/to/claude

MPLBACKEND=Agg uv run python scripts/analyze_deep.py
tectonic paper/deep_main.tex --outdir output/pdf
```

The synthetic, conditional-null, and planted-signal runs are seeded. A full run
is computationally expensive because it executes 2,293,500 candidate
evaluations. The empirical-null and agent experiments download five French
archives if they are absent locally. Their expected SHA-256 values are recorded
in [DATA.md](DATA.md).

The hosted-model arm is not bit-reproducible. Provider-side model updates and
sampling can change proposals even when prompts and parameters are unchanged.
The release therefore treats the recorded transcript set as the experimental
observation and exposes prompts, responses, model aliases and canonical IDs,
token use, latency, cost, failures, and fallbacks.

## Evidence boundaries

- The PDF and figure byte hashes are provenance only because PDF toolchains may
  insert machine-specific metadata.
- The trial CSVs, manuscript tables, and JSON summaries are deterministic
  evidence objects and are compared numerically or exactly.
- French source archives are not redistributed.
- The sign-randomized panels are exact conditional nulls. They are not evidence
  of a profitable historical strategy.
- The internal analysis plan was frozen before the confirmatory runs but was not
  registered with an external service.
