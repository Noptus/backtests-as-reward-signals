# Data provenance

## Synthetic markets

The synthetic null and planted-signal experiments are generated locally from
the frozen parameters in `configs/deep_paper.toml`. The null has heavy-tailed,
heteroskedastic, cross-sectionally correlated returns and zero conditional mean.
No external dataset is required.

## Kenneth French panels

The empirical experiment uses daily value-weighted portfolio returns from the
[Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library.html).
The code downloads the archives directly from Dartmouth. This repository does
not redistribute them.

The confirmatory run used these files:

| Panel | Source archive | SHA-256 |
|---|---|---|
| 49 industries | `49_Industry_Portfolios_daily_CSV.zip` | `9b368230566fbf636b64b7a766a74b9a94a15dda5bb9101ff141670526dc1a34` |
| 25 size–book-to-market | `25_Portfolios_5x5_Daily_CSV.zip` | `6482311908b03e2aa9520fc508ed3f828a4400014cec4aefc408702a7b667fd3` |
| 100 size–book-to-market | `100_Portfolios_10x10_Daily_CSV.zip` | `5baab34db5aa2b6c690ed91b00b02db7467bd19da99c113fa35ceb0450c4c878` |
| 25 size–momentum | `25_Portfolios_ME_Prior_12_2_Daily_CSV.zip` | `7ed235883d1711dc9000c3298136bf3f8b4943be0ee201267e312ff052194bfa` |
| 25 size–short-term reversal | `25_Portfolios_ME_Prior_1_0_Daily_CSV.zip` | `a09847f219a2b7408b6bfc30b2a53073eddb8e08b558495f970adc813a9ec738` |

The code extracts the first value-weighted daily table from each archive. Three
eras begin in 1990, 2002, and 2014. For each dataset–era cell, independent
Rademacher signs remove predictable signed returns while preserving the
realized cross-sectional second moment on every date. Target, sham, and
confirmation paths use independent sign sequences.

If Dartmouth revises an archive, the new file may be legitimate but will not be
the file used in the frozen experiment. A hash mismatch should therefore be
treated as a provenance difference, not silently accepted.
