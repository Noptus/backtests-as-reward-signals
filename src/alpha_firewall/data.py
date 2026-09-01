from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

FRENCH_BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
FRENCH_DAILY_PANELS = {
    "industries_49": "49_Industry_Portfolios_daily_CSV.zip",
    "size_bm_25": "25_Portfolios_5x5_Daily_CSV.zip",
    "size_bm_100": "100_Portfolios_10x10_Daily_CSV.zip",
    "size_momentum_25": "25_Portfolios_ME_Prior_12_2_Daily_CSV.zip",
    "size_short_reversal_25": "25_Portfolios_ME_Prior_1_0_Daily_CSV.zip",
}
FRENCH_49_DAILY_URL = f"{FRENCH_BASE_URL}/{FRENCH_DAILY_PANELS['industries_49']}"


def simulate_null_returns(
    n_days: int,
    n_assets: int,
    seed: int,
    *,
    degrees_of_freedom: float = 6.0,
) -> pd.DataFrame:
    """Simulate heavy-tailed, heteroskedastic returns with zero conditional mean.

    A persistent common volatility state and a random factor-loading structure
    produce realistic dependence without introducing return predictability.
    Signals measurable at t-1 therefore have zero population alpha at t.
    """

    if n_days < 3 or n_assets < 4:
        raise ValueError("simulation requires at least 3 days and 4 assets")

    rng = np.random.default_rng(seed)
    n_factors = min(4, max(2, n_assets // 8))
    loadings = rng.normal(0.0, 0.35, size=(n_assets, n_factors))
    idio_scale = rng.uniform(0.55, 1.0, size=n_assets)

    log_vol = np.empty(n_days)
    log_vol[0] = -0.15
    vol_shocks = rng.normal(0.0, 0.10, size=n_days)
    for day in range(1, n_days):
        log_vol[day] = 0.975 * log_vol[day - 1] + vol_shocks[day]
    common_scale = np.exp(log_vol)

    factor_draws = rng.standard_t(degrees_of_freedom, size=(n_days, n_factors))
    idio_draws = rng.standard_t(degrees_of_freedom, size=(n_days, n_assets))
    raw = factor_draws @ loadings.T + idio_draws * idio_scale
    t_variance = degrees_of_freedom / (degrees_of_freedom - 2.0)
    unconditional_scale = np.sqrt(
        t_variance * (np.square(loadings).sum(axis=1) + np.square(idio_scale))
    )
    raw /= unconditional_scale[None, :]
    daily = 0.012 * common_scale[:, None] * raw

    dates = pd.bdate_range("2000-01-03", periods=n_days)
    columns = [f"asset_{index:02d}" for index in range(n_assets)]
    return pd.DataFrame(daily, index=dates, columns=columns)


def _download_archive(url: str, path: Path) -> None:
    if path.exists():
        return
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "alpha-firewall-research/0.2"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        path.write_bytes(response.read())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_french_daily_archive(archive: Path) -> pd.DataFrame:
    """Parse the first value-weighted daily table from a French CSV archive."""

    with zipfile.ZipFile(archive) as zipped:
        csv_names = [name for name in zipped.namelist() if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"expected one CSV in archive, found {csv_names}")
        text = zipped.read(csv_names[0]).decode("utf-8", errors="replace")

    lines = text.splitlines()
    header_index = next(
        index
        for index, line in enumerate(lines[:-1])
        if line.lstrip().startswith(",")
        and lines[index + 1].split(",", maxsplit=1)[0].strip().isdigit()
        and len(lines[index + 1].split(",", maxsplit=1)[0].strip()) == 8
    )
    end_index = next(
        index for index in range(header_index + 1, len(lines)) if not lines[index].strip()
    )
    block = "\n".join(lines[header_index:end_index])
    frame = pd.read_csv(io.StringIO(block), index_col=0)
    frame.index = pd.to_datetime(frame.index.astype(str), format="%Y%m%d")
    frame = frame.apply(pd.to_numeric, errors="coerce") / 100.0
    frame = frame.replace((-0.9999, -9.99), np.nan).dropna(how="any")
    frame.index.name = "date"
    return frame


def download_french_daily_panel(cache_dir: Path, panel: str) -> pd.DataFrame:
    """Download one canonical daily value-weighted French return panel."""

    if panel not in FRENCH_DAILY_PANELS:
        raise ValueError(f"unknown French panel {panel!r}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = FRENCH_DAILY_PANELS[panel]
    archive = cache_dir / filename
    _download_archive(f"{FRENCH_BASE_URL}/{filename}", archive)
    return _parse_french_daily_archive(archive)


def download_french_49_daily(cache_dir: Path) -> pd.DataFrame:
    """Download and parse value-weighted daily 49-industry portfolio returns."""

    return download_french_daily_panel(cache_dir, "industries_49")


def rademacher_null_returns(returns: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Create an exact conditional-mean null while preserving each cross-section.

    One independent sign is applied jointly to every asset at a timestamp. This
    keeps realized magnitudes, missingness, jumps, and same-date outer products
    intact while making the signed return unpredictable from the past.
    """

    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=len(returns))
    return returns.mul(signs, axis=0)
