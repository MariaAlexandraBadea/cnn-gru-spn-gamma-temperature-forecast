#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CNN1D + skip + GRU encoder + conditional SPN-Gamma decoder for JOINT multi-horizon forecasting.

Goal: model the full predictive distribution p(y_{t+1:t+H} | x_{t-W+1:t}) of monthly temperature anomalies,
with tractable exact log-likelihood, exact posterior responsibilities p(k|x,y), and explicit horizon-wise
factorization constraints.

Model summary:
- Encoder: 1D CNN with skip connection + GRU to obtain a context embedding h(x).
- Decoder (conditional SPN head): root Sum over K components with neural gating pi_k(h).
  Each component is a Product over horizons, yielding a mixture-of-products that induces cross-horizon
  dependence via the shared latent component k.
- Per horizon and component: signed magnitude model with Bernoulli sign gate s_{k,h}(h) and Gamma mixtures
  for |y_h|, with sign-specific parameters (pos/neg) to capture asymmetry and heavy tails.

Protocol and leakage controls:
- Monthly aggregation is performed first; anomalies are computed per city using a fixed climatology estimated
  only from that city's training years (<= train_end).
- Input gaps are imputed causally inside the past window only (window-only PCHIP, max short-gap), while targets
  are never imputed (windows with missing targets are discarded).
- Geographic generalization: train/val/test are disjoint by cities; time splits are enforced using the horizon
  boundaries (train_end/val_end) and the first/last target year in the horizon.

Evaluation and robustness:
- Metrics include point accuracy (RMSE/MAE), probabilistic quality (CRPS, NLL per horizon), and multivariate
  dependence-aware scoring (Energy Score) computed for (i) joint sampling and (ii) an independent-k baseline
  (per-horizon latent k) to isolate the contribution of cross-horizon coupling.
- Results are reported globally on pooled test windows and additionally by decade/regime (grouped by the first
  horizon year) to assess non-stationarity.
- Sensitivity to held-out city choice is addressed via multiple stratified, disjoint 10-city test sets.
"""

import argparse
import math
import random
import re
import unicodedata
import os
import json
import time
import zipfile
import platform
import sys
import gc

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import t as student_t, chi2

# ---------------- Hardcoded path + Config ----------------

CSV_PATH = "/content/sample_data/city_temperature.csv"

WINDOW = 12
HORIZON = 6          # JOINT multi-horizon length (set to 6 or 12)

HIDDEN = 64
K_ROOT = 8           # root Sum fan-out
M_MAG = 4            # magnitude mixture leaves per (k, horizon, sign)

DROPOUT = 0.30
BATCH = 256
EPOCHS = 200
LR = 3e-3
WARMUP = 10
PATIENCE = 20

# preprocessing
MAX_PCHIP_GAP = 2
MIN_MONTHS_PER_YEAR = 10
MIN_TOTAL_YEARS_FOR_CITY = 18 # of ~26 years available
MIN_TRAIN_YEARS_FOR_CLIM = 5  # years with >=10 months, up to 2000

# probabilistic / regularization
ALPHA_FLOOR = 1e-4
BETA_FLOOR = 1e-4
L2_LOG_PARAM = 1e-4
ENTROPY_WEIGHT = 1e-3

# geographic split
VAL_FRACTION = 0.10
MIN_VAL_CITIES = 20
MAX_VAL_CITIES = 60

# test/eval cities
EVAL_N = 10
N_EVAL_SETS = 5         # no of different sets of 10 cities
EVAL_SEED_BASE = 123    # base seed for city selection
SPLIT_SEED_BASE = 1000  # base seed for geographic split (train/wave vs test)

EVAL_MODE = "stratified_sets"   # "fixed" / "stratified_sets"
ENFORCE_DISJOINT_EVAL_SETS = True

#EVAL_CITIES_FIXED = ["Lome","Manila","Sydney","Lisbon","Bratislava","Kuwait","Anchorage","Buffalo","Albany","Sao Paulo"]
#EVAL_CITIES_FIXED = ["Cairo", "Seoul", "Perth", "London", "Lisbon", "Kuwait", "Cleveland", "Cincinnati", "Brownsville", "Rio de Janeiro"]

SEED = 42

# ---------------- Artifacts (figures/tables) ----------------
ART_DIR = Path("artifacts")
FIG_DIR = ART_DIR / "figures"
TAB_DIR = ART_DIR / "tables"
NPZ_DIR = ART_DIR / "npz"

ART_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)
NPZ_DIR.mkdir(parents=True, exist_ok=True)

def save_fig(fig, name: str, dpi: int = 200):
    png = FIG_DIR / f"{name}.png"
    pdf = FIG_DIR / f"{name}.pdf"
    fig.tight_layout()
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[ART] saved figure: {png} and {pdf}")
    return str(png), str(pdf)

def save_table(df: pd.DataFrame, name: str, float_fmt: str = "%.6f"):
    csv_path = TAB_DIR / f"{name}.csv"
    tex_path = TAB_DIR / f"{name}.tex"
    df.to_csv(csv_path, index=False)
    df.to_latex(tex_path, index=False, float_format=lambda x: float_fmt % x)
    print(f"[ART] saved table: {csv_path} and {tex_path}")
    return str(csv_path), str(tex_path)

def zip_artifacts(zip_name: str = "artifacts.zip"):
    zip_path = ART_DIR / zip_name
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in ART_DIR.rglob("*"):
            if p.is_file() and p.name != zip_name:
                z.write(p, p.relative_to(ART_DIR))
    print(f"[ART] zipped -> {zip_path}")
    return str(zip_path)

# flags (generate once)
_FIG_R24_DONE = False
_FIG_R26_DONE = False


# ---------------- Reproducibility ----------------

def set_seed(seed: int = SEED, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # hard determinism
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


# ---------------- Data utilities ----------------

REQ_COLS = ["Region", "Country", "State", "City", "Month", "Day", "Year", "AvgTemperature"]


def norm_city_name(s: str) -> str:
    s = str(s).strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z\s\-]", "", s)
    return s.strip()


def read_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False, usecols=REQ_COLS)

    for c in ["Region", "Country", "State", "City"]:
        df[c] = df[c].astype(str)

    df["Date"] = pd.to_datetime(df[["Year", "Month", "Day"]], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()

    # sentinel + Fahrenheit -> Celsius
    df.loc[df["AvgTemperature"] <= -90, "AvgTemperature"] = np.nan
    df["AvgTemperature"] = (df["AvgTemperature"] - 32.0) * (5.0 / 9.0)
    return df


def probe_span(df: pd.DataFrame) -> tuple[int, int]:
    years = df["Date"].dt.year.to_numpy()
    return int(np.nanmin(years)), int(np.nanmax(years))


def aggregate_monthly(df: pd.DataFrame, year_min: int, year_max: int) -> pd.DataFrame:
    df = df[(df["Date"].dt.year >= year_min) & (df["Date"].dt.year <= year_max)].copy()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    monthly = df.groupby(["City", "Year", "Month"], as_index=False)[["AvgTemperature"]].mean()
    return monthly


def exclude_incomplete_years(monthly: pd.DataFrame, min_months: int) -> pd.DataFrame:
    counts = (
        monthly.dropna(subset=["AvgTemperature"])
        .groupby(["City", "Year"])["Month"].nunique()
        .reset_index(name="n_months")
    )
    keep = counts[counts["n_months"] >= min_months][["City", "Year"]]
    return monthly.merge(keep, on=["City", "Year"], how="inner")


def reindex_month_grid(monthly: pd.DataFrame, y0: int, y1: int) -> pd.DataFrame:
    """
    Much lower-RAM version:
    - City stored as categorical + integer CityCode in the big grid
    - merge happens on CityCode (int), not on City (object strings)
    """
    monthly = monthly.copy()

    # compress City column
    monthly["City"] = monthly["City"].astype("category")
    city_cat = monthly["City"].cat.categories

    monthly["CityCode"] = monthly["City"].cat.codes.astype(np.int32)

    # big full grid uses int codes instead of strings
    city_codes = np.arange(len(city_cat), dtype=np.int32)
    idx = pd.MultiIndex.from_product(
        [city_codes, range(y0, y1 + 1), range(1, 13)],
        names=["CityCode", "Year", "Month"],
    )
    full = pd.DataFrame(index=idx).reset_index()

    # merge on ints
    merged = full.merge(
        monthly.drop(columns=["City"]),
        on=["CityCode", "Year", "Month"],
        how="left",
        copy=False,
    )

    # restore City as categorical
    merged["City"] = pd.Categorical.from_codes(
        merged["CityCode"].astype(np.int32),
        categories=city_cat,
    )
    merged = merged.drop(columns=["CityCode"])

    return merged

def split_years(year_min: int, year_max: int) -> tuple[int, int]:
    if year_max - year_min + 1 < 25:
        years = list(range(year_min, year_max + 1))
        n = len(years)
        i1 = int(0.6 * n)
        i2 = int(0.8 * n)
        return years[i1 - 1], years[i2 - 1]

    # Fixed temporal split to avoid information leakage between training, validation and test periods
    train_end = min(1999, year_max - 21)
    val_end   = min(2000, year_max - 10)
    return train_end, val_end


def build_anomalies_fixed_climatology(monthly: pd.DataFrame, train_end: int) -> pd.DataFrame:
    out_parts = []
    for city, g in monthly.groupby("City", sort=False):
        g = g.sort_values(["Year", "Month"]).reset_index(drop=True).copy()

        mask_tr = g["Year"] <= train_end
        tr = g.loc[mask_tr, ["Month", "AvgTemperature"]].copy()

        clim = tr.groupby("Month")["AvgTemperature"].mean().reindex(range(1, 13))
        clim = clim.ffill().bfill()

        if clim.isna().any():
            gmean = float(tr["AvgTemperature"].mean()) if np.isfinite(tr["AvgTemperature"]).any() else 0.0
            clim = clim.fillna(gmean)

        g["Clim"] = g["Month"].map(clim.to_dict())
        g["Anomaly_raw"] = g["AvgTemperature"] - g["Clim"]
        g["is_observed_target"] = g["AvgTemperature"].notna()
        out_parts.append(g)

    return pd.concat(out_parts, ignore_index=True)


def build_city_region_map(df_raw: pd.DataFrame) -> dict[str, str]:
    # most common Region per City (robust)
    tmp = df_raw[["City", "Region"]].dropna().copy()
    tmp["City"] = tmp["City"].astype(str)
    tmp["Region"] = tmp["Region"].astype(str)

    reg = (
        tmp.groupby(["City", "Region"])
        .size()
        .reset_index(name="n")
        .sort_values(["City", "n"], ascending=[True, False])
    )
    # take first Region (mode) per City
    city_region = reg.drop_duplicates("City")[["City", "Region"]]
    return dict(zip(city_region["City"], city_region["Region"]))


def count_test_windows_for_city(
    g_city: pd.DataFrame,
    window: int,
    horizon: int,
    val_end: int,
    max_pchip_gap: int,
) -> int:
    g = g_city.sort_values(["Year", "Month"]).copy()

    a_raw = g["Anomaly_raw"].to_numpy(dtype=np.float32)
    yy = g["Year"].to_numpy(dtype=np.int32)
    is_obs_tgt = g["is_observed_target"].to_numpy(dtype=bool)

    T = len(a_raw)
    max_start = T - window - horizon
    if max_start <= 0:
        return 0

    cnt = 0
    for i in range(max_start):
        t = i + window
        t_end = t + horizon - 1

        # test condition: first target year > val_end
        if int(yy[t]) <= val_end:
            continue

        # targets must be observed
        yvec = a_raw[t:t + horizon]
        obsvec = is_obs_tgt[t:t + horizon]
        if (not np.all(np.isfinite(yvec))) or (not np.all(obsvec)):
            continue

        # causal window must be imputable with short-gap PCHIP
        w_raw = a_raw[i:i + window].astype(np.float32)
        w = pchip_impute_short_gaps_window(w_raw, max_gap=max_pchip_gap)
        if not np.all(np.isfinite(w)):
            continue

        cnt += 1

    return cnt

def city_year_counts(monthly_city: pd.DataFrame, train_end: int) -> tuple[int, int]:
    obs = monthly_city.dropna(subset=["AvgTemperature"])
    years = obs.groupby("Year")["Month"].nunique()
    total_years = int((years >= MIN_MONTHS_PER_YEAR).sum())
    train_years = int(((years.index <= train_end) & (years >= MIN_MONTHS_PER_YEAR)).sum())
    return total_years, train_years

def build_eval_sets_disjoint(
    df_raw: pd.DataFrame,
    monthly: pd.DataFrame,
    window: int,
    horizon: int,
    train_end: int,
    val_end: int,
    max_pchip_gap: int,
    n_eval_sets: int,
    n_eval: int,
    seed_base: int,
    min_test_windows: int = 60,
    enforce_disjoint: bool = True,
) -> list[list[str]]:
    used_norm: set[str] = set()
    eval_sets: list[list[str]] = []

    for i in range(n_eval_sets):
        eval_seed = seed_base + i
        exclude = used_norm if enforce_disjoint else set()

        cities = select_eval_cities_stratified(
            df_raw=df_raw,
            monthly=monthly,
            window=window,
            horizon=horizon,
            train_end=train_end,
            val_end=val_end,
            max_pchip_gap=max_pchip_gap,
            n_eval=n_eval,
            seed=eval_seed,
            min_test_windows=min_test_windows,
            exclude_cities_norm=exclude,
        )

        eval_sets.append(cities)
        if enforce_disjoint:
            used_norm.update(norm_city_name(c) for c in cities)

    return eval_sets

def select_eval_cities_stratified(
    df_raw: pd.DataFrame,
    monthly: pd.DataFrame,
    window: int,
    horizon: int,
    train_end: int,
    val_end: int,
    max_pchip_gap: int,
    n_eval: int,
    seed: int,
    min_test_windows: int = 60,
    exclude_cities_norm: set[str] | None = None,
) -> list[str]:
    """
    Select n_eval cities stratified by Region -> cities with sufficient TEST windows.
    Deterministic (seed) only for tie-break/top pick.
    """
    if exclude_cities_norm is None:
            exclude_cities_norm = set()

    city_region = build_city_region_map(df_raw)

    # count test windows per city
    counts = {}
    for city, g in monthly.groupby("City", sort=False):
        counts[city] = count_test_windows_for_city(
            g, window=window, horizon=horizon, val_end=val_end, max_pchip_gap=max_pchip_gap
        )

    # candidate pool
    candidates = [c for c, k in counts.items() if (k >= min_test_windows) and (c in city_region) and (norm_city_name(c) not in exclude_cities_norm)]

    if len(candidates) < n_eval:
        # fallback: relax threshold
        candidates = [c for c, k in counts.items() if (k > 0) and (c in city_region) and (norm_city_name(c) not in exclude_cities_norm)]

    filtered = []
    for c in candidates:
        g = monthly[monthly["City"] == c]
        tot_y, tr_y = city_year_counts(g, train_end=train_end)
        if tot_y >= MIN_TOTAL_YEARS_FOR_CITY and tr_y >= MIN_TRAIN_YEARS_FOR_CLIM:
            filtered.append(c)
    candidates = filtered

    # group by region
    region2cities = {}
    for c in candidates:
        r = city_region.get(c, "Unknown")
        region2cities.setdefault(r, []).append(c)

    # sort cities in each region by test window count desc
    for r in region2cities:
        region2cities[r].sort(key=lambda c: counts[c], reverse=True)

    regions = sorted(region2cities.keys())

    # quota: 1 per region, rest proportional (based on candidate counts)
    base = {r: 1 for r in regions}
    remaining = n_eval - len(regions)
    if remaining < 0:
        # too many regions for n_eval: keep top regions by size
        regions = sorted(regions, key=lambda r: len(region2cities[r]), reverse=True)[:n_eval]
        base = {r: 1 for r in regions}
        remaining = 0

    # proportional add
    sizes = np.array([len(region2cities[r]) for r in regions], dtype=np.float64)
    if remaining > 0 and sizes.sum() > 0:
        frac = sizes / sizes.sum()
        extra = np.floor(frac * remaining).astype(int)
        # distribute leftover by largest remainder
        leftover = remaining - int(extra.sum())
        rema = frac * remaining - extra
        order = np.argsort(-rema)
        for j in range(leftover):
            extra[order[j]] += 1
        for r, e in zip(regions, extra):
            base[r] += int(e)

    rng = np.random.default_rng(seed)

    chosen = []
    # pick per region from top list (tie-break with RNG among top chunk)
    for r in regions:
        need = base[r]
        pool = region2cities[r]
        if len(pool) == 0:
            continue
        # take from top 20 for diversity
        top = pool[: min(20, len(pool))]
        # deterministic shuffle in top then take need
        idx = np.arange(len(top))
        rng.shuffle(idx)
        pick = [top[i] for i in idx[: min(need, len(top))]]
        chosen.extend(pick)

    chosen = list(dict.fromkeys(chosen))  # unique preserve order

    # fill remaining globally by highest count
    if len(chosen) < n_eval:
        remaining_pool = [c for c in candidates if c not in chosen]
        remaining_pool.sort(key=lambda c: counts[c], reverse=True)
        chosen.extend(remaining_pool[: (n_eval - len(chosen))])

    return chosen[:n_eval]

def diagnose_eval_city_pool(
    df_raw: pd.DataFrame,
    monthly: pd.DataFrame,
    window: int,
    horizon: int,
    val_end: int,
    max_pchip_gap: int,
    min_test_windows: int = 60,
):
    city_region = build_city_region_map(df_raw)

    counts = {}
    for city, g in monthly.groupby("City", sort=False):
        counts[city] = count_test_windows_for_city(
            g, window=window, horizon=horizon, val_end=val_end, max_pchip_gap=max_pchip_gap
        )

    candidates = [c for c, k in counts.items() if k >= min_test_windows and c in city_region]
    print(f"[DIAG] Candidate cities with >= {min_test_windows} TEST windows: {len(candidates)}")

    # per region
    reg2n = {}
    for c in candidates:
        r = city_region.get(c, "Unknown")
        reg2n[r] = reg2n.get(r, 0) + 1

    for r in sorted(reg2n.keys()):
        print(f"  - {r}: {reg2n[r]}")

def plot_dataset_examples_once(monthly_with_anom: pd.DataFrame, city_list: list[str],
                               train_end: int, val_end: int):
    global _FIG_R26_DONE
    if _FIG_R26_DONE:
        return
    _FIG_R26_DONE = True

    fig, ax = plt.subplots(figsize=(12, 4))

    for city in city_list:
        d = monthly_with_anom[monthly_with_anom["City"] == city].copy()
        if len(d) == 0:
            continue
        d["date"] = pd.to_datetime(dict(year=d["Year"], month=d["Month"], day=1))
        d = d.sort_values("date")
        ax.plot(d["date"], d["Anomaly_raw"], label=city, linewidth=1.0)

    ax.axvline(pd.Timestamp(train_end, 12, 31), linestyle="--", linewidth=1.0)
    ax.axvline(pd.Timestamp(val_end, 12, 31), linestyle="--", linewidth=1.0)
    ax.set_title("Monthly temperature anomalies - dataset examples")
    ax.set_ylabel("Anomaly (C)")
    ax.legend(ncol=2, fontsize=8)

    save_fig(fig, "fig_r26_dataset_examples")


# --------- Causal, window-only PCHIP (no future leakage) ---------

def pchip_impute_short_gaps_window(values: np.ndarray, max_gap: int) -> np.ndarray:
    y = values.astype(np.float64).copy()
    n = len(y)
    isnan = ~np.isfinite(y)
    if not isnan.any():
        return y.astype(np.float32)

    obs = np.isfinite(y)
    if obs.sum() < 2:
        return y.astype(np.float32)

    x = np.arange(n, dtype=np.float64)
    f = PchipInterpolator(x[obs], y[obs], extrapolate=False)

    i = 0
    while i < n:
        if isnan[i]:
            j = i
            while j < n and isnan[j]:
                j += 1
            gap_len = j - i
            internal = (i > 0) and (j < n)
            if internal and gap_len <= max_gap:
                y[i:j] = f(x[i:j])
            i = j
        else:
            i += 1

    return y.astype(np.float32)


# --------- Geographic split (train/val/test cities disjoint) ---------
def split_cities_geographic(all_cities: list[str], eval_set_norm: set[str], seed: int) -> tuple[set[str], set[str], set[str]]:
    all_norm = sorted({norm_city_name(c) for c in all_cities})
    test_norm = set(all_norm) & set(eval_set_norm)
    non_test_norm = [cn for cn in all_norm if cn not in test_norm]

    rng = np.random.default_rng(seed)
    n_val = int(round(len(non_test_norm) * VAL_FRACTION))
    n_val = max(n_val, MIN_VAL_CITIES)
    n_val = min(n_val, MAX_VAL_CITIES, max(1, len(non_test_norm) // 2))
    n_val = min(n_val, len(non_test_norm))  # safety

    val_norm = set(rng.choice(non_test_norm, size=n_val, replace=False).tolist()) if n_val > 0 else set()
    train_norm = set([cn for cn in non_test_norm if cn not in val_norm])

    return train_norm, val_norm, test_norm


def make_windows_global_joint(
    df: pd.DataFrame,
    window: int,
    horizon: int,
    train_end: int,
    val_end: int,
    train_cities_norm: set[str],
    val_cities_norm: set[str],
    test_cities_norm: set[str],
    max_pchip_gap: int,
) -> tuple[
    dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]
]:
    """
    JOINT horizon windowing: targets are vectors y[t : t+horizon]
    Strict rules:
    - Inputs built from past window only; causal PCHIP inside window only
    - Skip if any target in horizon is missing / not observed
    - Time split uses horizon end:
        TRAIN: last target year <= train_end
        VAL:   first target year > train_end and last target year <= val_end
        TEST:  first target year > val_end (test cities only)
    """
    cities = sorted(df["City"].unique())
    city2id = {c: i for i, c in enumerate(cities)}

    Xs = {"train": [], "val": [], "test": []}
    Ys = {"train": [], "val": [], "test": []}
    Cs = {"train": [], "val": [], "test": []}
    Y1years = {"train": [], "val": [], "test": []}  # year of first horizon target (for decade grouping)

    for city, g in df.groupby("City", sort=False):
        g = g.sort_values(["Year", "Month"]).copy()
        cid = city2id[city]
        city_norm = norm_city_name(city)

        a_raw = g["Anomaly_raw"].to_numpy(dtype=np.float32)
        m = g["Month"].to_numpy(dtype=np.int32)
        yy = g["Year"].to_numpy(dtype=np.int32)
        is_obs_tgt = g["is_observed_target"].to_numpy(dtype=bool)

        if city_norm in test_cities_norm:
            which_city = "test"
        elif city_norm in val_cities_norm:
            which_city = "val"
        elif city_norm in train_cities_norm:
            which_city = "train"
        else:
            continue

        T = len(a_raw)
        max_start = T - window - horizon + 1
        if max_start <= 0:
            continue

        for i in range(max_start):
            t = i + window
            t_end = t + horizon - 1

            # targets must be observed (no imputation for scoring)
            yvec = a_raw[t:t + horizon]
            obsvec = is_obs_tgt[t:t + horizon]
            if (not np.all(np.isfinite(yvec))) or (not np.all(obsvec)):
                continue

            y_year_first = int(yy[t])
            y_year_last = int(yy[t_end])

            # enforce split by city + time (using horizon end)
            if which_city == "train":
                if y_year_last > train_end:
                    continue
                split_key = "train"
            elif which_city == "val":
                if not (y_year_first > train_end and y_year_last <= val_end):
                    continue
                split_key = "val"
            else:  # test
                if y_year_first <= val_end:
                    continue
                split_key = "test"

            # build causal input window
            w_raw = a_raw[i:i + window].astype(np.float32)
            w = pchip_impute_short_gaps_window(w_raw, max_gap=max_pchip_gap)
            if not np.all(np.isfinite(w)):
                continue

            sin_m = np.sin(2 * np.pi * ((m[i:i + window] - 1) / 12.0))
            cos_m = np.cos(2 * np.pi * ((m[i:i + window] - 1) / 12.0))
            delta = np.concatenate([[0.0], np.diff(w)])
            feats = np.stack([w, sin_m, cos_m, delta], axis=1).astype(np.float32)  # [W,4]

            Xs[split_key].append(feats)
            Ys[split_key].append(yvec.astype(np.float32))
            Cs[split_key].append(cid)
            Y1years[split_key].append(y_year_first)

    def stack_X(lst):
        return np.asarray(lst, np.float32) if len(lst) > 0 else np.empty((0, window, 4), np.float32)

    def stack_Y(lst):
        return np.asarray(lst, np.float32) if len(lst) > 0 else np.empty((0, horizon), np.float32)

    def stack_c(lst):
        return np.asarray(lst, np.int64) if len(lst) > 0 else np.empty((0,), np.int64)

    def stack_yr(lst):
        return np.asarray(lst, np.int32) if len(lst) > 0 else np.empty((0,), np.int32)

    return (
        {k: stack_X(v) for k, v in Xs.items()},
        {k: stack_Y(v) for k, v in Ys.items()},
        {k: stack_c(v) for k, v in Cs.items()},
        {k: stack_yr(v) for k, v in Y1years.items()},
        city2id,
    )


# ---------------- Model: Joint multi-horizon conditional SPN ----------------
class JointConditionalSPNHead(nn.Module):
    def __init__(self, in_dim: int, k_root: int, horizon: int, m_mag: int,
                 alpha_floor: float, beta_floor: float,
                 shared_magnitude: bool = False):
        super().__init__()
        self.k_root = k_root
        self.horizon = horizon
        self.m_mag = m_mag
        self.shared_magnitude = shared_magnitude

        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 64)

        self.fc_pi_root = nn.Linear(64, k_root)
        self.fc_s = nn.Linear(64, k_root * horizon)

        size_khm = k_root * horizon * m_mag

        if self.shared_magnitude:
            self.fc_pi_mag = nn.Linear(64, size_khm)
            self.fc_alpha_mag = nn.Linear(64, size_khm)
            self.fc_beta_mag  = nn.Linear(64, size_khm)
        else:
            self.fc_pi_pos = nn.Linear(64, size_khm)
            self.fc_pi_neg = nn.Linear(64, size_khm)
            self.fc_alpha_pos = nn.Linear(64, size_khm)
            self.fc_beta_pos  = nn.Linear(64, size_khm)
            self.fc_alpha_neg = nn.Linear(64, size_khm)
            self.fc_beta_neg  = nn.Linear(64, size_khm)

        self.alpha_floor = alpha_floor
        self.beta_floor = beta_floor

    def forward(self, h: torch.Tensor):
        z = F.relu(self.fc1(h))
        z = F.relu(self.fc2(z))

        pi_root = F.softmax(self.fc_pi_root(z), dim=-1)  # [B,K]
        s = torch.sigmoid(self.fc_s(z)).view(-1, self.k_root, self.horizon)  # [B,K,H]

        def smx(v):
            v = v.view(-1, self.k_root, self.horizon, self.m_mag)
            return F.softmax(v, dim=-1)

        def sp(v):
            v = v.view(-1, self.k_root, self.horizon, self.m_mag)
            return F.softplus(v)

        if self.shared_magnitude:
            pi_mag = smx(self.fc_pi_mag(z))
            a_mag  = sp(self.fc_alpha_mag(z)) + self.alpha_floor
            b_mag  = sp(self.fc_beta_mag(z))  + self.beta_floor

            # tie pos/neg to same magnitude params
            pi_pos, pi_neg = pi_mag, pi_mag
            alpha_pos, alpha_neg = a_mag, a_mag
            beta_pos,  beta_neg  = b_mag, b_mag
        else:
            pi_pos = smx(self.fc_pi_pos(z))
            pi_neg = smx(self.fc_pi_neg(z))

            alpha_pos = sp(self.fc_alpha_pos(z)) + self.alpha_floor
            beta_pos  = sp(self.fc_beta_pos(z))  + self.beta_floor
            alpha_neg = sp(self.fc_alpha_neg(z)) + self.alpha_floor
            beta_neg  = sp(self.fc_beta_neg(z))  + self.beta_floor

        return pi_root, s, pi_pos, alpha_pos, beta_pos, pi_neg, alpha_neg, beta_neg


# Encoder that maps the input window to a compact context embedding used by the SPN gating network
class Encoder(nn.Module):
    def __init__(self, in_ch: int, hidden: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_ch, out_channels=16, kernel_size=3, padding=1)
        self.skip = nn.Linear(in_ch, 16)
        self.gru = nn.GRU(input_size=16, hidden_size=hidden, num_layers=1,
                          batch_first=True, bidirectional=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_in = self.skip(x)  # [B,W,16]
        z = F.relu(self.conv(x.permute(0, 2, 1))).permute(0, 2, 1)  # [B,W,16]
        z = z + skip_in
        out, _ = self.gru(z)
        return self.drop(out[:, -1, :])  # [B,2H]


# End-to-end model: CNN+BiGRU encoder followed by a joint conditional SPN-Gamma decoder
class GlobalJointConditionalSPN(nn.Module):
    def __init__(self, in_ch: int, hidden: int, k_root: int, horizon: int, m_mag: int,
                 dropout: float, alpha_floor: float, beta_floor: float, shared_magnitude: bool = False):
        super().__init__()
        self.k_root = k_root
        self.horizon = horizon
        self.m_mag = m_mag

        self.enc = Encoder(in_ch=in_ch, hidden=hidden, dropout=dropout)
        self.head = JointConditionalSPNHead(
            in_dim=hidden, k_root=k_root, horizon=horizon, m_mag=m_mag,
            alpha_floor=alpha_floor, beta_floor=beta_floor, shared_magnitude = shared_magnitude
        )

    # Log-density of Gamma(|y|) in (alpha, beta) parameterization, used for magnitude likelihood.
    @staticmethod
    def gamma_logpdf_scale(u: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        return -torch.lgamma(alpha) - alpha * torch.log(beta) + (alpha - 1.0) * torch.log(u) - (u / beta)

    def forward_params(self, x: torch.Tensor):
        h = self.enc(x)
        return self.head(h)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        y: [B,H]
        Exact SPN evaluation:
          log p(y|x) = logsumexp_k [ log pi_k + sum_h log p_h(y_h|k,x) ]
        """
        pi_root, s, pi_pos, a_pos, b_pos, pi_neg, a_neg, b_neg = self.forward_params(x)

        eps = 1e-8
        B, H = y.shape
        K = self.k_root
        M = self.m_mag

        y_b1h1 = y.view(B, 1, H, 1)                      # [B,1,H,1]
        u = torch.abs(y_b1h1) + eps                      # [B,1,H,1]
        sign_pos = (y >= 0).float().view(B, 1, H)        # [B,1,H]

        # broadcast to [B,K,H,M]
        u_bkhm = u.expand(B, K, H, M)

        log_ga_pos = self.gamma_logpdf_scale(u_bkhm, a_pos, b_pos)  # [B,K,H,M]
        log_ga_neg = self.gamma_logpdf_scale(u_bkhm, a_neg, b_neg)  # [B,K,H,M]

        log_pu_pos = torch.logsumexp(torch.log(pi_pos + 1e-12) + log_ga_pos, dim=3)  # [B,K,H]
        log_pu_neg = torch.logsumexp(torch.log(pi_neg + 1e-12) + log_ga_neg, dim=3)  # [B,K,H]

        log_s = torch.log(s + 1e-12)                # [B,K,H]
        log_1s = torch.log(1.0 - s + 1e-12)         # [B,K,H]

        # choose sign branch per y_h using indicator
        sign_pos_bkh = sign_pos.expand(B, K, H)
        log_ph = sign_pos_bkh * (log_s + log_pu_pos) + (1.0 - sign_pos_bkh) * (log_1s + log_pu_neg)  # [B,K,H]

        log_pk = log_ph.sum(dim=2)                  # Product over horizons -> sum logs, [B,K]
        log_mix = torch.log(pi_root + 1e-12) + log_pk
        return torch.logsumexp(log_mix, dim=1)       # [B]

    @torch.no_grad()
    def posterior_k(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Returns responsibilities r_{b,k} = p(k | y, x), shape [B,K].
        This is exact inference in the circuit (upward + normalization).
        """
        pi_root, s, pi_pos, a_pos, b_pos, pi_neg, a_neg, b_neg = self.forward_params(x)

        eps = 1e-8
        B, H = y.shape
        K = self.k_root
        M = self.m_mag

        y_b1h1 = y.view(B, 1, H, 1)
        u = torch.abs(y_b1h1) + eps
        sign_pos = (y >= 0).float().view(B, 1, H)

        u_bkhm = u.expand(B, K, H, M)

        log_ga_pos = self.gamma_logpdf_scale(u_bkhm, a_pos, b_pos)
        log_ga_neg = self.gamma_logpdf_scale(u_bkhm, a_neg, b_neg)

        log_pu_pos = torch.logsumexp(torch.log(pi_pos + 1e-12) + log_ga_pos, dim=3)  # [B,K,H]
        log_pu_neg = torch.logsumexp(torch.log(pi_neg + 1e-12) + log_ga_neg, dim=3)  # [B,K,H]

        log_s = torch.log(s + 1e-12)
        log_1s = torch.log(1.0 - s + 1e-12)

        sign_pos_bkh = sign_pos.expand(B, K, H)
        log_ph = sign_pos_bkh * (log_s + log_pu_pos) + (1.0 - sign_pos_bkh) * (log_1s + log_pu_neg)  # [B,K,H]

        log_pk = log_ph.sum(dim=2)  # [B,K]
        log_joint = torch.log(pi_root + 1e-12) + log_pk  # proportional to p(k,y|x)
        return torch.softmax(log_joint, dim=1)


    def log_prob_masked(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Exact marginal log-likelihood with missing horizons marginalized out.
        y:    [B,H]
        mask: [B,H] with 1 for observed horizons, 0 for marginalized horizons

        Because the circuit is decomposable and has a Product over horizons,
        marginalizing a horizon corresponds to removing its factor (multiply by 1),
        i.e., skip its log-term.
        """
        pi_root, s, pi_pos, a_pos, b_pos, pi_neg, a_neg, b_neg = self.forward_params(x)

        eps = 1e-8
        B, H = y.shape
        K = self.k_root
        M = self.m_mag

        y_b1h1 = y.view(B, 1, H, 1)
        u = torch.abs(y_b1h1) + eps
        sign_pos = (y >= 0).float().view(B, 1, H)

        u_bkhm = u.expand(B, K, H, M)

        log_ga_pos = self.gamma_logpdf_scale(u_bkhm, a_pos, b_pos)
        log_ga_neg = self.gamma_logpdf_scale(u_bkhm, a_neg, b_neg)

        log_pu_pos = torch.logsumexp(torch.log(pi_pos + 1e-12) + log_ga_pos, dim=3)  # [B,K,H]
        log_pu_neg = torch.logsumexp(torch.log(pi_neg + 1e-12) + log_ga_neg, dim=3)  # [B,K,H]

        log_s = torch.log(s + 1e-12)
        log_1s = torch.log(1.0 - s + 1e-12)

        sign_pos_bkh = sign_pos.expand(B, K, H)
        log_ph = sign_pos_bkh * (log_s + log_pu_pos) + (1.0 - sign_pos_bkh) * (log_1s + log_pu_neg)  # [B,K,H]

        # apply mask: marginalized horizons contribute 0 to sum of logs
        mask_bkh = mask.view(B, 1, H).expand(B, K, H).float()
        log_pk = (log_ph * mask_bkh).sum(dim=2)  # [B,K]

        log_mix = torch.log(pi_root + 1e-12) + log_pk
        return torch.logsumexp(log_mix, dim=1)  # [B]

    # Training objective: maximize average log-likelihood with mild regularization and optional gating-entropy term
    def train_objective(self, x: torch.Tensor, y: torch.Tensor, l2_log_param: float, entropy_weight: float):
        logp = self.log_prob(x, y).mean()

        pi_root, _, pi_pos, a_pos, b_pos, pi_neg, a_neg, b_neg = self.forward_params(x)

        reg = l2_log_param * (
            torch.log(a_pos).pow(2).mean() + torch.log(b_pos).pow(2).mean() +
            torch.log(a_neg).pow(2).mean() + torch.log(b_neg).pow(2).mean()
        )

        pr = pi_root.clamp_min(1e-12)
        entropy_root = -(pr * torch.log(pr)).sum(dim=1).mean()
        reg -= entropy_weight * entropy_root

        return -(logp) + reg

    # Predictive mean E[y|x] computed in closed form by marginalizing the SPN mixture components
    def predictive_mean(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns mean per horizon: [B,H]
        """
        pi_root, s, pi_pos, a_pos, b_pos, pi_neg, a_neg, b_neg = self.forward_params(x)

        eu_pos = (pi_pos * (a_pos * b_pos)).sum(dim=3)  # [B,K,H]
        eu_neg = (pi_neg * (a_neg * b_neg)).sum(dim=3)  # [B,K,H]

        mean_kh = s * eu_pos - (1.0 - s) * eu_neg        # [B,K,H]
        return (pi_root.unsqueeze(-1) * mean_kh).sum(dim=1)  # [B,H]

    # Draw joint multi-horizon samples; shared latent component induces cross-horizon dependence
    def sample(self, x: torch.Tensor, n_samples: int, dispersion_scale: float | None = None) -> torch.Tensor:
        """
        Joint sampling: [S,B,H]
        Mixture-of-products induces dependencies across horizons via shared k.
        """
        pi_root, s, pi_pos, a_pos, b_pos, pi_neg, a_neg, b_neg = self.forward_params(x)
        B, K = pi_root.shape
        H = self.horizon
        M = self.m_mag
        S = n_samples

        cat_k = torch.distributions.Categorical(probs=pi_root)
        k_idx = cat_k.sample((S,))  # [S,B]

        # expand tensors for gather
        idx_k = k_idx.unsqueeze(-1).unsqueeze(-1)  # [S,B,1,1]

        def gather_k_3(t_bkh: torch.Tensor) -> torch.Tensor:
            t = t_bkh.unsqueeze(0).expand(S, B, K, H)
            idx = k_idx.unsqueeze(-1).unsqueeze(-1).expand(S, B, 1, H)
            return torch.gather(t, 2, idx).squeeze(2)  # [S,B,H]

        def gather_k_4(t_bkhm: torch.Tensor) -> torch.Tensor:
            t = t_bkhm.unsqueeze(0).expand(S, B, K, H, M)
            idx = k_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(S, B, 1, H, M)
            return torch.gather(t, 2, idx).squeeze(2)  # [S,B,H,M]

        s_k = gather_k_3(s)             # [S,B,H]

        # sample sign per horizon
        sign_draw = torch.distributions.Bernoulli(probs=s_k).sample()  # [S,B,H] in {0,1}
        sign_factor = 2.0 * sign_draw - 1.0

        pi_pos_k = gather_k_4(pi_pos)
        pi_neg_k = gather_k_4(pi_neg)
        a_pos_k  = gather_k_4(a_pos)
        b_pos_k  = gather_k_4(b_pos)
        a_neg_k  = gather_k_4(a_neg)
        b_neg_k  = gather_k_4(b_neg)

        sign_mask = sign_draw.unsqueeze(-1)  # [S,B,H,1]
        pi_mix = torch.where(sign_mask > 0.5, pi_pos_k, pi_neg_k)  # [S,B,H,M]
        a_mix  = torch.where(sign_mask > 0.5, a_pos_k,  a_neg_k)
        b_mix  = torch.where(sign_mask > 0.5, b_pos_k,  b_neg_k)

        cat_m = torch.distributions.Categorical(probs=pi_mix)
        m_idx = cat_m.sample()  # [S,B,H]

        idx_m = m_idx.unsqueeze(-1)  # [S,B,H,1]
        a_g = torch.gather(a_mix, 3, idx_m).squeeze(-1)  # [S,B,H]
        b_g = torch.gather(b_mix, 3, idx_m).squeeze(-1)  # [S,B,H]

        # --- post-hoc dispersion calibration (VAL-fitted) ---
        # Gamma: mean=alpha*beta, var=alpha*beta^2
        # Keep mean fixed, scale variance by v: alpha' = alpha / v, beta' = beta * v
        if dispersion_scale is not None:
            v = torch.as_tensor(dispersion_scale, device=a_g.device, dtype=a_g.dtype)
            a_g = (a_g / v).clamp_min(self.head.alpha_floor)
            b_g = (b_g * v).clamp_min(self.head.beta_floor)

        gam = torch.distributions.Gamma(concentration=a_g, rate=1.0 / (b_g + 1e-12)).sample()  # [S,B,H]
        return sign_factor * gam

    def sample_independent(self, x: torch.Tensor, n_samples: int, dispersion_scale: float | None = None) -> torch.Tensor:
      """
      Sampling with independent latent k per horizon.
      Keeps per-horizon marginals, removes cross-horizon dependence.
      Returns: [S,B,H]
      """
      pi_root, s, pi_pos, a_pos, b_pos, pi_neg, a_neg, b_neg = self.forward_params(x)

      B, K = pi_root.shape
      H = self.horizon
      M = self.m_mag
      S = n_samples

      cat_k = torch.distributions.Categorical(probs=pi_root)

      out = torch.empty((S, B, H), device=x.device, dtype=torch.float32)

      for h in range(H):
          # sample k independently for this horizon
          k_idx = cat_k.sample((S,))  # [S,B]

          # gather s for (B,K) at horizon h -> [S,B]
          s_bk = s[:, :, h]  # [B,K]
          s_exp = s_bk.unsqueeze(0).expand(S, B, K)
          idx = k_idx.unsqueeze(-1)  # [S,B,1]
          s_k = torch.gather(s_exp, 2, idx).squeeze(-1)  # [S,B]

          # sample sign
          sign_draw = torch.distributions.Bernoulli(probs=s_k).sample()  # [S,B]
          sign_factor = 2.0 * sign_draw - 1.0

          # gather mixture params for this horizon: [B,K,M] -> [S,B,M]
          def gather_bkm(t_bkhm: torch.Tensor) -> torch.Tensor:
              t = t_bkhm[:, :, h, :]  # [B,K,M]
              t = t.unsqueeze(0).expand(S, B, K, M)
              idx4 = k_idx.unsqueeze(-1).unsqueeze(-1).expand(S, B, 1, M)
              return torch.gather(t, 2, idx4).squeeze(2)  # [S,B,M]

          pi_pos_k = gather_bkm(pi_pos)
          pi_neg_k = gather_bkm(pi_neg)
          a_pos_k  = gather_bkm(a_pos)
          b_pos_k  = gather_bkm(b_pos)
          a_neg_k  = gather_bkm(a_neg)
          b_neg_k  = gather_bkm(b_neg)

          sign_mask = sign_draw.unsqueeze(-1)  # [S,B,1]
          pi_mix = torch.where(sign_mask > 0.5, pi_pos_k, pi_neg_k)  # [S,B,M]
          a_mix  = torch.where(sign_mask > 0.5, a_pos_k,  a_neg_k)
          b_mix  = torch.where(sign_mask > 0.5, b_pos_k,  b_neg_k)

          cat_m = torch.distributions.Categorical(probs=pi_mix)
          m_idx = cat_m.sample()  # [S,B]

          idx_m = m_idx.unsqueeze(-1)  # [S,B,1]
          a_g = torch.gather(a_mix, 2, idx_m).squeeze(-1)  # [S,B]
          b_g = torch.gather(b_mix, 2, idx_m).squeeze(-1)  # [S,B]

          if dispersion_scale is not None:
            v = torch.as_tensor(dispersion_scale, device=a_g.device, dtype=a_g.dtype)
            a_g = (a_g / v).clamp_min(self.head.alpha_floor)
            b_g = (b_g * v).clamp_min(self.head.beta_floor)

          gam = torch.distributions.Gamma(concentration=a_g, rate=1.0 / (b_g + 1e-12)).sample()  # [S,B]
          out[:, :, h] = sign_factor * gam

      return out

    def covariance_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """
        Exact Cov[y|x] for mixture-of-products:
          Cov = E[Var(y|k,x)] + Var(E[y|k,x])
        Returns [B,H,H].
        """
        pi_root, s, pi_pos, a_pos, b_pos, pi_neg, a_neg, b_neg = self.forward_params(x)

        # Means per leaf
        m1_pos = a_pos * b_pos
        m1_neg = a_neg * b_neg

        # Second moments per leaf: E[u^2] = Var + mean^2, Var(Gamma)=alpha*beta^2
        m2_pos = a_pos * (b_pos ** 2) + (m1_pos ** 2)
        m2_neg = a_neg * (b_neg ** 2) + (m1_neg ** 2)

        # Mixture moments per (B,K,H)
        eu_pos  = (pi_pos * m1_pos).sum(dim=3)  # E[u]   pos
        eu_neg  = (pi_neg * m1_neg).sum(dim=3)  # E[u]   neg
        eu2_pos = (pi_pos * m2_pos).sum(dim=3)  # E[u^2] pos
        eu2_neg = (pi_neg * m2_neg).sum(dim=3)  # E[u^2] neg

        # E[y|k] and E[y^2|k] (since sign^2=1 => y^2=u^2)
        mu_kh  = s * eu_pos - (1.0 - s) * eu_neg          # [B,K,H]
        ey2_kh = s * eu2_pos + (1.0 - s) * eu2_neg        # [B,K,H]

        # Overall mean
        mu = (pi_root.unsqueeze(-1) * mu_kh).sum(dim=1)   # [B,H]

        # Var(E[y|k]) term via E[yy^T] from means
        Eyy = torch.einsum("bk,bkh,bkj->bhj", pi_root, mu_kh, mu_kh)  # [B,H,H]
        cov = Eyy - torch.einsum("bh,bj->bhj", mu, mu)

        # Add E[Var(y|k)] on diagonal: E[y^2|k] - (E[y|k])^2
        diag_add = (pi_root.unsqueeze(-1) * (ey2_kh - mu_kh.pow(2))).sum(dim=1)  # [B,H]
        idx = torch.arange(self.horizon, device=x.device)
        cov[:, idx, idx] += diag_add

        return cov

# ---------------- Figures ----------------
import matplotlib.patches as patches

def plot_framework_diagram_once():
    global _FIG_R24_DONE
    if _FIG_R24_DONE:
        return
    _FIG_R24_DONE = True

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")

    boxes = [
        (0.02, 0.35, 0.18, 0.30, "Input\n(lookback window)"),
        (0.25, 0.35, 0.18, 0.30, "CNN1D + skip\nfeature extractor"),
        (0.48, 0.35, 0.18, 0.30, "GRU\nsequence encoder"),
        (0.71, 0.15, 0.27, 0.70, "JOINT conditional SPN head\npi_k, s_{k,h}\nGamma mixtures per sign\nProduct over horizons"),
    ]

    for x, y, w, h, t in boxes:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02")
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=9)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.5))

    arrow(0.20, 0.50, 0.25, 0.50)
    arrow(0.43, 0.50, 0.48, 0.50)
    arrow(0.66, 0.50, 0.71, 0.50)

    ax.text(0.845, 0.85, "Outputs:\n- predictive samples\n- quantiles/intervals\n- log-likelihood",
            ha="center", va="center", fontsize=9)

    save_fig(fig, "fig_r24_framework")


def plot_mag_gof(y_true: np.ndarray, samples: np.ndarray, fig_name: str = "fig_mag_gof"):
    # y_true: [N,H], samples: [S,N,H]
    mag_emp = np.abs(y_true).reshape(-1)
    mag_smp = np.abs(samples).reshape(-1)

    mag_emp = mag_emp[np.isfinite(mag_emp)]
    mag_smp = mag_smp[np.isfinite(mag_smp)]

    x_emp = np.sort(mag_emp)
    y_emp = np.linspace(0, 1, len(x_emp), endpoint=True)

    x_smp = np.sort(mag_smp)
    y_smp = np.linspace(0, 1, len(x_smp), endpoint=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_emp, y_emp, label="Empirical |y|", linewidth=2.0)
    ax.plot(x_smp, y_smp, label="Model samples |y|", linewidth=2.0)
    ax.set_xlabel("|y| (°C)")
    ax.set_ylabel("CDF")
    ax.set_title("Magnitude goodness-of-fit for |y|")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_fig(fig, fig_name)

    qs = [0.5, 0.75, 0.9, 0.95, 0.99]
    rows = []
    for q in qs:
        qe = float(np.quantile(mag_emp, q))
        qsmp = float(np.quantile(mag_smp, q))
        rows.append({"q": q, "emp": qe, "sample": qsmp, "abs_diff": float(abs(qe - qsmp))})
    save_table(pd.DataFrame(rows), name="mag_gof_quantiles", float_fmt="%.6f")

# ---------------- Training ----------------

class WarmupCosine:
    def __init__(self, warm: int, total: int):
        self.warm = warm
        self.total = total

    def __call__(self, ep: int) -> float:
        if ep < self.warm:
            return float(ep + 1) / float(max(1, self.warm))
        prog = (ep - self.warm) / float(max(1, self.total - self.warm))
        return 0.5 * (1.0 + math.cos(math.pi * prog))


# Training loop with validation monitoring and early stopping for robust model selection
def train_model(
    model: GlobalJointConditionalSPN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
    warmup: int,
    patience: int,
    l2_log_param: float,
    entropy_weight: float,
    grad_clip: float | None = 1.0,
) -> GlobalJointConditionalSPN:
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=WarmupCosine(warmup, epochs))

    best = float("inf")
    best_state = None
    pat = patience

    for ep in range(epochs):
        model.train()
        train_sum = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            loss = model.train_objective(xb, yb, l2_log_param=l2_log_param, entropy_weight=entropy_weight)
            opt.zero_grad()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            train_sum += loss.item() * xb.size(0)

        sch.step()

        model.eval()
        val_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                vloss = model.train_objective(xb, yb, l2_log_param=l2_log_param, entropy_weight=entropy_weight)
                val_sum += vloss.item() * xb.size(0)
                n_val += xb.size(0)

        train_obj = train_sum / max(1, len(train_loader.dataset))
        val_obj = val_sum / max(1, n_val)

        print(
            f"Epoch {ep+1:3d}/{epochs} | Train Obj: {train_obj:.4f} | Val Obj: {val_obj:.4f} | LR: {sch.get_last_lr()[0]:.6f}"
        )

        if val_obj < best - 1e-4:
            best = val_obj
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            pat = patience
        else:
            pat -= 1
            if pat <= 0:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model

# ---------------- Transformer baseline ----------------
class TinyTransformerForecaster(nn.Module):
    def __init__(self, in_dim: int, H: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dim_ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.H = H
        self.inp = nn.Linear(in_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,W,4]
        h = self.inp(x)
        h = self.enc(h)
        out = self.head(h[:, -1, :])
        return out  # [B,H]

def train_tiny_transformer(train_loader, val_loader, H: int, device: str,
                           max_epochs: int = 50, lr: float = 1e-3, patience: int = 8, seed: int = 0):
    torch.manual_seed(seed)
    model = TinyTransformerForecaster(in_dim=4, H=H).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best = None
    best_state = None
    bad = 0

    def _rmse(pred, y):
        return float(torch.sqrt(torch.mean((pred - y) ** 2)).detach().cpu())

    for _ep in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = torch.mean((pred - yb) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        vals = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                vals.append(_rmse(pred, yb))
        v = float(np.mean(vals))

        if (best is None) or (v < best):
            best = v
            best_state = {k: vv.cpu().clone() for k, vv in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ---------------- Metrics ----------------

def rmse_all(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae_all(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def _weighted_corr(x, y, w, eps=1e-12):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    ws = w.sum()
    if ws <= eps:
        return float("nan")
    w = w / (ws + eps)
    mx = (w * x).sum()
    my = (w * y).sum()
    cov = (w * (x - mx) * (y - my)).sum()
    vx = (w * (x - mx) ** 2).sum()
    vy = (w * (y - my) ** 2).sum()
    return float(cov / (np.sqrt(vx * vy) + eps))

def _weighted_mi_sign_mag(sign01, mag, w, n_bins=10, eps=1e-12):
    sign01 = np.asarray(sign01, dtype=np.int64)
    mag = np.asarray(mag, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    ws = w.sum()
    if ws <= eps:
        return float("nan")

    # bin by quantiles (robust)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(mag, qs)
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    b = np.digitize(mag, edges[1:-1], right=True)  # 0..n_bins-1

    joint = np.zeros((2, n_bins), dtype=np.float64)
    for s in (0, 1):
        for j in range(n_bins):
            joint[s, j] = w[(sign01 == s) & (b == j)].sum()

    joint = joint / (joint.sum() + eps)
    ps = joint.sum(axis=1, keepdims=True)
    pb = joint.sum(axis=0, keepdims=True)

    mi_nats = (joint * np.log((joint + eps) / (ps * pb + eps))).sum()
    return float(mi_nats / np.log(2.0))  # bits

# Empirical interval coverage from Monte Carlo samples at the requested quantile levels
def coverage_from_samples_1d(y_true: np.ndarray, samples: np.ndarray, q_low: float, q_high: float) -> float:
    lo = np.quantile(samples, q_low, axis=0)
    hi = np.quantile(samples, q_high, axis=0)
    return float(np.mean((y_true >= lo) & (y_true <= hi)))

def _mean_abs_pairwise_sorted(samples_sorted: np.ndarray) -> np.ndarray:
    S = samples_sorted.shape[0]
    k = np.arange(1, S + 1, dtype=np.float64).reshape(-1, 1)
    w = (2.0 * k - S - 1.0)
    sum_abs = 2.0 * np.sum(w * samples_sorted, axis=0)
    return sum_abs / (S * S)

# CRPS estimate from samples as a proper scoring rule for distributional forecasts
def crps_from_samples_1d(y_true: np.ndarray, samples: np.ndarray) -> float:
    term1 = np.mean(np.abs(samples - y_true.reshape(1, -1)), axis=0)
    s_sorted = np.sort(samples, axis=0)
    e_abs_xx = _mean_abs_pairwise_sorted(s_sorted)
    crps = term1 - 0.5 * e_abs_xx
    return float(np.mean(crps))

def cia95_from_cov95(cov95_percent: float) -> float:
    cov = cov95_percent / 100.0
    return 100.0 * (1.0 - abs(cov - 0.95) / 0.95)

def energy_score(y_true: np.ndarray, samples: np.ndarray, seed: int = 0) -> float:
    """
    Multivariate probabilistic score for JOINT forecasts.
    y_true:  [N,H]
    samples: [S,N,H]
    """
    rng = np.random.default_rng(seed)
    S, N, H = samples.shape

    y = y_true[None, :, :]  # [1,N,H]
    d1 = np.linalg.norm(samples - y, axis=2)  # [S,N]
    term1 = d1.mean()

    # approximate E||X-X'|| with random pairing (O(S))
    idx = rng.integers(0, S, size=S)
    d2 = np.linalg.norm(samples - samples[idx], axis=2)  # [S,N]
    term2 = 0.5 * d2.mean()

    return float(term1 - term2)

def bootstrap_ci(metric_fn, y_true: np.ndarray, y_pred: np.ndarray,
                 B: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """
    Resamples on window dimension N (rows), keeps horizon dimension.
    metric_fn should accept (y_true, y_pred) arrays with shape [N,H].
    """
    rng = np.random.default_rng(seed)
    N = y_true.shape[0]
    vals = np.empty(B, dtype=np.float64)

    for b in range(B):
        idx = rng.integers(0, N, size=N)
        vals[b] = metric_fn(y_true[idx], y_pred[idx])

    lo = float(np.quantile(vals, alpha / 2))
    hi = float(np.quantile(vals, 1 - alpha / 2))
    return lo, hi

def _mbb_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    if block_len <= 1:
        return rng.integers(0, n, size=n)
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n - block_len + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_len) for s in starts])[:n]
    return idx

def bootstrap_ci_mbb(metric_fn, y_true: np.ndarray, y_pred: np.ndarray,
                     block_len: int, B: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    N = y_true.shape[0]
    vals = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = _mbb_indices(N, block_len, rng)
        vals[b] = metric_fn(y_true[idx], y_pred[idx])
    lo = float(np.quantile(vals, alpha / 2))
    hi = float(np.quantile(vals, 1 - alpha / 2))
    return lo, hi

def dm_test_newey_west(d: np.ndarray, lag: int = 5) -> tuple[float, float]:
    """
    d_t = loss_model_t - loss_base_t
    Newey-West HAC variance, t-approx p-value (two-sided).
    """
    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d)]
    T = d.size
    if T < 20:
        return float("nan"), float("nan")

    mu = d.mean()
    x = d - mu

    gamma0 = np.mean(x * x)
    var = gamma0
    L = min(lag, T - 1)
    for l in range(1, L + 1):
        cov = np.mean(x[l:] * x[:-l])
        w = 1.0 - l / (L + 1.0)
        var += 2.0 * w * cov

    dm = mu / np.sqrt(var / T + 1e-12)
    p = 2.0 * (1.0 - student_t.cdf(abs(dm), df=max(T - 1, 1)))
    return float(dm), float(p)

def nw_lag_rule(T: int, cap: int = 24) -> int:
    # standard rule: ~ 1.5 * T^(1/3)
    if T <= 1:
        return 1
    L = int(round(1.5 * (T ** (1.0 / 3.0))))
    return int(max(1, min(cap, T - 1, L)))

def fisher_pvalue(pvals: list[float]) -> float:
    p = np.asarray([x for x in pvals if np.isfinite(x) and (x > 0.0) and (x <= 1.0)], dtype=np.float64)
    if p.size == 0:
        return float("nan")
    stat = -2.0 * np.sum(np.log(p))
    return float(1.0 - chi2.cdf(stat, df=2 * p.size))


def pit_from_samples(y_true: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """
    PIT values using samples.
    y_true:  [N,H]
    samples: [S,N,H]
    returns: [N,H] values in [0,1]
    """
    return np.mean(samples <= y_true[None, :, :], axis=0)

def pit_ks(pit: np.ndarray) -> float:
    """
    Kolmogorov-Smirnov distance to Uniform(0,1) for PIT values (flattened).
    """
    u = np.sort(pit.reshape(-1))
    n = u.size
    if n == 0:
        return float("nan")
    ecdf = np.arange(1, n + 1) / n
    return float(np.max(np.abs(ecdf - u)))

def interval_width_from_samples(samples: np.ndarray, q_low: float, q_high: float) -> float:
    """
    Mean interval width across all N,H.
    samples: [S,N,H]
    """
    lo = np.quantile(samples, q_low, axis=0)
    hi = np.quantile(samples, q_high, axis=0)
    return float(np.mean(hi - lo))

def calibration_abs_error(y_true: np.ndarray, samples: np.ndarray, levels=(0.5, 0.9, 0.95)) -> float:
    """
    Average absolute calibration error across nominal central interval levels.
    """
    errs = []
    for lvl in levels:
        q = (1 - lvl) / 2
        cov = coverage_from_samples_1d(y_true.reshape(-1), samples.reshape(samples.shape[0], -1), q, 1 - q)
        errs.append(abs(cov - lvl))
    return float(np.mean(errs))

def cov95_joint(y_true: np.ndarray, samples: np.ndarray) -> float:
    """
    y_true:  [N,H]
    samples: [S,N,H]
    """
    lo = np.quantile(samples, 0.025, axis=0)
    hi = np.quantile(samples, 0.975, axis=0)
    return float(np.mean((y_true >= lo) & (y_true <= hi)))

@torch.no_grad()
def fit_dispersion_scale_val(
    model,
    Xva_t: torch.Tensor,
    yva: np.ndarray,
    device: str,
    n_samples: int,
    seed: int,
    max_val_windows: int = 3000,
) -> tuple[float, float]:
    """
    Return: (v_best, cov95_val_best)
    """
    rng = np.random.default_rng(seed)
    N = yva.shape[0]
    if N > max_val_windows:
        idx = rng.choice(N, size=max_val_windows, replace=False)
        Xc = Xva_t[idx]
        yc = yva[idx]
    else:
        Xc = Xva_t
        yc = yva

    grid = [0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.9, 2.2, 2.6, 3.0]

    best_v = 1.0
    best_err = 1e9
    best_cov = np.nan

    model.eval()
    for v in grid:
        smp = model.sample(Xc.to(device), n_samples=n_samples, dispersion_scale=float(v)).cpu().numpy()
        cov = cov95_joint(yc, smp)
        err = abs(cov - 0.95)
        if err < best_err:
            best_err = err
            best_v = float(v)
            best_cov = float(cov)

    return best_v, best_cov

@torch.no_grad()
def avg_test_nll_per_horizon(model: GlobalJointConditionalSPN, loader: DataLoader, device: str, horizon: int) -> float:
    model.eval()
    total = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logp = model.log_prob(xb, yb)   # joint logp
        total += float((-logp).sum().item())
        n += xb.size(0)
    # report per-horizon average NLL
    return (total / max(1, n)) / float(horizon)


@torch.no_grad()
def sign_mag_dependence_table(model, X_np, y_np, device, n_bins=10, max_rows=None):
    """
    Computes dependence between sign(y_h) and |y_h| inside each root component k,
    using posterior responsibilities r_{b,k} = p(k | x_b, y_b) from the SPN.
    Returns df with columns: h, k, N_eff, corr(sign,|y|), MI_bits
    """
    model.eval()
    X = torch.tensor(X_np).to(device)
    y = torch.tensor(y_np).to(device)

    # optional subsample for speed
    if (max_rows is not None) and (X.shape[0] > max_rows):
        idx = torch.randperm(X.shape[0], device=device)[:max_rows]
        X = X[idx]
        y = y[idx]

    r = model.posterior_k(X, y).detach().cpu().numpy()   # [B,K]
    yv = y.detach().cpu().numpy()                        # [B,H]

    B, K = r.shape
    H = yv.shape[1]

    rows = []
    for h in range(H):
        sign01 = (yv[:, h] >= 0).astype(np.int64)
        mag    = np.abs(yv[:, h]).astype(np.float64)

        for k in range(K):
            w = r[:, k].astype(np.float64)

            ws = w.sum()
            w2 = (w * w).sum()
            N_eff = (ws * ws) / (w2 + 1e-12)

            corr = _weighted_corr(sign01, mag, w)
            mi   = _weighted_mi_sign_mag(sign01, mag, w, n_bins=n_bins)

            rows.append({
                "h": h + 1,
                "k": k,
                "N_eff": float(N_eff),
                "corr(sign,|y|)": float(corr),
                "MI_bits": float(mi),
            })

    return pd.DataFrame(rows)

    # ---- call (right after you have model, Xte, yte) ----
    df_dep = sign_mag_dependence_table(model, Xte, yte, device=device, n_bins=10, max_rows=None)
    save_table(df_dep, name=f"sign_mag_dependence_{tag}", float_fmt="%.6f")


# ---------------- CLI ----------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Joint multi-horizon conditional SPN (Colab-friendly)")
    p.add_argument("--csv_path", type=str, default=CSV_PATH)
    p.add_argument("--n_samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--seed_split", type=int, default=SEED)
    p.add_argument("--seed_model", type=int, default=SEED)
    p.add_argument("--k_root", type=int, default=K_ROOT)
    p.add_argument("--m_mag", type=int, default=M_MAG)
    p.add_argument("--hidden", type=int, default=HIDDEN)
    p.add_argument("--dropout", type=float, default=DROPOUT)
    p.add_argument("--window", type=int, default=WINDOW)
    p.add_argument("--seeds", type=str, default="42,43,44")  #"42,43,44"
    p.add_argument("--n_eval_sets", type=int, default=5)
    p.add_argument("--eval_seed_base", type=int, default=123)
    p.add_argument("--split_seed_base", type=int, default=1000)
    return p.parse_args(args=([] if argv is None else argv))


def run_once(
      args: argparse.Namespace,
      seed_split: int,
      seed_model: int,
      device: str,
      eval_set_id: int,
      eval_seed: int,
      eval_cities: list[str],
) -> dict:
    """
    One full experiment run for a given (seed_split, seed_model).
    Returns a dict with scalar metrics so main() can summarize across seeds.
    """

    # ---- seeds & determinism
    set_seed(seed_model, deterministic=True)

    H = int(args.horizon)
    print("\n" + "=" * 70)
    print(f"RUN | seed_split: {seed_split} | seed_model: {seed_model} | device: {device} | H={H}")
    print("=" * 70)
    run_wall_t0 = time.time()
    plot_framework_diagram_once()

    # ---------------- Load + preprocess ----------------
    df = read_csv(args.csv_path)
    df = df[df["Year"] >= 1900].copy()
    y0, y1 = probe_span(df)
    print(f"[DIAG] Available years: {y0}..{y1}, cities: {df['City'].nunique()}")

    # build once, then we can drop raw daily df to save RAM
    city_region = build_city_region_map(df)

    monthly = aggregate_monthly(df, y0, y1)
    monthly = exclude_incomplete_years(monthly, min_months=MIN_MONTHS_PER_YEAR)
    monthly = reindex_month_grid(monthly, y0, y1)

    train_end, val_end = split_years(y0, y1)
    print(f"[DIAG] Time splits -> TRAIN<= {train_end}, VAL: {train_end+1}..{val_end}, TEST: {val_end+1}..{y1}")
    print(f"[DIAG] Horizon H={H}")

    # anomalies with train-only climatology
    monthly = build_anomalies_fixed_climatology(monthly, train_end=train_end)

    # raw daily df no longer needed (we already extracted city_region)
    del df
    gc.collect()

    # dataset examples (generate once)
    example_cities = []
    for c in eval_cities[:3]:
        if (monthly["City"] == c).any():
            example_cities.append(c)
    if len(example_cities) < 3:
        # fallback
        example_cities = list(monthly["City"].drop_duplicates().head(3).to_list())

    plot_dataset_examples_once(monthly, example_cities, train_end=train_end, val_end=val_end)


    # ----- Choose eval/test cities (fixed or stratified) -----
    print(f"[DIAG] Eval set id: {eval_set_id} | eval_seed: {eval_seed}")
    print("[DIAG] Eval (TEST) cities:", eval_cities)
    eval_set_norm = {norm_city_name(c) for c in eval_cities}

    # geographic split: train/val/test cities disjoint
    all_cities = sorted(monthly["City"].unique())
    train_cities_norm, val_cities_norm, test_cities_norm = split_cities_geographic(
        all_cities, eval_set_norm, seed=seed_split
    )

    print(f"[DIAG] Geographic split -> TRAIN cities: {len(train_cities_norm)}, VAL cities: {len(val_cities_norm)}, TEST cities: {len(test_cities_norm)}")
    if len(test_cities_norm) == 0:
        raise RuntimeError("No test cities matched EVAL_CITIES in the dataset (check city names).")

    # ---- artifacts filenames (so multiple seeds do not overwrite)
    tag = f"set{eval_set_id}_seedS{seed_split}_seedM{seed_model}_H{H}"
    run_manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "csv_path": args.csv_path,
        "years": {"min": y0, "max": y1, "train_end": train_end, "val_end": val_end},
        "seeds": {"split": seed_split, "model": seed_model},
        "hyperparams": {
            "window": args.window,
            "horizon": H,
            "k_root": args.k_root,
            "m_mag": args.m_mag,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "lr": LR,
            "batch": BATCH,
            "epochs": EPOCHS,
            "warmup": WARMUP,
            "patience": PATIENCE,
            "max_pchip_gap": MAX_PCHIP_GAP,
            "min_months_per_year": MIN_MONTHS_PER_YEAR,
        },
        "eval_cities": eval_cities,
        "splits": {
            "train_cities_norm": sorted(list(train_cities_norm)),
            "val_cities_norm": sorted(list(val_cities_norm)),
            "test_cities_norm": sorted(list(test_cities_norm)),
        },
        "versions": {"torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "eval_set": {"id": eval_set_id, "seed": eval_seed},
        "pipeline": {
            "aggregation": "Daily AvgTemperature -> monthly mean per City-Year-Month.",
            "climatology": f"Per-city month-of-year mean computed using only years <= {train_end}.",
            "anomaly": "Anomaly_raw = AvgTemperature - climatology(month).",
            "targets": "Targets are raw anomalies; any window with missing targets is discarded (no target imputation).",
            "pchip": f"Input-only causal PCHIP inside the past window; only internal gaps <= {MAX_PCHIP_GAP} months are imputed.",
            "scaling": "StandardScaler is fit on TRAIN windows only for features [anomaly, delta] and applied to VAL/TEST."
        }
    }

    with open(f"run_manifest_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    # joint windowing
    X_split, Y_split, c_split, y1year_split, city2id = make_windows_global_joint(
        monthly,
        window=args.window,
        horizon=H,
        train_end=train_end,
        val_end=val_end,
        train_cities_norm=train_cities_norm,
        val_cities_norm=val_cities_norm,
        test_cities_norm=test_cities_norm,
        max_pchip_gap=MAX_PCHIP_GAP,
    )

    for kname in ["train", "val", "test"]:
        print(f"[DIAG] {kname.upper()} samples: {len(X_split[kname])}")

    if len(X_split["train"]) == 0 or len(X_split["val"]) == 0 or len(X_split["test"]) == 0:
        raise RuntimeError("Empty split. Try smaller --horizon (e.g., 3 or 6) or check dataset coverage.")


    # ---------------- Scale inputs (fit on TRAIN only) ----------------
    scaler = StandardScaler()
    flat_tr = X_split["train"][..., [0, 3]].reshape(-1, 2)  # anomaly + delta
    scaler.fit(flat_tr)

    np.savez(f"scaler_anom_delta_{tag}.npz", mean=scaler.mean_, scale=scaler.scale_)

    def apply_scale(X: np.ndarray) -> np.ndarray:
        Xc = X.copy()
        flat = Xc[..., [0, 3]].reshape(-1, 2)
        flat = scaler.transform(flat)
        Xc[..., [0, 3]] = flat.reshape(Xc.shape[0], Xc.shape[1], 2)
        return Xc

    Xtr_raw = X_split["train"]
    Xva_raw = X_split["val"]
    Xte_raw = X_split["test"]

    Xtr = apply_scale(Xtr_raw); ytr = Y_split["train"]
    Xva = apply_scale(Xva_raw); yva = Y_split["val"]
    Xte = apply_scale(Xte_raw); yte = Y_split["test"]

    cte = c_split["test"]
    yte_year1 = y1year_split["test"]

    # keep only what we need from raw (for persistence baseline)
    xte_last_anom = Xte_raw[:, -1, 0:1].copy()
    del Xtr_raw, Xva_raw, Xte_raw
    del X_split, Y_split, c_split, y1year_split
    gc.collect()

    Xtr_t = torch.from_numpy(Xtr)
    ytr_t = torch.from_numpy(ytr)
    Xva_t = torch.from_numpy(Xva)
    yva_t = torch.from_numpy(yva)
    Xte_t_cpu = torch.from_numpy(Xte)
    yte_t_cpu = torch.from_numpy(yte)

    train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader   = DataLoader(TensorDataset(Xva_t, yva_t), batch_size=BATCH, shuffle=False, num_workers=0)
    test_loader  = DataLoader(TensorDataset(Xte_t_cpu, yte_t_cpu), batch_size=BATCH, shuffle=False, num_workers=0)

    ###
    def train_eval_variant(variant_name: str, shared_mag: bool):
      set_seed(seed_model, deterministic=True)

      model_v = GlobalJointConditionalSPN(
          in_ch=4,
          hidden=args.hidden,
          k_root=args.k_root,
          horizon=H,
          m_mag=args.m_mag,
          dropout=args.dropout,
          alpha_floor=ALPHA_FLOOR,
          beta_floor=BETA_FLOOR,
          shared_magnitude=shared_mag,
      )

      t0 = time.time()
      model_v = train_model(
          model=model_v,
          train_loader=train_loader,
          val_loader=val_loader,
          device=device,
          epochs=EPOCHS,
          lr=LR,
          warmup=WARMUP,
          patience=PATIENCE,
          l2_log_param=L2_LOG_PARAM,
          entropy_weight=ENTROPY_WEIGHT,
          grad_clip=1.0,
      )
      t1 = time.time()

      train_seconds = float(t1 - t0)
      n_params = int(sum(p.numel() for p in model_v.parameters()))

      model_v.eval()
      with torch.no_grad():
          Xte_t = torch.tensor(Xte).to(device)
          Xva_t = torch.tensor(Xva).to(device)

          v_disp, _ = fit_dispersion_scale_val(
              model_v, Xva_t, yva, device=device,
              n_samples=min(400, args.n_samples),
              seed=seed_model,
          )

          y_pred_v = model_v.predictive_mean(Xte_t).cpu().numpy()
          samples_v = model_v.sample(Xte_t, n_samples=args.n_samples, dispersion_scale=v_disp).cpu().numpy()

      rmse_v = rmse_all(yte, y_pred_v)
      mae_v  = mae_all(yte, y_pred_v)
      nll_ph_v = float(avg_test_nll_per_horizon(model_v, test_loader, device=device, horizon=H))
      crps_h_v = [crps_from_samples_1d(yte[:, h], samples_v[:, :, h]) for h in range(H)]
      crps_avg_v = float(np.mean(crps_h_v))

      return {
          "Variant": variant_name,
          "RMSE": float(rmse_v),
          "MAE": float(mae_v),
          "CRPS": float(crps_avg_v),
          "NLL_per_h": float(nll_ph_v),
          "samples": samples_v,
          "y_pred": y_pred_v,
          "model": model_v,
          "v_disp": float(v_disp),
          "train_seconds": train_seconds,
          "n_params": n_params,
      }


    # ---------------- Train + Predict ----------------
    res_ours = train_eval_variant("Sign-specific magnitude (ours)", shared_mag=False)
    model   = res_ours["model"]
    y_pred  = res_ours["y_pred"]
    samples = res_ours["samples"]
    v_disp  = res_ours["v_disp"]


    # ---------------- SPN diagnostics ----------------
    # 1) GOF magnitude |y|
    # uncomment next line for mag gof plot
    # plot_mag_gof(yte, samples, fig_name=f"fig_mag_gof_{tag}")

    # 2) Dependence sign - |y| on (h,k) using responsibilities p(k|x,y)
    df_dep = sign_mag_dependence_table(
        model=model,
        X_np=Xte,
        y_np=yte,
        device=device,
        n_bins=10,
        max_rows=None,
    )
    save_table(df_dep, name=f"sign_mag_dependence_{tag}", float_fmt="%.6f")


    run_manifest["compute"] = {
        "train_seconds": float(res_ours["train_seconds"]),
        "n_params": int(res_ours["n_params"]),
        "device": device,
        "cuda_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
    }

    # ===== Extra metrics that were previously undefined in return() =====
    with torch.no_grad():
        Xte_t = torch.tensor(Xte).to(device)

        # independent-latent samples (same marginals, no cross-horizon dependence)
        samples_ind = model.sample_independent(Xte_t, n_samples=args.n_samples, dispersion_scale=v_disp).cpu().numpy()

    # Energy score (joint vs independent)
    es_joint = energy_score(yte, samples, seed=seed_model)
    es_ind   = energy_score(yte, samples_ind, seed=seed_model)

    # PIT + KS distance to Uniform(0,1)
    pit_vals   = pit_from_samples(yte, samples)   # [N,H]
    pit_ks_val = pit_ks(pit_vals)

    # Mean 95% interval width
    iw95 = interval_width_from_samples(samples, 0.025, 0.975)

    # Average absolute calibration error over levels
    cal_ae = calibration_abs_error(yte, samples, levels=(0.5, 0.9, 0.95))

    # Off-diagonal correlation magnitude (batched, safe)
    def corr_offdiag_batched(model, X_np, device, batch_size=2048):
        model.eval()
        H = model.horizon
        mask = ~np.eye(H, dtype=bool)
        vals = []

        with torch.no_grad():
            for i in range(0, X_np.shape[0], batch_size):
                xb = torch.tensor(X_np[i:i+batch_size]).to(device)
                cov = model.covariance_matrix(xb).detach().cpu().numpy()  # [B,H,H]

                # corr = cov / sqrt(var_i var_j)
                var = np.diagonal(cov, axis1=1, axis2=2)                  # [B,H]
                denom = np.sqrt(np.clip(var, 1e-12, None))
                corr = cov / (denom[:, :, None] * denom[:, None, :] + 1e-12)

                vals.append(np.mean(np.abs(corr[:, mask])))

        return float(np.mean(vals)) if len(vals) else float("nan")

    corr_off = corr_offdiag_batched(model, Xte, device=device, batch_size=2048)


    # ---------------- Ablation ----------------
    if (eval_set_id == 0) and (seed_model == 42):
        res_shared = train_eval_variant("Shared magnitude", shared_mag=True)

        df_ab = pd.DataFrame([
            {k: res_shared[k] for k in ["Variant","RMSE","MAE","CRPS","NLL_per_h"]},
            {k: res_ours[k]   for k in ["Variant","RMSE","MAE","CRPS","NLL_per_h"]},
        ])
        save_table(df_ab, name="signed_gamma_ablation", float_fmt="%.4f")


    # ---------------- Metrics ----------------
    rmse_g = rmse_all(yte, y_pred)
    mae_g  = mae_all(yte, y_pred)
    nll_ph = float(avg_test_nll_per_horizon(model, test_loader, device=device, horizon=H))

    # JOINT score
    es = energy_score(yte, samples, seed=seed_model)

    # confidence intervals (bootstrap over windows)
    rmse_ci = bootstrap_ci(rmse_all, yte, y_pred, B=1000, seed=seed_model)
    mae_ci  = bootstrap_ci(mae_all,  yte, y_pred, B=1000, seed=seed_model)

    # per-horizon metrics
    rmse_h, mae_h, crps_h, cov95_h = [], [], [], []
    for h in range(H):
        rmse_h.append(rmse_all(yte[:, h], y_pred[:, h]))
        mae_h.append(mae_all(yte[:, h], y_pred[:, h]))
        crps_h.append(crps_from_samples_1d(yte[:, h], samples[:, :, h]))
        cov95_h.append(100.0 * coverage_from_samples_1d(yte[:, h], samples[:, :, h], 0.025, 0.975))

    rmse_avg  = float(np.mean(rmse_h))
    mae_avg   = float(np.mean(mae_h))
    crps_avg  = float(np.mean(crps_h))
    cov95_avg = float(np.mean(cov95_h))

    # baselines
    y_pred_clim0 = np.zeros_like(yte, dtype=np.float32)
    y_pred_persist = np.repeat(xte_last_anom, repeats=H, axis=1).astype(np.float32)


    # ----- Statistical tests: DM + Moving-Block Bootstrap (seasonal blocks) -----
    loss_model = np.mean((y_pred - yte) ** 2, axis=1)          # [N]
    loss_pers  = np.mean((y_pred_persist - yte) ** 2, axis=1)  # [N]
    d_mp = loss_model - loss_pers                              # DM diff

    dm_stat, dm_p = dm_test_newey_west(d_mp, lag=5)
    print(f"[DM] Model vs Persist | DM={dm_stat:.3f} | p={dm_p:.4g} | mean(d)={d_mp.mean():.6f} (negative=better)")

    for L in [6, 12, 24]:
        lo, hi = bootstrap_ci_mbb(lambda yt, yp: np.mean((yp - yt) ** 2), yte, y_pred_persist, block_len=L, B=1000, seed=seed_model + 10*L)
        # CI for MSE(persist)
        dlo, dhi = bootstrap_ci_mbb(lambda yt, yp: np.mean(np.mean((y_pred - yt) ** 2, axis=1) - np.mean((yp - yt) ** 2, axis=1)),
                                    yte, y_pred_persist, block_len=L, B=1000, seed=seed_model + 100*L)
        print(f"[MBB] block={L} | CI95(mean d_model-pers)=[{dlo:.6f},{dhi:.6f}]")

    # MBB CI for RMSE(flat) (requested block-length robustness)
    for L in [6, 12, 24]:
        rm_lo, rm_hi = bootstrap_ci_mbb(rmse_all, yte, y_pred, block_len=L, B=1000, seed=seed_model + 1000*L)
        print(f"[MBB] block={L} | RMSE(flat) CI95=[{rm_lo:.3f},{rm_hi:.3f}]")


    rmse_clim0 = rmse_all(yte, y_pred_clim0)
    mae_clim0  = mae_all(yte, y_pred_clim0)
    rmse_pers  = rmse_all(yte, y_pred_persist)
    mae_pers   = mae_all(yte, y_pred_persist)

    print("\n===== GLOBAL TEST (JOINT; TEST cities; years > VAL) =====")
    print(
        f"H={H} | RMSE(flat): {rmse_g:.2f} °C [{rmse_ci[0]:.2f}, {rmse_ci[1]:.2f}] | "
        f"MAE(flat): {mae_g:.2f} °C [{mae_ci[0]:.2f}, {mae_ci[1]:.2f}] | "
        f"NLL(per-h): {nll_ph:.4f} nats | EnergyScore(joint): {es:.4f}"
    )
    print(f"AVG over horizons | RMSE: {rmse_avg:.2f} | MAE: {mae_avg:.2f} | CRPS: {crps_avg:.4f} | Cov@95(avg): {cov95_avg:.1f}%")
    print(f"Clim-0(flat)  | RMSE: {rmse_clim0:.2f} | MAE: {mae_clim0:.2f}")
    print(f"Persist(flat) | RMSE: {rmse_pers:.2f} | MAE: {mae_pers:.2f}")

    del xte_last_anom
    gc.collect()

    # ---------------- TinyTransformer baseline ----------------
    RUN_TINY_TRANSFORMER = True
    if RUN_TINY_TRANSFORMER:
        tt0 = time.time()
        tinyT = train_tiny_transformer(train_loader, val_loader, H=H, device=device,
                                      max_epochs=50, lr=1e-3, patience=8, seed=seed_split)
        tt1 = time.time()

        tinyT.eval()
        preds_t = []
        ys_t = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                pred = tinyT(xb).detach().cpu().numpy()
                preds_t.append(pred)
                ys_t.append(yb.detach().cpu().numpy())
        preds_t = np.concatenate(preds_t, axis=0)
        ys_t = np.concatenate(ys_t, axis=0)

        rmse_tt = float(np.sqrt(np.mean((preds_t - ys_t) ** 2)))
        mae_tt = float(np.mean(np.abs(preds_t - ys_t)))
        print(f"TinyTransformer(flat) | RMSE: {rmse_tt:.2f} | MAE: {mae_tt:.2f} | train_sec: {tt1-tt0:.1f}")


    print("\n----- Per-horizon summary (h=1..H) -----")
    for h in range(H):
        print(f"h={h+1:2d} | RMSE={rmse_h[h]:.2f} | MAE={mae_h[h]:.2f} | CRPS={crps_h[h]:.4f} | Cov@95={cov95_h[h]:.1f}%")

    # Decade-wise aggregation using the first horizon year to study temporal stability of performance
    decades = np.unique((yte_year1 // 10) * 10)
    print("\n----- Decade-wise (TEST; grouped by first horizon year) -----")
    for d0 in decades:
        m = ((yte_year1 // 10) * 10) == d0
        if m.sum() < 10:
            continue
        yt = yte[m]           # [N_d, H]
        yp = y_pred[m]        # [N_d, H]
        sm = samples[:, m, :] # [S, N_d, H]

        crps_list = []
        cov95_list = []
        for h in range(H):
            crps_list.append(crps_from_samples_1d(yt[:, h], sm[:, :, h]))
            cov95_list.append(100.0 * coverage_from_samples_1d(yt[:, h], sm[:, :, h], 0.025, 0.975))

        print(
            f"{d0}s | N={m.sum():4d} | RMSE(flat)={rmse_all(yt, yp):.2f} | "
            f"MAE(flat)={mae_all(yt, yp):.2f} | CRPS(avg)={float(np.mean(crps_list)):.4f} | "
            f"Cov@95(avg)={float(np.mean(cov95_list)):.1f}%"
        )


    # regime-wise
    print("\n----- Regime-wise (TEST; grouped by first horizon year ranges) -----")
    regimes = [
      ("2001-2004", 2001, 2004),
      ("2005-2007", 2005, 2007),
      ("2008-2010", 2008, 2010),
      ("2011-2014", 2011, 2014),
      ("2015-2017", 2015, 2017),
      ("2018-2020", 2018, 2020),
    ]
    for name, a, b in regimes:
        m_reg = (yte_year1 >= a) & (yte_year1 <= b)
        if m_reg.sum() < 20:
            continue

        yt = yte[m_reg]
        yp = y_pred[m_reg]
        sm = samples[:, m_reg, :]

        crps_list = []
        cov95_list = []
        for h in range(H):
            crps_list.append(crps_from_samples_1d(yt[:, h], sm[:, :, h]))
            cov95_list.append(100.0 * coverage_from_samples_1d(yt[:, h], sm[:, :, h], 0.025, 0.975))

        print(
            f"{name} | N={m_reg.sum():4d} | RMSE(flat)={rmse_all(yt, yp):.2f} | "
            f"MAE(flat)={mae_all(yt, yp):.2f} | CRPS(avg)={float(np.mean(crps_list)):.4f} | "
            f"Cov@95(avg)={float(np.mean(cov95_list)):.1f}%"
        )

    # per-city table (same as your code)
    dataset_city_by_norm = {norm_city_name(city): city for city in city2id.keys()}

    rows = []
    for canon in eval_cities:
        cn = norm_city_name(canon)
        ds_city = dataset_city_by_norm.get(cn)

        if ds_city is None:
            rows.append([canon, "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"])
            continue

        cid = city2id[ds_city]
        m_city = (cte == cid)
        if m_city.sum() == 0:
            rows.append([canon, "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"])
            continue

        # --- DM per-city (model vs persist) ---
        d_city = d_mp[m_city]  # d_mp already exists (model - persist), shape [N_test]
        lag_city = nw_lag_rule(int(d_city.size))
        dm_city, p_city = dm_test_newey_west(d_city, lag=lag_city)

        region = city_region.get(ds_city, "n/a")
        n_test_windows = count_test_windows_for_city(
            monthly[monthly["City"] == ds_city],
            window=args.window,
            horizon=H,
            val_end=val_end,
            max_pchip_gap=MAX_PCHIP_GAP,
        )

        yt = yte[m_city]
        yp = y_pred[m_city]
        sm = samples[:, m_city, :]

        rm = rmse_all(yt, yp)
        ma = mae_all(yt, yp)

        with torch.no_grad():
            Xc = torch.tensor(Xte[m_city]).to(device)
            yc = torch.tensor(yt).to(device)
            logp = model.log_prob(Xc, yc)
            nll_city_ph = float((-logp).mean().item()) / float(H)

        crps_list, cov50_list, cov90_list, cov95_list, cia95_list = [], [], [], [], []
        for h in range(H):
            crps_hh = crps_from_samples_1d(yt[:, h], sm[:, :, h])
            cov50_hh = 100.0 * coverage_from_samples_1d(yt[:, h], sm[:, :, h], 0.25, 0.75)
            cov90_hh = 100.0 * coverage_from_samples_1d(yt[:, h], sm[:, :, h], 0.05, 0.95)
            cov95_hh = 100.0 * coverage_from_samples_1d(yt[:, h], sm[:, :, h], 0.025, 0.975)
            cia_hh = cia95_from_cov95(cov95_hh)

            crps_list.append(crps_hh)
            cov50_list.append(cov50_hh)
            cov90_list.append(cov90_hh)
            cov95_list.append(cov95_hh)
            cia95_list.append(cia_hh)

        rows.append([
            canon,
            region,
            int(n_test_windows),
            f"{rm:.2f}",
            f"{ma:.2f}",
            f"{nll_city_ph:.4f}",
            f"{float(np.mean(crps_list)):.4f}",
            f"{float(np.mean(cia95_list)):.2f}%",
            f"{float(np.mean(cov50_list)):.1f}%",
            f"{float(np.mean(cov90_list)):.1f}%",
            f"{float(np.mean(cov95_list)):.1f}%",
            int(m_city.sum()),
            f"{dm_city:.3f}",
            f"{p_city:.4g}",
        ])

    table = pd.DataFrame(
        rows,
        columns=[
            "City",
            "Region",
            "N_test_windows",
            "RMSE (°C)",
            "MAE (°C)",
            "NLL (per-h)",
            "CRPS (°C)",
            "CIA95",
            "Cov@50",
            "Cov@90",
            "Cov@95",
            "N_windows_in_test",
            "DM_vs_Persist",
            "DM_p",
        ],
    )

    print("\n----- Per-city (eval cities; years > VAL) -----")
    print(table.to_string(index=False))
    # --- rewrite manifest at the end so it includes any late-added fields ---
    with open(f"run_manifest_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)


    pvals = []
    for _, r in table.iterrows():
        try:
            p = float(r["DM_p"])
        except Exception:
            p = np.nan
        if np.isfinite(p):
            pvals.append(p)

    p_fisher = fisher_pvalue(pvals)
    sig_count = int(np.sum(np.asarray(pvals) < 0.05)) if len(pvals) else 0
    print(f"\n[DM] Per-city summary | significant(p<0.05): {sig_count}/{len(pvals)} | Fisher combined p={p_fisher:.4g}")

    run_wall_t1 = time.time()
    run_total_sec = float(run_wall_t1 - run_wall_t0)
    run_manifest["compute"]["run_total_seconds"] = run_total_sec

    # ---- return scalars for seed summary
    return {
        "seed_split": seed_split,
        "seed_model": seed_model,
        "H": H,
        "rmse_flat": float(rmse_g),
        "mae_flat": float(mae_g),
        "nll_per_h": float(nll_ph),
        "energy_joint": float(es_joint),
        "energy_ind": float(es_ind),
        "pit_ks": float(pit_ks_val),
        "iw95": float(iw95),
        "cal_ae": float(cal_ae),
        "corr_offdiag": float(corr_off),
        "n_test": int(Xte.shape[0]),
        "eval_set_id": int(eval_set_id),
        "eval_seed": int(eval_seed),
        "eval_cities": "|".join(eval_cities),
        "crps_avg": float(crps_avg),
        "cov95_avg": float(cov95_avg),
        "dispersion_v_val": float(v_disp),
        "dm_vs_persist": float(dm_stat),
        "dm_p_vs_persist": float(dm_p),
        "mean_d_mse_vs_persist": float(d_mp.mean()),
        "run_total_seconds": float(run_total_sec),
    }

def compute_epistemic_for_evalset(eval_set_id: int, seed_split: int, seed_models: list[int], H: int):
    files = []
    for sm in seed_models:
        tag_prefix = f"set{eval_set_id}_seedS{seed_split}_seedM{sm}_H{H}"
        p = NPZ_DIR / f"epi_moments_{tag_prefix}.npz"
        if p.exists():
            files.append(str(p))

    if len(files) < 2:
        print(f"Not enough models for epistemic on eval_set={eval_set_id} (found {len(files)}).")
        return

    mus = []
    vars_ = []
    for fp in files:
        d = np.load(fp)
        mus.append(d["mu"])
        vars_.append(d["var"])

    mus = np.stack(mus, axis=0)     # [M,N,H]
    vars_ = np.stack(vars_, axis=0) # [M,N,H]

    aleatoric = vars_.mean(axis=0)         # [N,H]
    epistemic = mus.var(axis=0, ddof=0)    # [N,H]
    total = aleatoric + epistemic
    frac = epistemic / (total + 1e-12)

    rows = []
    for h in range(H):
        rows.append({
            "eval_set_id": eval_set_id,
            "seed_split": seed_split,
            "h": h + 1,
            "aleatoric_mean": float(aleatoric[:, h].mean()),
            "epistemic_mean": float(epistemic[:, h].mean()),
            "epistemic_frac": float(frac[:, h].mean()),
        })

    df_epi = pd.DataFrame(rows)
    print("\nEpistemic decomposition (ensemble across seed_model):")
    print(df_epi.to_string(index=False))

    save_table(df_epi, name=f"epistemic_evalset{eval_set_id}_seedS{seed_split}_H{H}")

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(df_epi["h"], df_epi["epistemic_frac"], marker="o")
    ax.set_title("Epistemic uncertainty fraction vs horizon")
    ax.set_xlabel("Horizon (months)")
    ax.set_ylabel("Epistemic / (total)")
    save_fig(fig, f"fig_r33_epistemic_frac_evalset{eval_set_id}_seedS{seed_split}_H{H}")


# ---------------- MAIN ----------------
def main(argv=None) -> None:
    args = parse_args(argv)

    seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip() != ""]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=== Reproducibility (ENV) ===")
    print(f"Device: {device}")
    print(f"Torch: {torch.__version__} | NumPy: {np.__version__} | Pandas: {pd.__version__}")
    print(f"CSV_PATH: {args.csv_path}")
    print(f"Seeds to run: {seed_list}")
    print("=============================")

    all_runs = []

    # Pre-pass: read the dataset once, build monthly + anomalies, to generate the disjoint sets
    df_raw_all = read_csv(args.csv_path)
    df_raw_all = df_raw_all[df_raw_all["Year"] >= 1900].copy()
    y0_all, y1_all = probe_span(df_raw_all)

    monthly_all = aggregate_monthly(df_raw_all, y0_all, y1_all)
    monthly_all = exclude_incomplete_years(monthly_all, min_months=MIN_MONTHS_PER_YEAR)
    monthly_all = reindex_month_grid(monthly_all, y0_all, y1_all)

    train_end_all, val_end_all = split_years(y0_all, y1_all)
    monthly_all = build_anomalies_fixed_climatology(monthly_all, train_end=train_end_all)

    diagnose_eval_city_pool(
        df_raw_all,
        monthly_all,
        window=args.window,
        horizon=int(args.horizon),
        val_end=val_end_all,
        max_pchip_gap=MAX_PCHIP_GAP,
        min_test_windows=60,
    )

    # build eval_sets only once (here is the key for disjoint)
    if (EVAL_MODE == "fixed"):
        eval_sets = [list(EVAL_CITIES_FIXED)]
    else:
        eval_sets = build_eval_sets_disjoint(
            df_raw=df_raw_all,
            monthly=monthly_all,
            window=args.window,
            horizon=int(args.horizon),
            train_end=train_end_all,
            val_end=val_end_all,
            max_pchip_gap=MAX_PCHIP_GAP,
            n_eval_sets=int(args.n_eval_sets),
            n_eval=EVAL_N,
            seed_base=int(args.eval_seed_base),
            min_test_windows=60,
            enforce_disjoint=ENFORCE_DISJOINT_EVAL_SETS,
        )

    print("\n[DIAG] Eval sets (disjoint):")
    for i, s in enumerate(eval_sets):
        print(i, s)
    print("[DIAG] Unique eval cities total:",
          len({norm_city_name(c) for ss in eval_sets for c in ss}))

    del df_raw_all
    del monthly_all
    gc.collect()

    # Run: each eval set has its own seed_split, and inside you run 3 model seeds
    for set_id, eval_cities in enumerate(eval_sets):
        eval_seed = int(args.eval_seed_base + set_id)
        seed_split = int(args.split_seed_base + set_id)

        for seed_model in seed_list:
            metrics = run_once(
                args,
                seed_split=seed_split,
                seed_model=seed_model,
                device=device,
                eval_set_id=set_id,
                eval_seed=eval_seed,
                eval_cities=eval_cities,
            )
            all_runs.append(metrics)

        compute_epistemic_for_evalset(set_id, seed_split, seed_list, int(args.horizon))

    # Save raw runs
    df_runs = pd.DataFrame(all_runs)
    df_runs.to_csv("runs_all.csv", index=False)

    metric_keys = [
        "rmse_flat", "mae_flat", "nll_per_h", "energy_joint", "energy_ind",
        "pit_ks", "iw95", "cal_ae", "corr_offdiag", "crps_avg", "cov95_avg"
    ]

    # Aggregate per eval set (mean over model seeds)
    df_set = (
        df_runs.groupby("eval_set_id", as_index=False)[metric_keys]
        .mean()
        .sort_values("eval_set_id")
        .reset_index(drop=True)
    )
    df_set.to_csv("runs_by_evalset.csv", index=False)

    # Print distributions across eval sets
    if len(eval_sets) > 1:
        print("\n" + "#" * 70)
        print(f"EVAL-SET DISTRIBUTION (over {len(eval_sets)} different 10-city TEST sets)")
        print("#" * 70)

        for k in metric_keys:
            vals = df_set[k].to_numpy(dtype=np.float64)
            mu = float(vals.mean())
            sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            print(f"{k}: {mu:.4f} ± {sd:.4f}")

        plt.figure()
        plt.boxplot(df_set["rmse_flat"].to_numpy(dtype=np.float64))
        plt.ylabel("RMSE (°C)")
        plt.title(f"RMSE across {len(eval_sets)} eval city sets (mean over model seeds)")
        plt.savefig("boxplot_rmse_evalsets.png", dpi=200, bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.boxplot(df_set["energy_joint"].to_numpy(dtype=np.float64))
        plt.ylabel("EnergyScore (joint)")
        plt.title(f"EnergyScore across {len(eval_sets)} eval city sets (mean over model seeds)")
        plt.savefig("boxplot_energy_evalsets.png", dpi=200, bbox_inches="tight")
        plt.close()

    # Single-run final line
    if len(all_runs) == 1:
        r = all_runs[0]
        print("\nFINAL:")
        print(
            f"seed={r['seed_model']} | RMSE={r['rmse_flat']:.3f} | MAE={r['mae_flat']:.3f} | "
            f"NLL={r['nll_per_h']:.3f} | ES(joint)={r['energy_joint']:.3f} | PIT-KS={r['pit_ks']:.3f}"
        )

    # Zip everything in artifacts/ for easy download
    zip_path = zip_artifacts("artifacts.zip")

    # optional Colab download
    try:
        from google.colab import files
        files.download(zip_path)
    except Exception:
        print("[ART] Zip ready at:", zip_path)


if __name__ == "__main__":
    main()
