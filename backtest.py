#!/usr/bin/env python3
"""Medicare Fraud Backtest POC — Multi-state, best-available billing year."""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import requests

DATA_DIR = Path("data")
FIGURES_DIR = Path("figures")

LEIE_URL = "https://oig.hhs.gov/exclusions/downloadables/UPDATED.csv"

TOP_STATES = None  # None = all states
EXCL_YEARS = range(2018, 2024)  # exclusion date range
BILLING_EXCL_TYPES = ["1128a1", "1128a3", "1128b7"]  # fraud-related exclusions

CMS_PROVIDER_UUIDS = {
    2017: "44ea2789-993f-4d55-ac44-ed7f160b58fa",
    2018: "900850df-c9a9-47ce-a9e0-d0ceae5a811f",
    2019: "6a53afe5-1cbc-4b33-9dc8-926ee532dc66",
    2020: "29d799aa-c660-44fe-a51a-72c4b3e661ac",
    2021: "44e0a489-666c-4ea4-a1a2-360b6cdc19db",
    2022: "21555c17-ec1b-4e74-b2c6-925c6cbb3147",
}

CMS_SERVICE_UUIDS = {
    2017: "85bf3c9c-2244-490d-ad7d-c34e4c28f8ea",
    2018: "fb6d9fe8-38c1-4d24-83d4-0b7b291000b2",
    2019: "867b8ac7-ccb7-4cc9-873d-b24340d89e32",
    2020: "c957b49e-1323-49e7-8678-c09da387551d",
    2021: "31dc2c47-f297-4948-bfb4-075e1bec3a02",
    2022: "e650987d-01b7-4f09-b75e-b0b075afbf98",
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_leie() -> pd.DataFrame:
    path = DATA_DIR / "UPDATED.csv"
    if path.exists() and (time.time() - path.stat().st_mtime) < 86400:
        print(f"  Cached LEIE: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"  Downloading LEIE ...")
        resp = requests.get(LEIE_URL, stream=True, timeout=120)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB", end="", flush=True)
        print(f"\n  Saved {path}")
    df = pd.read_csv(path, dtype={"NPI": str}, low_memory=False)
    print(f"  LEIE: {len(df):,} records")
    return df


def fetch_cms_paginated(dataset_id: str, params: dict, label: str) -> pd.DataFrame:
    base_url = f"https://data.cms.gov/data-api/v1/dataset/{dataset_id}/data"
    page_size = 5000
    offset = 0
    all_rows = []
    retries = 0
    while True:
        p = {**params, "size": page_size, "offset": offset}
        try:
            resp = requests.get(base_url, params=p, timeout=120)
            resp.raise_for_status()
            rows = resp.json()
            if not rows:
                break
            all_rows.extend(rows)
            offset += page_size
            print(f"\r  {label}: {len(all_rows):,}", end="", flush=True)
            retries = 0
        except Exception as e:
            retries += 1
            if retries > 3:
                print(f"\n  ERROR: {label} failed: {e}")
                break
            time.sleep(2 ** retries)
    if all_rows:
        print()
    return pd.DataFrame(all_rows)


def download_cms_state_year(state: str, year: int, kind: str) -> pd.DataFrame:
    """Download Part B provider or service data for one state+year. Cached to parquet."""
    uuids = CMS_PROVIDER_UUIDS if kind == "provider" else CMS_SERVICE_UUIDS
    if year not in uuids:
        return pd.DataFrame()

    cache = DATA_DIR / f"cms_{kind}_{state}_{year}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    df = fetch_cms_paginated(
        uuids[year],
        {"filter[Rndrng_Prvdr_State_Abrvtn]": state},
        f"{kind} {state} {year}",
    )
    if not df.empty:
        for col in df.columns:
            if any(k in col for k in ["Tot_", "Bene_Avg", "Avg_", "Bene_Age",
                                       "Bene_Feml", "Bene_Male", "Bene_Race",
                                       "Bene_Dual", "Bene_Ndual", "Bene_CC_",
                                       "Drug_Tot", "Med_Tot", "Drug_S", "Med_S",
                                       "Drug_M", "Med_M"]):
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df.to_parquet(cache)
    return df


# ---------------------------------------------------------------------------
# Build excluded cohort
# ---------------------------------------------------------------------------

def build_excluded_cohort(leie_df: pd.DataFrame) -> pd.DataFrame:
    df = leie_df.copy()
    if TOP_STATES is not None:
        df = df[df["STATE"].isin(TOP_STATES)]
        print(f"  Filtered to {len(TOP_STATES)} states: {len(df):,}")
    else:
        df = df[df["STATE"].notna() & (df["STATE"].str.len() == 2)]
        print(f"  All US states: {len(df):,}")

    df["EXCLDATE"] = pd.to_datetime(df["EXCLDATE"], format="%Y%m%d", errors="coerce")
    df = df[df["EXCLDATE"].dt.year.isin(EXCL_YEARS)]
    print(f"  Excluded {min(EXCL_YEARS)}-{max(EXCL_YEARS)}: {len(df):,}")

    df = df[df["EXCLTYPE"].isin(BILLING_EXCL_TYPES)]
    print(f"  Billing-specific types ({', '.join(BILLING_EXCL_TYPES)}): {len(df):,}")

    df = df[df["NPI"].notna() & (df["NPI"].str.strip() != "") & (df["NPI"] != "0000000000")]
    df = df.drop_duplicates(subset="NPI")
    print(f"  Valid unique NPIs: {len(df):,}")
    print(f"  By state: {df['STATE'].value_counts().to_dict()}")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Find best billing year for each excluded NPI
# ---------------------------------------------------------------------------

def find_best_billing_year(cohort_df: pd.DataFrame) -> pd.DataFrame:
    """For each excluded NPI, find the most recent CMS billing year before exclusion.
    Returns cohort_df with added columns: billing_year, billing_state."""

    results = []
    npis_by_state = cohort_df.groupby("STATE")["NPI"].apply(set).to_dict()
    excl_year_map = dict(zip(cohort_df["NPI"], cohort_df["EXCLDATE"].dt.year))

    for state, npis in npis_by_state.items():
        print(f"  Searching {state} ({len(npis)} NPIs) ...", end="", flush=True)
        found_in_state = set()

        # Check years from most recent backwards
        for year in sorted(CMS_PROVIDER_UUIDS.keys(), reverse=True):
            remaining = npis - found_in_state
            if not remaining:
                break

            # Only check NPIs where this year is pre-exclusion
            eligible = {n for n in remaining if excl_year_map.get(n, 9999) > year}
            if not eligible:
                continue

            provider_df = download_cms_state_year(state, year, "provider")
            if provider_df.empty:
                continue

            provider_df["Rndrng_NPI"] = provider_df["Rndrng_NPI"].astype(str)
            cms_npis = set(provider_df["Rndrng_NPI"])
            matched = eligible & cms_npis

            for npi in matched:
                results.append({"NPI": npi, "billing_year": year, "billing_state": state})
                found_in_state.add(npi)

        not_found = npis - found_in_state
        print(f" found {len(found_in_state)}, missing {len(not_found)}")

    if not results:
        return cohort_df.assign(billing_year=pd.NA, billing_state=pd.NA)

    match_df = pd.DataFrame(results)
    merged = cohort_df.merge(match_df, on="NPI", how="left")
    matched_count = merged["billing_year"].notna().sum()
    print(f"  Total matched: {matched_count} of {len(cohort_df)}")
    print(f"  By billing year: {merged['billing_year'].value_counts().sort_index().to_dict()}")
    return merged


# ---------------------------------------------------------------------------
# Load billing data for matched cohort + peers
# ---------------------------------------------------------------------------

def load_billing_data(matched_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load provider and service data for all state+year combos needed."""
    combos = (matched_df[matched_df["billing_year"].notna()]
              .groupby(["billing_state", "billing_year"])
              .size().reset_index(name="n"))

    all_provider = []
    all_service = []

    for _, row in combos.iterrows():
        state = row["billing_state"]
        year = int(row["billing_year"])
        print(f"  Loading {state}/{year} (n={row['n']}) ...")

        pdf = download_cms_state_year(state, year, "provider")
        if not pdf.empty:
            pdf["_state"] = state
            pdf["_year"] = year
            pdf["Rndrng_NPI"] = pdf["Rndrng_NPI"].astype(str)
            all_provider.append(pdf)

        sdf = download_cms_state_year(state, year, "service")
        if not sdf.empty:
            sdf["_state"] = state
            sdf["_year"] = year
            sdf["Rndrng_NPI"] = sdf["Rndrng_NPI"].astype(str)
            all_service.append(sdf)

    provider_df = pd.concat(all_provider, ignore_index=True) if all_provider else pd.DataFrame()
    service_df = pd.concat(all_service, ignore_index=True) if all_service else pd.DataFrame()
    print(f"  Provider rows: {len(provider_df):,}, Service rows: {len(service_df):,}")
    return provider_df, service_df


# ---------------------------------------------------------------------------
# Build peer group — same state, same year, same specialty
# ---------------------------------------------------------------------------

def build_peer_group(
    provider_df: pd.DataFrame, excluded_npis: set[str]
) -> pd.DataFrame:
    peers = provider_df[~provider_df["Rndrng_NPI"].isin(excluded_npis)].copy()
    peers = peers[peers["Tot_Benes"] >= 11]
    print(f"  Peer group (all states/years, same specialties): {len(peers):,}")
    return peers


# ---------------------------------------------------------------------------
# Compute features
# ---------------------------------------------------------------------------

def compute_features(
    provider_df: pd.DataFrame,
    service_df: pd.DataFrame,
    excluded_npis: set[str],
) -> pd.DataFrame:
    f = pd.DataFrame()
    f["npi"] = provider_df["Rndrng_NPI"].values
    f["group"] = f["npi"].apply(lambda x: "excluded" if x in excluded_npis else "peer")
    f["provider_type"] = provider_df["Rndrng_Prvdr_Type"].values
    f["state"] = provider_df["_state"].values
    f["year"] = provider_df["_year"].values

    for src, dst in [
        ("Tot_HCPCS_Cds", "unique_hcpcs"),
        ("Tot_Benes", "total_benes"),
        ("Tot_Srvcs", "total_services"),
        ("Tot_Sbmtd_Chrg", "submitted_charges"),
        ("Tot_Mdcr_Alowd_Amt", "allowed_amount"),
        ("Tot_Mdcr_Pymt_Amt", "medicare_payment"),
        ("Bene_Avg_Age", "bene_avg_age"),
        ("Bene_Avg_Risk_Scre", "bene_avg_risk"),
    ]:
        if src in provider_df.columns:
            f[dst] = provider_df[src].values

    f["services_per_bene"] = f["total_services"] / f["total_benes"].replace(0, np.nan)
    f["charge_to_payment_ratio"] = f["submitted_charges"] / f["medicare_payment"].replace(0, np.nan)
    f["avg_charge_per_service"] = f["submitted_charges"] / f["total_services"].replace(0, np.nan)
    f["payment_per_bene"] = f["medicare_payment"] / f["total_benes"].replace(0, np.nan)

    if "Bene_Dual_Cnt" in provider_df.columns:
        dual = pd.to_numeric(provider_df["Bene_Dual_Cnt"], errors="coerce")
        ndual = pd.to_numeric(provider_df["Bene_Ndual_Cnt"], errors="coerce")
        f["dual_share"] = (dual / (dual + ndual).replace(0, np.nan)).values

    # HCPCS concentration
    if not service_df.empty:
        svc = service_df[["Rndrng_NPI", "Tot_Srvcs"]].copy()
        svc.columns = ["npi", "services"]
        svc["services"] = pd.to_numeric(svc["services"], errors="coerce").fillna(0)

        total_by_npi = svc.groupby("npi")["services"].sum()
        svc = svc.merge(total_by_npi.rename("total"), on="npi")
        svc["share"] = svc["services"] / svc["total"].replace(0, np.nan)

        hhi = svc.groupby("npi")["share"].apply(lambda x: (x ** 2).sum()).rename("hcpcs_herfindahl")
        top_share = svc.groupby("npi")["share"].max().rename("top_hcpcs_share")
        n_codes = svc[svc["services"] > 0].groupby("npi").size().rename("n_hcpcs_billed")

        f = f.merge(hhi, left_on="npi", right_index=True, how="left")
        f = f.merge(top_share, left_on="npi", right_index=True, how="left")
        f = f.merge(n_codes, left_on="npi", right_index=True, how="left")

    excl_n = (f["group"] == "excluded").sum()
    peer_n = (f["group"] == "peer").sum()
    print(f"  Excluded: {excl_n}, Peers: {peer_n:,}")
    return f


# ---------------------------------------------------------------------------
# Statistical comparison
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "total_services", "total_benes", "unique_hcpcs",
    "submitted_charges", "medicare_payment",
    "services_per_bene", "charge_to_payment_ratio",
    "avg_charge_per_service", "payment_per_bene",
    "bene_avg_age", "bene_avg_risk", "dual_share",
    "hcpcs_herfindahl", "top_hcpcs_share", "n_hcpcs_billed",
]


def compare_groups(features_df: pd.DataFrame) -> pd.DataFrame:
    excl = features_df[features_df["group"] == "excluded"]
    peer = features_df[features_df["group"] == "peer"]
    results = []

    for feat in [c for c in FEATURE_COLS if c in features_df.columns]:
        e = excl[feat].dropna()
        p = peer[feat].dropna()
        if len(e) < 2 or len(p) < 10:
            continue

        p_std = p.std()
        cohens_d = (e.mean() - p.mean()) / p_std if p_std > 0 else 0.0

        try:
            u_stat, p_mw = stats.mannwhitneyu(e, p, alternative="two-sided")
        except Exception:
            u_stat, p_mw = np.nan, np.nan
        try:
            t_stat, p_t = stats.ttest_ind(e, p, equal_var=False)
        except Exception:
            t_stat, p_t = np.nan, np.nan

        results.append({
            "feature": feat, "n_excluded": len(e), "n_peer": len(p),
            "excluded_mean": e.mean(), "peer_mean": p.mean(),
            "excluded_median": e.median(), "peer_median": p.median(),
            "cohens_d": cohens_d, "u_stat": u_stat, "p_mw": p_mw,
            "t_stat": t_stat, "p_t": p_t,
        })

    comp = pd.DataFrame(results)
    if not comp.empty:
        n = len(comp)
        comp["p_mw_bonf"] = (comp["p_mw"] * n).clip(upper=1.0)
        comp["significant"] = comp["p_mw_bonf"] < 0.05
        comp["large_effect"] = comp["cohens_d"].abs() > 0.5

    print(f"\n  Statistical comparison ({len(comp)} features):")
    for _, r in comp.iterrows():
        sig = "*" if r.get("significant") else " "
        eff = "!" if r.get("large_effect") else " "
        print(f"    {sig}{eff} {r['feature']:28s}  d={r['cohens_d']:+.3f}  p={r['p_mw']:.4f}  p_bonf={r['p_mw_bonf']:.4f}")
    return comp


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def create_visualizations(features_df, comparison_df) -> list[str]:
    paths = []
    if comparison_df.empty:
        return paths

    plt.rcParams.update({"figure.dpi": 150, "font.size": 9})
    excl = features_df[features_df["group"] == "excluded"]
    peer = features_df[features_df["group"] == "peer"]

    # 1. Box plots — top 6 by effect size
    top = comparison_df.reindex(
        comparison_df["cohens_d"].abs().sort_values(ascending=False).index
    ).head(6)["feature"].tolist()
    top = [f for f in top if f in features_df.columns]

    if top:
        n = len(top)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
        axes = np.array(axes).flatten() if n > 1 else [axes]
        for i, feat in enumerate(top):
            ax = axes[i]
            ev = excl[feat].dropna()
            pv = peer[feat].dropna()
            bp = ax.boxplot([pv, ev], tick_labels=["Peers", "Excluded"],
                          patch_artist=True, widths=0.5)
            bp["boxes"][0].set_facecolor("#4A90D9"); bp["boxes"][0].set_alpha(0.6)
            bp["boxes"][1].set_facecolor("#D94A4A"); bp["boxes"][1].set_alpha(0.6)
            ax.scatter([2] * len(ev), ev, color="red", alpha=0.5, zorder=5,
                      s=12, edgecolors="black", linewidths=0.3)
            ax.set_title(feat.replace("_", " ").title(), fontsize=9, fontweight="bold")
        for j in range(len(top), len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Excluded vs Peer Distributions (Top Features by Effect Size)",
                    fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        p = FIGURES_DIR / "boxplots.png"
        fig.savefig(p, bbox_inches="tight"); plt.close(fig)
        paths.append(str(p))

    # 2. Radar
    radar_feats = [f for f in comparison_df["feature"].tolist() if f in features_df.columns][:8]
    if len(radar_feats) >= 3:
        z_means = []
        for feat in radar_feats:
            pstd = peer[feat].std()
            z_means.append((excl[feat].mean() - peer[feat].mean()) / pstd if pstd > 0 else 0)
        angles = np.linspace(0, 2 * np.pi, len(radar_feats), endpoint=False).tolist()
        z_plot = z_means + [z_means[0]]
        a_plot = angles + [angles[0]]
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(a_plot, z_plot, "o-", color="#D94A4A", linewidth=2)
        ax.fill(a_plot, z_plot, alpha=0.2, color="#D94A4A")
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xticks(angles)
        ax.set_xticklabels([f.replace("_", "\n") for f in radar_feats], size=7)
        ax.set_title("Excluded Providers: Mean Z-Score Profile\n(vs Peer Distribution)",
                    fontweight="bold", pad=20)
        p = FIGURES_DIR / "radar.png"
        fig.savefig(p, bbox_inches="tight"); plt.close(fig)
        paths.append(str(p))

    # 3. Scatter
    if "services_per_bene" in features_df.columns and "charge_to_payment_ratio" in features_df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        for grp, color, label, alpha, s in [
            ("peer", "#4A90D9", "Peers", 0.2, 10),
            ("excluded", "#D94A4A", "Excluded", 0.9, 30),
        ]:
            m = features_df["group"] == grp
            ax.scatter(features_df.loc[m, "services_per_bene"],
                      features_df.loc[m, "charge_to_payment_ratio"],
                      c=color, alpha=alpha, s=s, label=label,
                      edgecolors="black", linewidths=0.2)
        ax.set_xlabel("Services per Beneficiary")
        ax.set_ylabel("Charge-to-Payment Ratio")
        ax.set_title("Service Intensity vs Billing Markup", fontweight="bold")
        ax.legend()
        xq = features_df["services_per_bene"].quantile(0.99)
        yq = features_df["charge_to_payment_ratio"].quantile(0.99)
        if pd.notna(xq) and pd.notna(yq):
            ax.set_xlim(left=0, right=xq * 1.1)
            ax.set_ylim(bottom=0, top=yq * 1.1)
        p = FIGURES_DIR / "scatter.png"
        fig.savefig(p, bbox_inches="tight"); plt.close(fig)
        paths.append(str(p))

    # 4. Heatmap (sample up to 60 excluded for readability)
    if len(radar_feats) >= 3 and len(excl) > 0:
        excl_sample = excl.head(60)
        excl_f = excl_sample[["npi"] + radar_feats].set_index("npi")
        z_mat = pd.DataFrame(index=excl_f.index, columns=radar_feats, dtype=float)
        for feat in radar_feats:
            pstd = peer[feat].std()
            z_mat[feat] = (excl_f[feat] - peer[feat].mean()) / pstd if pstd > 0 else 0
        if not z_mat.empty:
            h = max(6, len(z_mat) * 0.25 + 2)
            fig, ax = plt.subplots(figsize=(max(8, len(radar_feats) * 1.2), h))
            vmax = max(abs(z_mat.min().min()), abs(z_mat.max().max()), 3)
            im = ax.imshow(z_mat.values.astype(float), cmap="RdBu_r",
                          vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(radar_feats)))
            ax.set_xticklabels([f.replace("_", "\n") for f in radar_feats],
                              rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(z_mat)))
            ax.set_yticklabels(z_mat.index, fontsize=6)
            ax.set_title("Z-Scores: Excluded Providers vs Peer Distribution", fontweight="bold")
            fig.colorbar(im, ax=ax, label="Z-Score", shrink=0.8)
            fig.tight_layout()
            p = FIGURES_DIR / "heatmap.png"
            fig.savefig(p, bbox_inches="tight"); plt.close(fig)
            paths.append(str(p))

    # 5. By-state breakdown
    if len(excl) > 0:
        state_counts = excl.groupby("state").size()
        if len(state_counts) > 1:
            sig_feats = comparison_df[comparison_df.get("significant", False) == True]["feature"].tolist()
            if not sig_feats:
                sig_feats = comparison_df.reindex(
                    comparison_df["cohens_d"].abs().sort_values(ascending=False).index
                ).head(3)["feature"].tolist()
            sig_feats = [f for f in sig_feats if f in features_df.columns][:3]

            if sig_feats:
                fig, axes = plt.subplots(1, len(sig_feats),
                                        figsize=(5 * len(sig_feats), 5))
                if len(sig_feats) == 1:
                    axes = [axes]
                for i, feat in enumerate(sig_feats):
                    ax = axes[i]
                    states_with_excl = features_df[features_df["group"] == "excluded"]["state"].unique()
                    data_peer = []
                    data_excl = []
                    labels = []
                    for st in sorted(states_with_excl):
                        pv = features_df[(features_df["state"] == st) & (features_df["group"] == "peer")][feat].dropna()
                        ev = features_df[(features_df["state"] == st) & (features_df["group"] == "excluded")][feat].dropna()
                        if len(ev) > 0:
                            labels.append(st)
                            data_peer.append(pv.median())
                            data_excl.append(ev.median())
                    x = np.arange(len(labels))
                    ax.bar(x - 0.2, data_peer, 0.35, label="Peers (median)", color="#4A90D9", alpha=0.7)
                    ax.bar(x + 0.2, data_excl, 0.35, label="Excluded (median)", color="#D94A4A", alpha=0.7)
                    ax.set_xticks(x)
                    ax.set_xticklabels(labels)
                    ax.set_title(feat.replace("_", " ").title(), fontweight="bold")
                    ax.legend(fontsize=7)
                fig.suptitle("Key Features by State", fontweight="bold")
                fig.tight_layout(rect=[0, 0, 1, 0.95])
                p = FIGURES_DIR / "by_state.png"
                fig.savefig(p, bbox_inches="tight"); plt.close(fig)
                paths.append(str(p))

    print(f"  Saved {len(paths)} figures")
    return paths


# ---------------------------------------------------------------------------
# Prosecution matching
# ---------------------------------------------------------------------------

DOJ_MATCHES_FILE = DATA_DIR / "doj_matches.json"


def load_doj_matches() -> dict:
    """Load pre-computed DOJ press release matches."""
    if not DOJ_MATCHES_FILE.exists():
        return {"metadata": {}, "matches": [], "no_match_sample": []}
    with open(DOJ_MATCHES_FILE) as f:
        return json.load(f)


def match_prosecutions(cohort_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-reference excluded cohort with DOJ press release matches."""
    doj = load_doj_matches()
    if not doj["matches"]:
        print("  No DOJ matches file found — skipping prosecution matching")
        return pd.DataFrame()

    matched_names = {m["name"].lower(): m for m in doj["matches"]}
    no_match_names = {n["name"].lower() for n in doj["no_match_sample"]}

    cohort = cohort_df.copy()
    cohort["full_name"] = (
        cohort["FIRSTNAME"].fillna("").str.strip() + " " + cohort["LASTNAME"].fillna("").str.strip()
    ).str.strip().str.title()

    results = []
    for _, row in cohort.iterrows():
        if not row["full_name"] or pd.isna(row["full_name"]):
            continue
        name_lower = row["full_name"].lower()
        has_billing = pd.notna(row.get("billing_year"))
        if name_lower in matched_names:
            m = matched_names[name_lower]
            results.append({
                "NPI": row["NPI"],
                "name": row["full_name"],
                "state": row["STATE"],
                "excl_type": row.get("EXCLTYPE", ""),
                "has_billing": has_billing,
                "doj_match": True,
                "doj_summary": m.get("summary", ""),
                "doj_url": m.get("doj_url", ""),
            })
        elif name_lower in no_match_names:
            results.append({
                "NPI": row["NPI"],
                "name": row["full_name"],
                "state": row["STATE"],
                "excl_type": row.get("EXCLTYPE", ""),
                "has_billing": has_billing,
                "doj_match": False,
                "doj_summary": "",
                "doj_url": "",
            })

    df = pd.DataFrame(results)
    meta = doj.get("metadata", {})
    n_searched = meta.get("sample_size", len(results))
    n_matched = meta.get("match_count", df["doj_match"].sum() if not df.empty else 0)
    rate = meta.get("match_rate", n_matched / max(n_searched, 1))

    print(f"  DOJ press release matching: {n_matched}/{n_searched} searched ({rate:.0%} match rate)")
    print(f"  Matched in cohort: {df['doj_match'].sum() if not df.empty else 0} providers with DOJ press releases")
    return df


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_report(cohort_df, provider_df, comparison_df, features_df, figure_paths,
                    prosecution_df=None):
    lines = []
    lines.append("# Medicare Fraud Backtest POC — Results\n")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    excl_n = (features_df["group"] == "excluded").sum()
    peer_n = (features_df["group"] == "peer").sum()
    lines.append("## Cohort Summary\n")
    lines.append(f"- **Excluded providers matched to Part B billing:** {excl_n}")
    lines.append(f"- **Peer providers (same state/specialty/year):** {peer_n:,}")
    states_str = ', '.join(TOP_STATES) if TOP_STATES else f"{features_df['state'].nunique()} states"
    lines.append(f"- **States:** {states_str}")
    lines.append(f"- **Exclusion date range:** {min(EXCL_YEARS)}-{max(EXCL_YEARS)}")
    lines.append(f"- **Billing data:** best available pre-exclusion year per provider\n")

    # State breakdown
    excl_by_state = features_df[features_df["group"] == "excluded"].groupby("state").size()
    lines.append("### By State\n")
    lines.append("| State | Excluded | Peers |")
    lines.append("|-------|----------|-------|")
    for st in sorted(excl_by_state.index):
        en = excl_by_state.get(st, 0)
        pn = (features_df[(features_df["group"] == "peer") & (features_df["state"] == st)]).shape[0]
        lines.append(f"| {st} | {en} | {pn:,} |")
    lines.append("")

    # Excluded provider table
    lines.append("### Excluded Providers\n")
    lines.append("<details><summary>Click to expand (all providers)</summary>\n")
    lines.append("| NPI | Name | State | Specialty | Excl Type | Excl Date | Billing Year |")
    lines.append("|-----|------|-------|-----------|-----------|-----------|-------------|")
    for _, row in cohort_df[cohort_df["billing_year"].notna()].iterrows():
        npi = row["NPI"]
        name = ""
        if not provider_df.empty:
            match = provider_df[provider_df["Rndrng_NPI"] == npi]
            if not match.empty:
                m = match.iloc[0]
                name = f"{m.get('Rndrng_Prvdr_First_Name', '')} {m.get('Rndrng_Prvdr_Last_Org_Name', '')}".strip()
        if not name:
            name = f"{row.get('FIRSTNAME', '')} {row.get('LASTNAME', '')}".strip()
        exdt = row["EXCLDATE"]
        if isinstance(exdt, pd.Timestamp):
            exdt = exdt.strftime("%Y-%m-%d")
        by = int(row["billing_year"]) if pd.notna(row["billing_year"]) else ""
        lines.append(f"| {npi} | {name} | {row['STATE']} | {row.get('SPECIALTY','')} | {row.get('EXCLTYPE','')} | {exdt} | {by} |")
    lines.append("\n</details>\n")

    # Stats
    if not comparison_df.empty:
        lines.append("## Statistical Comparison\n")
        lines.append("| Feature | n_excl | Excluded Mean | Peer Mean | Cohen's d | p (M-W) | p (Bonf) | Sig? |")
        lines.append("|---------|--------|-------------|-----------|----------|---------|----------|------|")
        for _, r in comparison_df.iterrows():
            sig = "YES" if r.get("significant") else ""
            lines.append(
                f"| {r['feature']} | {r['n_excluded']:.0f} | {r['excluded_mean']:.1f} | {r['peer_mean']:.1f} "
                f"| {r['cohens_d']:+.2f} | {r['p_mw']:.4f} | {r['p_mw_bonf']:.4f} | {sig} |"
            )
        lines.append("")

        sig_feats = comparison_df[comparison_df.get("significant", False) == True]
        trending = comparison_df[comparison_df["p_mw"] < 0.05]

        lines.append("## Key Findings\n")
        if len(sig_feats) >= 2:
            lines.append(
                f"**CLEAR SIGNAL:** Excluded providers show statistically significant deviation "
                f"on {len(sig_feats)} features (Bonferroni-corrected).\n")
            for _, r in sig_feats.iterrows():
                d = "higher" if r["cohens_d"] > 0 else "lower"
                lines.append(f"- **{r['feature']}**: excluded are {d} (d={r['cohens_d']:+.2f})")
        elif len(sig_feats) >= 1 or len(trending) >= 3:
            lines.append(
                f"**WEAK SIGNAL:** {len(sig_feats)} feature(s) survive Bonferroni, "
                f"{len(trending)} show raw p<0.05.\n")
            if len(sig_feats):
                lines.append("Significant (Bonferroni):")
                for _, r in sig_feats.iterrows():
                    d = "higher" if r["cohens_d"] > 0 else "lower"
                    lines.append(f"- **{r['feature']}**: excluded are {d} (d={r['cohens_d']:+.2f})")
            if len(trending[~trending["significant"]]):
                lines.append("\nTrending (raw p<0.05):")
                for _, r in trending[~trending["significant"]].iterrows():
                    d = "higher" if r["cohens_d"] > 0 else "lower"
                    lines.append(f"- **{r['feature']}**: excluded are {d} (d={r['cohens_d']:+.2f}, p={r['p_mw']:.4f})")
        else:
            lines.append("**NO SIGNAL:** Excluded providers are indistinguishable from peers.\n")
        lines.append("")

    # Prosecution matching section
    if prosecution_df is not None and not prosecution_df.empty:
        doj = load_doj_matches()
        meta = doj.get("metadata", {})
        n_searched = meta.get("sample_size", 0)
        n_matched = meta.get("match_count", 0)
        rate = meta.get("match_rate", 0)

        lines.append("## DOJ Prosecution Matching\n")
        lines.append(
            f"A sample of **{n_searched}** excluded providers was searched against "
            f"DOJ press releases on justice.gov. **{n_matched}** ({rate:.0%}) matched "
            f"to published federal prosecutions.\n"
        )
        lines.append(
            "This is a floor estimate — §1128(a)(1) exclusion requires a conviction, "
            "so all providers in that category were prosecuted. Many cases are handled "
            "at state level or via plea agreements that don't generate DOJ press releases.\n"
        )

        matched_rows = prosecution_df[prosecution_df["doj_match"]]
        if not matched_rows.empty:
            lines.append("### Matched Providers\n")
            lines.append("| Name | State | Type | DOJ Summary |")
            lines.append("|------|-------|------|-------------|")
            for _, r in matched_rows.iterrows():
                url = r["doj_url"]
                name_col = f"[{r['name']}]({url})" if url and str(url) != "nan" else r["name"]
                lines.append(f"| {name_col} | {r['state']} | {r['excl_type']} | {r['doj_summary']} |")
            lines.append("")

        # Match rate by state
        by_state = prosecution_df.groupby("state").agg(
            searched=("doj_match", "count"),
            matched=("doj_match", "sum"),
        )
        by_state["rate"] = by_state["matched"] / by_state["searched"]
        by_state = by_state.sort_values("rate", ascending=False)

        lines.append("### Match Rate by State\n")
        lines.append("| State | Searched | Matched | Rate |")
        lines.append("|-------|----------|---------|------|")
        for st, row in by_state.iterrows():
            lines.append(f"| {st} | {int(row['searched'])} | {int(row['matched'])} | {row['rate']:.0%} |")
        lines.append("")

    if figure_paths:
        lines.append("## Visualizations\n")
        for fp in figure_paths:
            name = Path(fp).stem.replace("_", " ").title()
            lines.append(f"### {name}\n![{name}]({fp})\n")

    lines.append("## Caveats\n")
    lines.append(f"- Sample: {excl_n} excluded providers (larger is better, still moderate)")
    lines.append("- Survivorship bias: only providers who billed Part B before exclusion")
    lines.append(f"- Multiple comparisons: {len(comparison_df)} features; Bonferroni correction applied")
    lines.append("- Mixed specialties across states — peer matching by specialty mitigates but doesn't eliminate")
    lines.append("- Best-available billing year varies per provider (some 2017, some 2022)")
    lines.append("- Peer group not matched on practice size or sub-state geography")
    lines.append("")

    Path("report.md").write_text("\n".join(lines))
    print(f"  Report: report.md")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 60)
    print("MEDICARE FRAUD BACKTEST POC — EXPANDED")
    print(f"States: {', '.join(TOP_STATES) if TOP_STATES else 'ALL'}")
    print(f"Exclusions: {min(EXCL_YEARS)}-{max(EXCL_YEARS)}")
    print(f"Types: {', '.join(BILLING_EXCL_TYPES)}")
    print("=" * 60)

    print("\n[1/6] LEIE ...")
    leie_df = download_leie()

    print("\n[2/6] Building excluded cohort ...")
    cohort_df = build_excluded_cohort(leie_df)
    if cohort_df.empty:
        print("  FATAL: No excluded providers."); sys.exit(1)

    print("\n[3/6] Finding best billing year per provider ...")
    cohort_df = find_best_billing_year(cohort_df)
    matched = cohort_df[cohort_df["billing_year"].notna()]
    print(f"  Matched: {len(matched)} providers with billing data")
    if len(matched) < 10:
        print("  FATAL: Too few matched providers."); sys.exit(1)

    excluded_npis = set(matched["NPI"])

    print("\n[4/6] Loading billing data + building peers ...")
    provider_df, service_df = load_billing_data(matched)

    # Get specialties of excluded for peer filtering
    excl_types = set(provider_df[provider_df["Rndrng_NPI"].isin(excluded_npis)]["Rndrng_Prvdr_Type"].dropna())
    provider_df_filtered = provider_df[provider_df["Rndrng_Prvdr_Type"].isin(excl_types)]
    peer_df = build_peer_group(provider_df_filtered, excluded_npis)

    # Only include each excluded NPI in their assigned billing year/state
    npi_year = dict(zip(matched["NPI"], matched["billing_year"].astype(int)))
    npi_state = dict(zip(matched["NPI"], matched["billing_state"]))
    excl_rows = provider_df_filtered[
        provider_df_filtered["Rndrng_NPI"].isin(excluded_npis)
        & provider_df_filtered.apply(
            lambda r: r["_year"] == npi_year.get(r["Rndrng_NPI"])
            and r["_state"] == npi_state.get(r["Rndrng_NPI"]),
            axis=1,
        )
    ]
    print(f"  Excluded rows after dedup: {len(excl_rows)} (unique NPIs: {excl_rows['Rndrng_NPI'].nunique()})")
    all_rows = pd.concat([excl_rows, peer_df], ignore_index=True)

    print("\n[5/6] Features, stats, visualizations ...")
    features_df = compute_features(all_rows, service_df, excluded_npis)
    comparison_df = compare_groups(features_df)
    figure_paths = create_visualizations(features_df, comparison_df)

    print("\n[6/6] Prosecution matching ...")
    prosecution_df = match_prosecutions(cohort_df)

    generate_report(cohort_df, provider_df, comparison_df, features_df, figure_paths,
                    prosecution_df)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
