# Medicare Fraud Backtest POC

Proof-of-concept backtest testing whether excluded Medicare providers show detectable billing differences from peers in the year before exclusion.

## Results

**289 excluded providers** matched to pre-exclusion Part B billing data across all US states, compared against **3.39 million peers** (same state, specialty, year). **14 of 15 features statistically significant** after Bonferroni correction.

Strongest signals:
- **Dual-eligible share** (d=+0.78) — excluded providers serve far more Medicaid-Medicare dual-eligible beneficiaries
- **Top HCPCS share** (d=+0.66) — 30% of billing from a single procedure code vs. 16% for peers
- **HCPCS Herfindahl** (d=+0.52) — less diverse service mix overall
- **Beneficiary avg age** (d=−0.31) — younger patient panels
- **Avg charge per service** (d=−0.23) — lower charges, higher volume pattern
- **Beneficiary avg risk** (d=+0.18) — higher-risk patients (only visible after filtering to fraud-specific exclusions)

Full results in [`report.md`](report.md).

## Exclusion Type Filtering

The cohort is restricted to fraud-related exclusion types under [Section 1128 of the Social Security Act](https://oig.hhs.gov/exclusions/authorities.asp):

| Code | Description | Count |
|------|-------------|-------|
| §1128(a)(1) | Program fraud conviction | 1,221 |
| §1128(a)(3) | Felony healthcare fraud conviction | 300 |
| §1128(b)(7) | Excessive claims / unnecessary services | 78 |

Excluded from the cohort:
- **§1128(a)(4)** — felony controlled substance convictions (drug crimes, not billing patterns)
- **§1128(b)(4)** — license revocations (administrative, could be anything)
- **§1128(a)(2)** — patient abuse/neglect convictions

This filtering matters. With all exclusion types mixed in, `bene_avg_risk` showed zero signal (d=0.00). After restricting to fraud-specific types, it became significant (d=+0.18) — license revocations and drug felonies were masking a real billing-fraud signal.

## Iteration History

The pipeline went through several iterations to reach the current design:

1. **DMEPOS + NPPES** — zero NPI overlap. DMEPOS data is 99.5% organizational NPIs; LEIE exclusions are individuals. Entity types don't align.
2. **Part B, Florida only, all exclusion types** — 28 matched providers. Signal present but underpowered.
3. **Part B, 10 states, all exclusion types** — 346 providers, 12/15 features significant. Data duplication bug found and fixed (excluded NPIs appearing in multiple state/year datasets).
4. **Part B, 10 states, §1128(a)(1) + §1128(b)(7) only** — 140 providers, 6/15 significant. Cleaner cohort but lost statistical power.
5. **Part B, all states, §1128(a)(1) + §1128(b)(7) only** — 245 providers, 10/15 significant. Power recovered, but concentration metrics weaker.
6. **Part B, all states, §1128(a)(1) + §1128(a)(3) + §1128(b)(7)** — 289 providers, 14/15 significant. Adding healthcare fraud felonies back in recovered concentration signal while keeping the clean cohort.

## Data Sources

| Source | URL | Purpose |
|--------|-----|---------|
| LEIE | https://oig.hhs.gov/exclusions/downloadables/UPDATED.csv | Excluded provider labels |
| CMS Part B Provider | https://data.cms.gov (year-specific UUIDs in script) | Provider-level billing |
| CMS Part B Service | https://data.cms.gov (year-specific UUIDs in script) | HCPCS-level billing |

All data is publicly available, no API keys required.

## Usage

```bash
pip install pandas scipy matplotlib requests pyarrow
python backtest.py
```

First run downloads ~4GB of CMS data for all states (cached to `data/` as parquet). Subsequent runs use cache. Outputs: `report.md` and PNGs in `figures/`.

## Pipeline

1. Download and cache LEIE (83K+ records)
2. Build excluded cohort — all US states, 2018–2023 exclusions, fraud-specific types (§1128(a)(1), §1128(a)(3), §1128(b)(7)), valid NPIs → 1,605 providers
3. Find best billing year per provider — walk backward through CMS Part B (2022→2017) → 289 matched
4. Download billing data — provider-level and service-level, cached to parquet
5. Build peer groups — same state, same specialty, same year, ≥11 beneficiaries
6. Compute 15 features — volume, intensity, concentration, demographics
7. Statistical comparison — Mann-Whitney U, Welch's t-test, Cohen's d, Bonferroni correction
8. Visualizations and report

## Caveats

- 82% of excluded NPIs didn't match any CMS Part B data (program siloing, suppression thresholds, NPI mismatches)
- Peer matching by state/specialty/year — not by practice size or sub-state geography
- Effect sizes are population-level, not individual-level diagnostic
- Billing years range from 2017–2022; Medicare norms shifted during COVID-19 (peer-matching by year mitigates but doesn't eliminate)
