# Medicare Fraud Backtest POC

Proof-of-concept backtest testing whether excluded Medicare providers show detectable billing differences from peers in the year before exclusion.

## Results

**346 excluded providers** matched to pre-exclusion Part B billing data across ten states, compared against **2.86 million peers** (same state, specialty, year). **12 of 15 features statistically significant** after Bonferroni correction.

Strongest signals:
- **Dual-eligible share** (d=+0.75) — excluded providers serve far more Medicaid-Medicare dual-eligible beneficiaries
- **Top HCPCS share** (d=+0.71) — 30% of billing from a single procedure code vs. 16% for peers
- **HCPCS Herfindahl** (d=+0.59) — less diverse service mix overall

Full results in [`report.md`](report.md).

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

First run downloads ~2GB of CMS data (cached to `data/` as parquet). Subsequent runs use cache. Outputs: `report.md` and PNGs in `figures/`.

## Pipeline

1. Download and cache LEIE (83K+ records)
2. Build excluded cohort — top-10 states, 2020–2023 exclusions, valid NPIs → 1,237 providers
3. Find best billing year per provider — walk backward through CMS Part B (2022→2017) → 346 matched
4. Download billing data — provider-level and service-level, cached to parquet
5. Build peer groups — same state, same specialty, same year, ≥11 beneficiaries
6. Compute 15 features — volume, intensity, concentration, demographics
7. Statistical comparison — Mann-Whitney U, Welch's t-test, Cohen's d, Bonferroni correction
8. Visualizations and report

## Caveats

- 72% of excluded NPIs didn't match any CMS Part B data (program siloing, suppression thresholds, NPI mismatches)
- Mixed exclusion types (fraud convictions, license revocations, excessive services) treated as one group
- Peer matching by state/specialty/year — not by practice size or sub-state geography
- Effect sizes are population-level, not individual-level diagnostic
