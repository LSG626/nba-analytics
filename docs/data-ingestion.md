# Data ingestion and quality report

The ingestion pipeline is intentionally schema-flexible. It reads all
top-level `*.csv` files in its input directory independently and writes
`<source-name>.parquet` files. It does not join tables, assume a specific NBA
provider's schema, or infer missing player/team identities.

After processing, `data/processed/data_quality_report.json` contains one
section per source with source and processed row counts, exact duplicate rows
removed, original and processed types, per-column missing-value counts,
unparseable date counts, and checks for available player IDs, team IDs, and
score fields. Fields absent from a CSV are not treated as errors.

Use `--input-dir`, `--output-dir`, and `--report` to customize paths. A failed
CSV is recorded in the report; the command exits nonzero if any input could not
be processed.
