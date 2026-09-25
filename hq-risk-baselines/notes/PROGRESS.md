# Progress log

What has been built so far, newest first. Decisions and assumptions are in `METHODS.md`; this file is the plain-language history.

Live site: https://hq-risk-explorer.netlify.app

---

## 2026-09-25 — Tables, drill-through, data sources

**Asked for:** tables on the live site; clicking a number should open its summary table; show and serve the data sources;
tables in a format usable in the paper (reference: the Cureus QFML/blockchain EHR review); less wordy UI; check the backend.

**Backend checked first (nothing changed in the experiments):**
- `build_summary.py` rebuilt `summary.csv` from `runs.csv`: byte-identical.
- `build_explorer.py` and `build_reference_table.py` rebuilt their outputs: byte-identical.
- `verify_quantum.py`: torch simulator equals Qiskit Statevector (max error 6e-16), training twice gives identical weights.
- Re-ran 4 stored experiments from scratch (S1 digits backdoor, S2 wine targeted-flip non-IID, S3 wine transpiler drift,
  S1 membership inference). Every metric matched the stored row exactly (difference 0.0).

**Added:**
- `scripts/sweep_catalog.py`: one place for each sweep's title, intensity label and headline metrics. The paper tables and
  the site both read it, so a label is written once.
- `scripts/build_tables.py`: 23 paper tables in `results/tables/`:
  - Tables 1–3: reference systems (every attack at its strongest point) for Breast Cancer, Digits and Wine.
  - Tables 4–23: one per sweep. Rows are attack strengths; columns are dataset × metric; cells show mean ± SD over 5 seeds.
  - Each table has a CSV file plus an entry in `tables.md`. The layout follows Cureus: the caption goes below the table as
    "TABLE n: …", then a note line, then an abbreviations line (only the abbreviations that table uses).
- `scripts/check_results.py`: one command that fails if anything drifted. It checks that summary matches runs, that every
  configuration has 5 seeds, that the catalogue covers every sweep, that the table CSVs match the summary, that the site
  data matches, that every served file exists, and that every clickable table cell points at a real configuration.
  It passes now.
- `build_explorer.py` now also embeds the tables, the sweep catalogue, and each run's `run_id`. It copies `runs.csv`,
  `summary.csv`, the 23 table CSVs and `tables.md` into `explorer/data/` so the site serves them. Each copy is listed with
  its row count and SHA-256.

**Site changes:**
- **Drill-through.** Clicking any number opens a panel for that exact configuration. This works on a chart point, a value
  in a sweep's "Numbers" table, a reference-system value, or a paper-table cell. The panel shows:
  - all 7 metrics (mean, SD, min, max);
  - the 5 per-seed rows with their `run_id`;
  - the source file and the exact filter that selects those rows;
  - a CSV download of just those rows.
- **Tables section.** Pick any of the 23 tables. It renders in paper style. You can download it as CSV, copy it as
  tab-separated text (pastes straight into Word or Excel with the caption), or get all tables as one `.md` file.
- **Data sources section.** Every result file is downloadable, with its row count, size, hash and GitHub link. The three
  input datasets are listed with row, feature and class counts, the scikit-learn loader, and the UCI source link.
- **Copy cut.** Section intros are now one line each, and sweep notes are one sentence.
- Tested in headless Chromium on desktop (1280 px) and phone (390 px) widths: no console errors, no horizontal scroll.
  All 1,068 clickable table cells open a 5-seed breakdown.

## 2026-09-24 — Full framework coverage (commits f002bf1, 714b91b)

Acted on the project lead's review:
- Added the Digits dataset to QML (S3) and the Wine dataset to federated learning (S2).
- Added targeted poisoning.
- Added ARF attacks: backdoor, steal (model extraction), and infer (membership inference and model inversion).
- Added QTF-TSV attacks through a malicious Qiskit transpiler pass: angle drift and qubit swap.
- Total: 2,130 runs, 20 sweeps × 3 datasets, 5 seeds each. Decisions D41–D52.
- The ARF, QTF and TSV definitions are this implementation's reading of the review and still need the project lead's
  confirmation.

## 2026-09-23 — S3 QML and the explorer (commits 3b2add5, cb12e3f, 13c4092, 4a8d6e5)

- S3 uses the project lead's QuantumNet circuit, trained with Adam using exact statevector gradients and evaluated on
  Qiskit Aer at 256 shots.
- Three S3 attacks: circuit tampering, shot manipulation and noise. Decisions D31–D40.
- Static results explorer, deployed on Netlify.

## 2026-09-07 — S1 and S2 baselines (commits d2aeab4 … a94cd5a)

- Classical ML (S1) and federated FedAvg (S2) harness covering poisoning, FGSM evasion and Byzantine clients. 375 runs.
- Decisions D1–D30. The one-pager was published.

---

## Open items

- The project lead's veto or confirmation of D1–D52, especially the ARF/QTF/TSV definitions (see METHODS §6).
- HPC cluster details: the sbatch file still has placeholders, and everything has run locally so far.
- The HQ-Risk score itself is not computed here; its column in the reference table is left blank on purpose.
