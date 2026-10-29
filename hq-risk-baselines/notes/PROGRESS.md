# Progress log

What has been built so far, newest first. Decisions and assumptions are in `METHODS.md`; this file is the plain-language history.

Live site: https://hq-risk-explorer.netlify.app

---

## 2026-10-29 — Validation phase: 20 more seeds, controls, and combined attacks (C1–C4)

**Completed 12:48 (8 local workers, 1 h 33 min):** 13,590 runs asked, 13,590 done, no worker errors. `runs.csv` 10,650 distinct runs (426 configurations × 25 seeds, 72 of them intensity-0 controls with a 0.0 pp drop on every run), `combined.csv` 2,940 runs (147 configurations × 20 seeds). `check_results.py` passes without `--partial`: 28 tables, explorer data, 33 served files. Paper text in `notes/paper/sections.md` moved to 25 seeds and the new counts, with a Setup paragraph and a Results paragraph on C1–C4; Word draft regenerated as `update-10-8/fabrice/HQ-Risk Draft paper-Kevin-validation.docx`.

**Asked for (project lead, email "HQ Risk Paper Update"):** rerun every configuration on 20 new seeds (101–119, "20
seeds total"), rerun the intensity-0 controls on the same seeds, then a second phase of combined attacks: C1 poisoning ×
FGSM (S1), C2 sign-flip clients × FGSM × partition (S2), C3 transpiler drift × forged shots × noise (S3), C4 backdoor ×
circuit tampering (S3). He also notes that E (exploitability) and I (technical impact) can now be computed from the results
instead of being fixed at 1, and offers the HPC server if the runs are too heavy.

**Done:**
- Seeds 101–120 added to all 20 sweep files (D53: his range is 19 seeds, his counts need 20). Dry run: 8,520 new runs,
  the exact total in his email (7,080 attack + 1,440 control).
- Combined attacks: a run can now carry several factors (`attack: combined` + a `factors` list in the YAML). Four new
  configs `c1_…`–`c4_…` give 27 + 54 + 54 + 12 = 147 cells, the same counts as his table, on seeds 101–120 (D54–D58).
  New code paths: poisoned-model FGSM (S1), FGSM against the federated global model (S2), drift→noise→forged-shots and
  backdoor→tamper (S3). Smoke-tested on Wine, seed 101; a stored noise run was re-executed and matched exactly.
- Workers: `scripts/run_validation.sh` splits every sweep by (dataset, partition, seed) across 8 local processes, so the
  quantum clean model is trained once per group; `scripts/validation_progress.py` shows rows done per sweep.
  Estimated wall time on this machine: about 2 hours (the 5-seed matrix took 2.5 CPU-hours, almost all of it S3).
- Summary, tables, explorer and the drift check now read the seed count from the data (`n` column) instead of assuming 5,
  and `check_results.py --partial` lets the site deploy while runs are still in flight. Tables 25–28 (one per block) are
  built from `combined_summary.csv` once it exists.

**Open:** the E and I mapping (which column feeds which term) is the lead's to confirm; the proposal in
`notes/paper/sections.md` (Validation protocol) still stands. Whether phase 2 should also use seeds 0–4.

## 2026-10-08 — Setup and Results for the 6-page paper

**Asked for (project lead, Word draft + WhatsApp):** fill the Setup section of the IEEE draft (how the experiments were run,
datasets, configurations), short, the whole paper is capped at 6 pages with references, Word required. The lead applies the
HQ-Risk formula himself once the results are in. Deadline for the conference is 30 December.

**Added:**
- `results/tables/table_24_compact.csv`: one number per attack and dataset at the strongest point of each sweep (clean
  accuracy, or drop when success is undefined, or success rate). 23 rows × 3 datasets; fits one IEEE column. The 23 Cureus
  tables stay for the explorer and the journal version.
- `notes/paper/sections.md`: the Setup text (4 paragraphs), a Results paragraph that reads table 24, and a comment for the
  lead about two attacks missing from his Table I (model inversion, federated backdoor).
- `scripts/fill_paper_docx.py`: pastes the sections under the matching headings of the Word draft, inserts table 24 under
  Results in the template's `table head` / `table footnote` styles, and anchors the comment on Table I. The draft is Strict
  OOXML, so the script rewrites the namespaces to Transitional before python-docx reads it. The draft itself stays outside
  the repo.
- `check_results.py` still passes; the explorer now lists 24 tables.
- Later the same day: the two missing attacks were written into Table I directly (`## Cell:` instruction in `sections.md`),
  and short drafts marked "[Draft by Kevin …]" were added under Validation protocol, Discussion and Limitations, and
  Conclusion. The validation text proposes how Table II feeds the formula's inputs; the lead decides.

**Open:** the lead's formula weights (w, β, α, λ, μ) are symbolic in the draft, so no HQ-Risk score is computed here.

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


### Next task (2026-10-29, after the validation runs finish)

1. Rebuild: `build_summary.py` (runs.csv), `build_summary.py --runs results/raw/combined.csv --output results/summary/combined_summary.csv`,
   `build_tables.py`, `build_explorer.py`, then `check_results.py` without `--partial`.
2. Paper text (`notes/paper/sections.md`): "5 seeds" → 25 seeds (0–4 and 101–120); run counts 2,130 → 10,650 single-attack + 2,940 combined;
   one sentence each on the intensity-0 controls and the combined attacks C1–C4 (tables 25–28).
3. Word draft: regenerate from the lead's original with `fill_paper_docx.py --draft "../update-10-8/fabrice/HQ-Risk Draft paper-Init.docx"
   --out "../update-10-8/fabrice/HQ-Risk Draft paper-Kevin-validation.docx" --author Kevin`; keep the 10-08 copy untouched.
4. Commit results + text on `feature/validation-runs`, push, open PR to `main`, merge once the test job is green.
5. Deploy the explorer: `netlify deploy --prod --dir explorer --site 54039fc9-496a-46e3-9f98-8bd96a78cb61`.
6. Create a separate Gmail draft to the lead for HQ-Risk only (the SoK draft "SoK: Section VI (Defense Landscape) draft" already
   exists): what was run, final run count, GitHub and site links, the Word draft attached by Kevin; Kevin sends it himself.
7. Still open with the lead: seeds 101–120 vs his "101–119"; same 20 seeds for C1–C4; how E and I map to the new numbers.

- The project lead's veto or confirmation of D1–D52, especially the ARF/QTF/TSV definitions (see METHODS §6).
- HPC cluster details: the sbatch file still has placeholders, and everything has run locally so far.
- The HQ-Risk score itself is not computed here; its column in the reference table is left blank on purpose.
- The SoK paper (quantum circuit partitioning, attack-vs-defence matrix, multi-layer attack) is a separate task; nothing of it
  lives in this repo yet.
