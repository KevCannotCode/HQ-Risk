# METHODS — what was run, and every assumption

**Written:** 2026-09-07. **Scope:** S1 (classical ML) and S2 (federated learning) baselines and attack sweeps.
**S3 (quantum):** not run. Blocked on the circuit. Nothing stubbed, nothing guessed.

Every decision below is **an implementer default, not an instruction from the project lead**. Each is one config value or one
constant; reversing any of them is a one-line change followed by a re-run. **Please veto.**

---

## 1. Environment

| Item | Value |
|---|---|
| Python | 3.12.3 |
| scikit-learn | 1.9.0 |
| numpy | 2.5.3 |
| pandas | 3.0.5 |
| Hardware | single CPU, local machine (the HPC cluster is not yet available — see blockers) |
| Data | `sklearn.datasets` bundled copies of Breast Cancer (569×30, 2 classes), Digits (1797×64, 10 classes), Wine (178×13, 3 classes) |

---

## 2. Experiment matrix as executed

| Sweep | Config | Datasets | Intensities | Seeds | Runs |
|---|---|---|---|---|---|
| S1 poisoning, symmetric | `configs/s1_poisoning.yaml` | Breast Cancer, Digits, Wine | 0, 5, 10, 20, 30, 40 % | 0–4 | 90 |
| S1 poisoning, targeted | `configs/s1_poisoning_targeted.yaml` | Breast Cancer | 0, 5, 10, 20, 30, 40 % | 0–4 | 30 |
| S1 evasion, FGSM | `configs/s1_evasion.yaml` | Breast Cancer, Digits, Wine | ε = 0, 0.1, 0.25, 0.5, 1.0 | 0–4 | 75 |
| S2 federated, label-flip | `configs/s2_federated.yaml` | Breast Cancer, Digits × {IID, non-IID} | 0, 10, 20, 30, 40, 50 % clients | 0–4 | 120 |
| S2 federated, sign-flip | `configs/s2_federated_signflip.yaml` | Breast Cancer × {IID, non-IID} | 0, 10, 20, 30, 40, 50 % clients | 0–4 | 60 |
| **Total** | | | | | **375** |

Every run trains its own clean model on the same split and seed, so each row is self-contained and every
clean-vs-attacked comparison is paired (same draw, only the attack differs).

---

## 3. Decisions from the task spec (D1–D18)

| # | Decision | Value used | Where |
|---|---|---|---|
| D1 | Classifier | `LogisticRegression(max_iter=5000, random_state=seed)` | `src/models.py` |
| D2 | Secondary model | `MLPClassifier(hidden_layer_sizes=(64,), relu, max_iter=2000)` — implemented, FGSM gradient implemented and verified, **not run in the main matrix** | `src/models.py`, `model:` in YAML |
| D3 | Feature scaling | `StandardScaler` fitted on train only; asserted (`n_samples_seen_ == len(x_train)`) | `src/datasets.py` |
| D4 | Split | 80 / 20, stratified on label | `src/datasets.py` |
| D5 | Seeds | 0, 1, 2, 3, 4 — same five for every configuration | `seeds:` in YAML |
| D6 | Poisoning target | training set only, after the split; test set fingerprinted before the run and asserted unchanged after | `src/experiment.py` |
| D7 | Poisoning mode | `symmetric` primary; `targeted` (malignant→benign) secondary on Breast Cancer | `attack_mode:` in YAML |
| D8 | Poisoning denominator | fraction of **training rows** | `src/attacks/label_flip.py` |
| D9 | Evasion | FGSM on the test set, white-box, ε ∈ {0.1, 0.25, 0.5, 1.0} in standardised units | `src/attacks/fgsm.py` |
| D10 | Attack success rate | evasion only: fraction of test rows correct before and wrong after. **Blank for poisoning and federated** until defined | `src/metrics.py` |
| D11 | Metrics | accuracy, balanced accuracy, positive-class recall, F1, accuracy drop (pp) | `src/metrics.py` |
| D12 | S2 client count | 10 | `federated.n_clients` |
| D13 | S2 rounds / local epochs | 30 rounds, 1 local epoch, all clients every round | `federated.rounds`, `federated.local_epochs` |
| D14 | S2 aggregation | FedAvg weighted by client sample count, no defence | `src/federated/server.py` |
| D15 | non-IID | Dirichlet label skew, α = 0.5 | `federated.dirichlet_alpha` |
| D16 | Malicious client | `label_flip` primary (trains honestly on flipped labels); `sign_flip` secondary (sends the negated update) | `src/attacks/byzantine.py` |
| D17 | S2 evaluation | global model on the central held-out test set, same split seed as S1 | `src/experiment.py` |
| D18 | Data source | `sklearn.datasets` loaders | `src/datasets.py` |

---

## 4. Decisions the spec did not cover, taken during implementation (D19–D30)

| # | Decision | Value used | Why |
|---|---|---|---|
| D19 | Breast Cancer label polarity | remapped so **1 = malignant** (positive class); scikit-learn ships 0 = malignant | D7 and D11 talk about malignant as the class of interest; recall of label 1 is then "malignant recall" |
| D20 | Multiclass metrics | Digits and Wine have no positive class: `positive_recall` and `f1` are **macro-averaged** | Only well-defined choice without picking an arbitrary class |
| D21 | S2 local trainer | `SoftmaxRegression` — multinomial logistic regression trained by mini-batch SGD (batch 32) | scikit-learn's LBFGS solver cannot start from arbitrary global weights. Same hypothesis class as D1; only the optimiser differs |
| D22 | S2 learning rate | 0.3 | Chosen so the clean federated baseline sits closest to the centralised one (Digits: 0.950 at 0.1 → 0.961 at 0.3; Breast Cancer unchanged; centralised 0.967 / 0.974) |
| D23 | Which clients are malicious | client ids 0 … k−1, with k = round(fraction × 10) | IID shards are exchangeable; Dirichlet shard-to-id assignment is already random |
| D24 | Byzantine `label_flip` | the malicious client flips **100 %** of its local labels (each to a uniformly random different class) | Strongest reading of "trains on flipped labels"; keeps one intensity axis (client fraction) |
| D25 | Symmetric flip, multiclass | each poisoned row gets a uniformly random *different* label | Direct generalisation of binary flip |
| D26 | Targeted flip denominator | **fraction of malignant (source-class) training rows**, not of all training rows — this is the one place D8 is not followed | Malignant rows are only ≈ 37 % of the training set, so 40 % of training rows would flip every malignant row and leave a single-class training set (the solver refuses to fit — observed). Under D8 the 40 % point cannot exist; under this reading the whole 5–40 % sweep does |
| D27 | Intensity 0 rows | attacked model = clean model; `attack_success_rate` blank | Clean baseline row exists for every dataset, scenario, mode and partition before any attack row |
| D28 | Federated clean baseline | FedAvg with 0 malicious clients on the same partition — **not** the centralised model | S2 accuracy drop measures the attack, not the cost of federation. The centralised number is on the S1 rows |
| D29 | `train_seconds` | wall time of the whole run (clean + attacked training + evaluation) | Simple, comparable |
| D30 | `git_commit` | short hash of HEAD; `uncommitted` when no repository | Traceability |

---

## 5. Reproducibility

- All randomness comes from `numpy.random.default_rng([seed, k])` streams keyed by seed and role (partition, client id, poisoning row choice) or from `random_state=seed` in scikit-learn. No global RNG.
- `run_id` = hash of the full configuration incl. seed; `config_hash` = the same without the seed. Re-running a point yields a byte-identical row apart from `timestamp` and `train_seconds`. Verified.
- FGSM gradients for both models were checked against central finite differences (max abs error ≈ 1e-11) on Breast Cancer (binary) and Digits (multiclass).
- Test set: SHA-256 fingerprint of `(x_test, y_test)` taken before and asserted after every run.

---

## 6. Blockers still open (not mine to resolve)

| Blocker | Owner | Effect |
|---|---|---|
| HPC cluster name, account, scheduler partition, module policy | the project lead | `scripts/submit_hpc.sbatch` has placeholders; everything ran locally instead |
| Attack success rate for poisoning / federated | the project lead | one column blank on 300 of 375 rows |
| Whether the formula consumes accuracy drop (circularity) | the project lead | decides whether these numbers can validate the score at all |
| QML circuit | the project lead | S3 not started |

---

## 7. Headline numbers (5 seeds, mean ± sd; full table in `results/summary/summary.csv`)

**Clean baselines (centralised, S1):** Breast Cancer 97.0 ± 2.0 %, Digits 97.0 ± 0.7 %, Wine 98.9 ± 1.5 %.
**Clean baselines (federated, S2, 0 malicious):** Breast Cancer IID 97.4 ± 1.4 %, non-IID 96.0 ± 2.5 %; Digits IID 95.9 ± 0.5 %, non-IID 96.4 ± 0.4 %.

| Sweep | Dataset | Weakest point | Strongest point (max intensity) |
|---|---|---|---|
| S1 symmetric poisoning | Breast Cancer | 5 %: −1.9 ± 1.9 pp | 40 %: −22.6 ± 2.7 pp |
| | Digits | 5 %: −4.2 ± 1.5 pp | 40 %: −11.2 ± 0.8 pp |
| | Wine | 5 %: −1.7 ± 4.2 pp | 40 %: −13.3 ± 6.0 pp (30 %: −15.0 ± 3.2 pp) |
| S1 targeted poisoning | Breast Cancer | 5 %: −1.9 ± 1.1 pp, malignant recall 0.88 | 40 %: −16.3 ± 2.4 pp, **malignant recall 0.48** (clean 0.94) |
| S1 FGSM evasion | Breast Cancer | ε 0.1: −7.2 ± 2.6 pp, ASR 7 % | ε 1.0: −90.9 ± 3.4 pp, ASR 94 % |
| | Digits | ε 0.1: −10.8 ± 1.4 pp, ASR 11 % | ε 1.0: −97.0 ± 0.7 pp, ASR 100 % |
| | Wine | ε 0.1: −2.2 ± 2.3 pp, ASR 2 % | ε 1.0: −94.4 ± 2.0 pp, ASR 96 % |
| S2 label-flip clients | Breast Cancer IID | 10 %: −1.4 ± 1.0 pp | 50 %: −44.9 ± 8.5 pp |
| | Breast Cancer non-IID | 10 %: −1.2 ± 2.8 pp | 50 %: −55.1 ± 33.7 pp |
| | Digits IID | 10 %: −0.5 ± 0.8 pp | 50 %: −8.2 ± 1.6 pp |
| | Digits non-IID | 10 %: −0.6 ± 0.5 pp | 50 %: −19.8 ± 8.8 pp |
| S2 sign-flip clients | Breast Cancer IID | 10 %: −1.8 ± 1.4 pp | 50 %: −70.4 ± 17.3 pp |
| | Breast Cancer non-IID | 10 %: 0.0 ± 1.1 pp | 50 %: −64.6 ± 39.7 pp |

the project lead's reference-system table with these numbers: `results/summary/reference_table.md`. Figures: `results/figures/`.

---

## 8. Observations to raise (not conclusions)

1. **Monotonicity holds in every sweep at the level of the mean**, with two within-noise exceptions: Wine symmetric poisoning 40 % (85.6 %) vs 30 % (83.9 %) — Wine's test set is 36 rows, one row is 2.8 pp; and Breast Cancer IID sign-flip 20 % (96.8 %) vs 10 % (95.6 %). Both differences are smaller than their sd.
2. **Accuracy hides the targeted attack.** At 40 % targeted poisoning, accuracy drops 16 pp but malignant recall falls from 0.94 to 0.48 — half of the malignant test cases are missed. This is the "imbalanced medical data" concern from the context brief, now measured.
3. **Non-IID federated results are bimodal, not noisy.** Breast Cancer non-IID at 40–50 % malicious clients has sd 0.32–0.47: some seeds keep ~95 % accuracy, others collapse to ~35 %. Cause: with Dirichlet α = 0.5 and 10 clients, shard sizes are very uneven, and D23 makes clients 0…k−1 malicious regardless of size — whether the malicious clients hold most of the data is decided by the partition draw. Mean ± sd is the wrong summary here; `summary.csv` carries min and max. Two options for the project lead: (a) keep as is and report min/max, (b) choose malicious clients by a seeded random draw or by data share — one line in `experiment.py`.
4. **Federated attacks below 40 % barely register on IID data** (≤ 3 pp on both datasets). FedAvg weighted by sample count dilutes a minority of label-flipping clients. The interesting region is 40–50 %, which is also where the theoretical guarantees of plain FedAvg end.
5. **FGSM at ε = 1.0 is total** on every dataset (≥ 94 % ASR). One standard deviation per feature is a very large perturbation for tabular data; ε = 0.1–0.25 is the informative range. Worth asking whether ε = 1.0 is meant to be a ceiling point or a realistic one.
6. **Federated clean baseline ≈ centralised baseline** (within 1.5 pp on every dataset/partition), so S2 accuracy drops are attributable to the attack, not to federation itself.
