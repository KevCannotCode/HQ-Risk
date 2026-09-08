# hq-risk-baselines

Clean baselines and attack sweeps for HQ-Risk scenarios **S1 (classical ML)** and **S2 (federated learning)**.
Reproducible, multi-seed, config-driven, runnable unattended on an HPC cluster.
S3 (quantum) is out of scope until the circuit is supplied.

Context: `../HQ-Risk-CONTEXT.md`. Task spec: `../HQ-Risk-TASK-S1-S2.md`. What was run and every assumption: `notes/METHODS.md`.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

One command per sweep. Each appends to `results/raw/runs.csv`, never overwrites.

```bash
python scripts/run_sweep.py --config configs/s1_poisoning.yaml
python scripts/run_sweep.py --config configs/s1_poisoning_targeted.yaml
python scripts/run_sweep.py --config configs/s1_evasion.yaml
python scripts/run_sweep.py --config configs/s2_federated.yaml
python scripts/run_sweep.py --config configs/s2_federated_signflip.yaml
```

Then aggregate, plot, and fill the project lead's table:

```bash
python scripts/build_summary.py        # results/summary/summary.csv  (mean, sd, min, max, n per configuration)
python scripts/make_figures.py         # results/figures/*.png
python scripts/build_reference_table.py   # results/summary/reference_table.md
```

One point, for debugging:

```bash
python scripts/run_single.py --scenario s1 --dataset breast_cancer --attack poisoning --attack-mode symmetric --intensity 0.2 --seed 0
python scripts/run_single.py --scenario s2 --dataset digits --attack byzantine --attack-mode label_flip --intensity 0.3 --partition non_iid --seed 0
```

`--dry-run` on `run_sweep.py` lists the runs without executing them.

## HPC

```bash
sbatch scripts/submit_hpc.sbatch                 # array job, one shard per task, one CSV per shard
python scripts/merge_shards.py                   # append shard CSVs into results/raw/runs.csv
python scripts/build_summary.py
```

Cluster name, partition, account and module loads are unknown; the sbatch file has placeholders.

## Changing an experiment

Every knob is a YAML value in `configs/`. Adding an intensity, a seed, or a dataset is a YAML edit, not a Python edit.
Every implementer default (D1–D30) is a single value and listed in `notes/METHODS.md` for veto.

## Layout

```
configs/            one YAML per sweep
src/
  datasets.py       DatasetLoader   stratified split, scaler fitted on train only
  models.py         ModelFactory    scikit-learn estimators for S1, SoftmaxRegression for S2
  metrics.py        MetricsCalculator
  storage.py        ResultWriter    append-only CSV, fixed schema
  experiment.py     ExperimentRunner, RunConfig   one configuration -> one row
  sweep.py          SweepConfig     YAML -> list of RunConfig
  attacks/          LabelFlipAttack, FgsmAttack, ByzantineBehaviour
  federated/        DataPartitioner, FederatedClient, FederatedServer
scripts/            run_single, run_sweep, build_summary, make_figures, build_reference_table, merge_shards, submit_hpc.sbatch
results/            raw/runs.csv, summary/summary.csv, summary/reference_table.md, figures/
notes/METHODS.md    what was run, every decision
```

## Results schema

One row per run in `results/raw/runs.csv`:

```
run_id, timestamp, scenario, dataset, model, attack, attack_mode, intensity,
seed, n_clients, partition, rounds,
clean_accuracy, attacked_accuracy, accuracy_drop_pp,
attack_success_rate, balanced_accuracy, positive_recall, f1,
train_seconds, git_commit, config_hash
```

`intensity` is the sweep value (poisoning fraction, ε, or malicious client fraction); `0` is the clean baseline.
`attack_success_rate` is blank except for evasion (decision D10). `run_id` is a hash of the full configuration
including the seed, so re-running a point produces the same id; `build_summary.py` keeps the latest row per id.
