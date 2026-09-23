# hq-risk-baselines

Clean baselines and attack sweeps for HQ-Risk scenarios **S1 (classical ML)**, **S2 (federated learning)** and **S3 (QML)**.
Reproducible, multi-seed, config-driven, runnable unattended on an HPC cluster.
S3 uses the project lead's `QuantumNet` Qiskit circuit, trained with Adam, evaluated on Qiskit Aer with shots.

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
python scripts/run_sweep.py --config configs/s3_circuit_tamper.yaml
python scripts/run_sweep.py --config configs/s3_shot_bias.yaml
python scripts/run_sweep.py --config configs/s3_noise.yaml
python scripts/verify_quantum.py          # torch simulator == Qiskit Statevector, training reproducible
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
python scripts/run_single.py --scenario s3 --dataset breast_cancer --model quantum_net --attack shot_bias --attack-mode forged_bitstring --intensity 0.2 --seed 0
```

`--dry-run` on `run_sweep.py` lists the runs without executing them.

## Results explorer

`explorer/index.html` is a static page for browsing every sweep and the reference-system table interactively. Live: https://hq-risk-explorer.netlify.app
Refresh its data after new runs, then open the file or deploy the folder (`netlify.toml` publishes `explorer/`):

```bash
python scripts/build_explorer.py          # explorer/data.js from summary.csv + runs.csv
```

## HPC

```bash
sbatch scripts/submit_hpc.sbatch                 # array job, one shard per task, one CSV per shard
python scripts/merge_shards.py                   # append shard CSVs into results/raw/runs.csv
python scripts/build_summary.py
```

Cluster name, partition, account and module loads are unknown; the sbatch file has placeholders.

## Changing an experiment

Every knob is a YAML value in `configs/`. Adding an intensity, a seed, or a dataset is a YAML edit, not a Python edit.
Every implementer default (D1–D40) is a single value and listed in `notes/METHODS.md` for veto.

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
  attacks/          LabelFlipAttack, FgsmAttack, ByzantineBehaviour, CircuitTamperAttack, ShotBiasAttack, BackendNoiseAttack
  federated/        DataPartitioner, FederatedClient, FederatedServer
  quantum/          QuantumNet (the circuit), TorchStatevector (exact gradients), QuantumTrainer (Adam),
                    PcaAngleEncoder, ShotExecutor (Aer sampling -- where S3 attacks act)
scripts/            run_single, run_sweep, build_summary, make_figures, build_reference_table, merge_shards, verify_quantum, submit_hpc.sbatch
explorer/           static results explorer (index.html + generated data.js)
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
`attack_success_rate` is blank except for evasion and S3 (decisions D10, D35). For S3, intensity is injected gate count,
forged-shot fraction, or noise probability. `run_id` is a hash of the full configuration
including the seed, so re-running a point produces the same id; `build_summary.py` keeps the latest row per id.
