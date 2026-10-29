"""One description per sweep: title, intensity meaning, headline metrics. Shared by the paper tables and the explorer,
so a label is written once."""

CORE = "Core attacks"
ARF = "ARF · adversarial-AI factor"
QTF = "QTF · quantum-threat factor"

DATASETS = {"breast_cancer": "Breast Cancer", "digits": "Digits", "wine": "Wine"}
PARTITIONS = {"iid": "IID", "non_iid": "non-IID"}

TARGETED = "source-class rows sent to the target class"
BACKDOOR = "triggered rows classified as the target class"
FLIPPED = "correct predictions turned wrong"

# metrics: which summary columns the paper table shows, in order. unit "%" scales by 100, "pp" is already in points.
SWEEPS = [
    dict(id="s1_poisoning", group=CORE, scenario="s1", attack="poisoning", mode="symmetric",
         title="S1 label-flip poisoning", x="Poisoning rate", x_unit="fraction",
         success=None, metrics=["attacked_accuracy", "accuracy_drop_pp"],
         note="Fraction of training labels flipped to a random other class."),
    dict(id="s1_poisoning_targeted", group=CORE, scenario="s1", attack="poisoning", mode="targeted",
         title="S1 targeted poisoning", x="Source-class rows flipped", x_unit="fraction",
         success=TARGETED, metrics=["attacked_accuracy", "attack_success_rate"],
         note="Source-class training rows relabelled as the target class (malignant → benign; class 0 → 1)."),
    dict(id="s1_evasion", group=CORE, scenario="s1", attack="evasion", mode="fgsm",
         title="S1 FGSM evasion", x="FGSM ε", x_unit="number",
         success=FLIPPED, metrics=["attacked_accuracy", "attack_success_rate"],
         note="White-box FGSM on standardised features; ε = 1 is one standard deviation."),
    dict(id="s2_federated", group=CORE, scenario="s2", attack="byzantine", mode="label_flip",
         title="S2 malicious clients, label flip", x="Malicious clients", x_unit="fraction",
         success=None, metrics=["attacked_accuracy", "accuracy_drop_pp"],
         note="FedAvg, 10 clients, 30 rounds. Malicious clients train on flipped labels."),
    dict(id="s2_federated_signflip", group=CORE, scenario="s2", attack="byzantine", mode="sign_flip",
         title="S2 malicious clients, sign flip", x="Malicious clients", x_unit="fraction",
         success=None, metrics=["attacked_accuracy", "accuracy_drop_pp"],
         note="Malicious clients send the negated update."),
    dict(id="s2_federated_targeted", group=CORE, scenario="s2", attack="byzantine", mode="targeted_flip",
         title="S2 malicious clients, targeted flip", x="Malicious clients", x_unit="fraction",
         success=TARGETED, metrics=["attacked_accuracy", "attack_success_rate"],
         note="Malicious clients relabel their source-class rows as the target class."),
    dict(id="s1_backdoor", group=ARF, scenario="s1", attack="backdoor", mode="trigger",
         title="S1 backdoor", x="Triggered training rows", x_unit="fraction",
         success=BACKDOOR, metrics=["attacked_accuracy", "attack_success_rate"],
         note="Trigger: last 3 features set to +3 SD, rows relabelled as the target class."),
    dict(id="s2_federated_backdoor", group=ARF, scenario="s2", attack="byzantine", mode="backdoor",
         title="S2 backdoor clients", x="Malicious clients", x_unit="fraction",
         success=BACKDOOR, metrics=["attacked_accuracy", "attack_success_rate"],
         note="Each malicious client triggers half its rows and relabels them as the target class."),
    dict(id="s3_backdoor", group=ARF, scenario="s3", attack="backdoor", mode="trigger",
         title="S3 backdoor", x="Triggered training rows", x_unit="fraction",
         success=BACKDOOR, metrics=["attacked_accuracy", "attack_success_rate"],
         note="Same trigger as S1; QuantumNet retrained on the poisoned data."),
    dict(id="s1_extraction", group=ARF, scenario="s1", attack="extraction", mode="label_only",
         title="S1 model extraction", x="Queries", x_unit="count",
         success="fidelity: stolen copy agrees with the victim", metrics=["attacked_accuracy", "attack_success_rate"],
         accuracy="stolen copy's accuracy",
         note="Label-only queries on random inputs; stolen copy is a logistic regression."),
    dict(id="s3_extraction", group=ARF, scenario="s3", attack="extraction", mode="label_only",
         title="S3 model extraction", x="Queries", x_unit="count",
         success="fidelity: stolen copy agrees with the victim", metrics=["attacked_accuracy", "attack_success_rate"],
         accuracy="stolen copy's accuracy",
         note="Every answer is a 256-shot Aer run; stolen copy is a logistic regression."),
    dict(id="s1_membership", group=ARF, scenario="s1", attack="membership", mode="confidence_threshold",
         title="S1 membership inference", x="Known members per side", x_unit="count",
         success="membership attack accuracy (50% = guessing)", metrics=["attack_success_rate"],
         note="Confidence threshold calibrated on known members and non-members."),
    dict(id="s3_membership", group=ARF, scenario="s3", attack="membership", mode="confidence_threshold",
         title="S3 membership inference", x="Known members per side", x_unit="count",
         success="membership attack accuracy (50% = guessing)", metrics=["attack_success_rate"],
         note="Same attack on shot-based class probabilities."),
    dict(id="s1_inversion", group=ARF, scenario="s1", attack="inversion", mode="zeroth_order",
         title="S1 model inversion", x="Queries per class", x_unit="count",
         success="cosine similarity to the true class mean", metrics=["attack_success_rate"],
         note="Black-box hill climb on class probability."),
    dict(id="s3_inversion", group=ARF, scenario="s3", attack="inversion", mode="zeroth_order",
         title="S3 model inversion", x="Queries per class", x_unit="count",
         success="cosine similarity to the true class mean", metrics=["attack_success_rate"],
         note="Same attack; every probability is a 256-shot Aer estimate."),
    dict(id="s3_circuit_tamper", group=QTF, scenario="s3", attack="circuit_tamper", mode="rx_half_pi",
         title="S3 circuit tampering", x="Injected RX(π/2) gates", x_unit="count",
         success=FLIPPED, metrics=["attacked_accuracy", "attack_success_rate"],
         note="Gates inserted at random positions of the trained circuit."),
    dict(id="s3_shot_bias", group=QTF, scenario="s3", attack="shot_bias", mode="forged_bitstring",
         title="S3 shot manipulation", x="Forged shots", x_unit="fraction",
         success=FLIPPED, metrics=["attacked_accuracy", "attack_success_rate"],
         note="Fraction of 256 shots replaced with the most-benign bitstring."),
    dict(id="s3_noise", group=QTF, scenario="s3", attack="noise", mode="depolarizing_readout",
         title="S3 noise increase", x="Error probability p", x_unit="number",
         success=FLIPPED, metrics=["attacked_accuracy", "attack_success_rate"],
         note="Depolarizing p on every gate plus readout flip p."),
    dict(id="s3_transpiler_drift", group=QTF, scenario="s3", attack="transpiler", mode="angle_drift",
         title="S3 transpiler angle drift (TSV)", x="RZ offset (rad)", x_unit="number",
         success=FLIPPED, metrics=["attacked_accuracy", "attack_success_rate"],
         note="Malicious pass in Qiskit's preset pass manager adds an offset to every RZ."),
    dict(id="s3_transpiler_swap", group=QTF, scenario="s3", attack="transpiler", mode="qubit_swap",
         title="S3 transpiler qubit swap (TSV)", x="Inserted SWAPs", x_unit="count",
         success=FLIPPED, metrics=["attacked_accuracy", "attack_success_rate"],
         note="Malicious pass swaps qubits before readout without tracking the layout."),
]

# Combined-attack blocks (second validation phase). Factors are applied in the listed order; see METHODS D53–D57.
COMBINED = [
    dict(id="c1_poison_evasion", block="C1", scenario="s1", title="C1 · symmetric poisoning × FGSM evasion",
         factors=[("poisoning", "symmetric", "Poisoning rate", "fraction"), ("evasion", "fgsm", "FGSM ε", "number")],
         success=FLIPPED, note="Logistic regression trained on the poisoned labels, then FGSM against that model at test time."),
    dict(id="c2_signflip_evasion", block="C2", scenario="s2", title="C2 · sign-flip clients × FGSM evasion",
         factors=[("byzantine", "sign_flip", "Sign-flip clients", "fraction"), ("evasion", "fgsm", "FGSM ε", "number")],
         success=FLIPPED, note="FedAvg, 10 clients, 30 rounds; FGSM against the resulting global model at test time."),
    dict(id="c3_drift_shots_noise", block="C3", scenario="s3", title="C3 · transpiler drift × forged shots × backend noise",
         factors=[("transpiler", "angle_drift", "RZ drift (rad)", "number"), ("shot_bias", "forged_bitstring", "Forged shots", "fraction"),
                  ("noise", "depolarizing_readout", "Error p", "number")],
         success=FLIPPED, note="Clean QuantumNet compiled through the drifting pass, run on the noisy backend, then shots forged."),
    dict(id="c4_backdoor_tamper", block="C4", scenario="s3", title="C4 · backdoor poisoning × circuit tampering",
         factors=[("backdoor", "trigger", "Triggered training rows", "fraction"), ("circuit_tamper", "rx_half_pi", "Injected RX(π/2) gates", "count")],
         success=BACKDOOR, note="QuantumNet retrained on the poisoned rows, then gates injected into the deployed circuit."),
]

METRIC_LABELS = {
    "clean_accuracy": ("Clean accuracy", "%"),
    "attacked_accuracy": ("Accuracy", "%"),
    "accuracy_drop_pp": ("Drop", "pp"),
    "attack_success_rate": ("Success", "%"),
    "balanced_accuracy": ("Balanced accuracy", "%"),
    "positive_recall": ("Malignant / macro recall", "%"),
    "f1": ("F1", "%"),
}


def by_id(sweep_id: str) -> dict:
    return next(s for s in SWEEPS if s["id"] == sweep_id)
