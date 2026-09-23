# Reference systems — real numbers (breast_cancer, 5 seeds, mean ± sd)

Poisoned ML = symmetric label flip at 40% of training rows. Malicious FL = 50% label-flipping clients, iid partition. Evaded ML = FGSM ε = 1.0 (added). Tampered QML = 16 RX(π/2) gates injected into the trained QuantumNet circuit; shot-biased QML = 40% of shots forged; noisy QML = depolarizing + readout p = 0.1 (both added). Every QML row is evaluated on Qiskit Aer, 256 shots.

| System | Clean accuracy (%) | Attacked accuracy (%) | Attack success (%) | Accuracy drop (pp) | HQ-Risk |
|---|---|---|---|---|---|
| Clean ML | 97.0 ± 2.0 | 97.0 ± 2.0 | — | 0.0 ± 0.0 | |
| Poisoned ML | 97.0 ± 2.0 | 74.4 ± 1.7 | — (undefined, see D10) | 22.6 ± 2.7 | |
| Evaded ML (added) | 97.0 ± 2.0 | 6.1 ± 2.1 | 93.7 ± 2.2 | 90.9 ± 3.4 | |
| Malicious FL | 97.4 ± 1.4 | 52.5 ± 7.6 | — (undefined, see D10) | 44.9 ± 8.5 | |
| Clean QML (added) | 92.3 ± 1.4 | 92.3 ± 1.4 | — | 0.0 ± 0.0 | |
| Tampered QML | 92.3 ± 1.4 | 42.6 ± 27.9 | 57.9 ± 29.2 | 49.6 ± 29.0 | |
| Shot-biased QML (added) | 92.3 ± 1.4 | 63.2 ± 0.0 | 35.0 ± 2.4 | 29.1 ± 1.4 | |
| Noisy QML (added) | 92.3 ± 1.4 | 75.1 ± 2.9 | 23.4 ± 2.4 | 17.2 ± 3.3 | |

Clean QML is the project lead's QuantumNet circuit trained with Adam; its clean accuracy is the QML baseline, not the logistic-regression one. QML attack success rate = test rows correct before and wrong after (same definition as evasion; all S3 attacks act at inference, D35). Malicious FL clean accuracy is the federated clean baseline (FedAvg, 0 malicious clients), not the centralised one. Attack success rate is undefined for poisoning and federated attacks until it is defined (decision D10).
