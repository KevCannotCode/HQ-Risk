# Reference systems — real numbers (breast_cancer, 5 seeds, mean ± sd)

Every attacked system is at the strongest point of its sweep. Federated rows use the iid partition. QML rows are evaluated on Qiskit Aer, 256 shots. ARF = adversarial-AI factor, QTF · TSV = quantum-threat factor, transpiler / supply chain.

| System | Intensity | Clean accuracy (%) | Attacked accuracy (%) | Attack success (%) | Accuracy drop (pp) | HQ-Risk |
|---|---|---|---|---|---|---|
| Clean ML | none | 97.0 ± 2.0 | 97.0 ± 2.0 | — | 0.0 ± 0.0 | |
| Poisoned ML | 0.4 | 97.0 ± 2.0 | 74.4 ± 1.7 | — undefined (D10) | 22.6 ± 2.7 | |
| Targeted-poisoned ML | 0.4 | 97.0 ± 2.0 | 80.7 ± 2.1 | 52.4 ± 5.6 (source rows → target class) | 16.3 ± 2.4 | |
| Evaded ML | 1 | 97.0 ± 2.0 | 6.1 ± 2.1 | 93.7 ± 2.2 (correct → wrong) | 90.9 ± 3.4 | |
| Backdoored ML (ARF) | 0.2 | 97.0 ± 2.0 | 90.5 ± 2.3 | 93.8 ± 6.2 (triggered rows → target class) | 6.5 ± 2.3 | |
| Stolen ML (ARF) | 1000 | 97.0 ± 2.0 | 96.8 ± 1.7 | 99.1 ± 0.6 (fidelity of stolen copy) | 0.2 ± 1.1 | |
| Membership-inferred ML (ARF) | 20 | 97.0 ± 2.0 | 97.0 ± 2.0 | 51.3 ± 2.1 (membership accuracy (50 = guess)) | 0.0 ± 0.0 | |
| Inverted ML (ARF) | 2000 | 97.0 ± 2.0 | 97.0 ± 2.0 | 70.5 ± 3.3 (cosine to class mean (×100)) | 0.0 ± 0.0 | |
| Clean FL | none | 97.4 ± 1.4 | 97.4 ± 1.4 | — | 0.0 ± 0.0 | |
| Malicious FL, label flip | 0.5 | 97.4 ± 1.4 | 52.5 ± 7.6 | — undefined (D10) | 44.9 ± 8.5 | |
| Malicious FL, sign flip | 0.5 | 97.4 ± 1.4 | 27.0 ± 16.7 | — undefined (D10) | 70.4 ± 17.3 | |
| Malicious FL, targeted flip | 0.5 | 97.4 ± 1.4 | 78.2 ± 2.4 | 59.0 ± 6.6 (source rows → target class) | 19.1 ± 2.9 | |
| Backdoored FL (ARF) | 0.5 | 97.4 ± 1.4 | 89.5 ± 2.6 | 91.9 ± 6.0 (triggered rows → target class) | 7.9 ± 2.5 | |
| Clean QML | none | 92.3 ± 1.4 | 92.3 ± 1.4 | — | 0.0 ± 0.0 | |
| Tampered QML | 16 | 92.3 ± 1.4 | 42.6 ± 27.9 | 57.9 ± 29.2 (correct → wrong) | 49.6 ± 29.0 | |
| Shot-biased QML | 0.4 | 92.3 ± 1.4 | 63.2 ± 0.0 | 35.0 ± 2.4 (correct → wrong) | 29.1 ± 1.4 | |
| Noisy QML | 0.1 | 92.3 ± 1.4 | 75.1 ± 2.9 | 23.4 ± 2.4 (correct → wrong) | 17.2 ± 3.3 | |
| Transpiler drift QML (QTF · TSV) | 0.8 | 92.3 ± 1.4 | 56.1 ± 9.5 | 43.0 ± 11.8 (correct → wrong) | 36.1 ± 9.1 | |
| Transpiler swap QML (QTF · TSV) | 4 | 92.3 ± 1.4 | 42.5 ± 13.8 | 58.2 ± 14.4 (correct → wrong) | 49.8 ± 13.4 | |
| Backdoored QML (ARF) | 0.2 | 92.3 ± 1.4 | 76.7 ± 6.9 | 81.9 ± 10.6 (triggered rows → target class) | 15.6 ± 7.3 | |
| Stolen QML (ARF) | 1000 | 92.3 ± 1.4 | 94.2 ± 2.0 | 94.2 ± 1.3 (fidelity of stolen copy) | -1.9 ± 1.1 | |
| Membership-inferred QML (ARF) | 20 | 92.3 ± 1.4 | 92.3 ± 1.4 | 52.2 ± 2.0 (membership accuracy (50 = guess)) | 0.0 ± 0.0 | |
| Inverted QML (ARF) | 2000 | 92.3 ± 1.4 | 92.3 ± 1.4 | 65.5 ± 7.9 (cosine to class mean (×100)) | 0.0 ± 0.0 | |

Clean QML is the project lead's QuantumNet circuit trained with Adam; QML drops are against it, not against logistic regression. FL clean accuracy is FedAvg with 0 malicious clients, not the centralised model. Stolen rows: attacked accuracy is the stolen copy's accuracy. Membership and inversion rows leave the model untouched (drop 0); their attack success is the attack's own score (METHODS D47, D48).
