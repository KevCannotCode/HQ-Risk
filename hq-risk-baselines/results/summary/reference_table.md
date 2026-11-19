# Reference systems — real numbers (breast_cancer, mean ± sd over 20 seeds)

Every attacked system is at the strongest point of its sweep. Federated rows use the iid partition. QML rows are evaluated on Qiskit Aer, 256 shots. ARF = adversarial-AI factor, QTF · TSV = quantum-threat factor, transpiler / supply chain.

| System | Intensity | Clean accuracy (%) | Attacked accuracy (%) | Attack success (%) | Accuracy drop (pp) | HQ-Risk |
|---|---|---|---|---|---|---|
| Clean ML | none | 97.5 ± 1.5 | 97.5 ± 1.5 | — | 0.0 ± 0.0 | |
| Poisoned ML | 0.4 | 97.5 ± 1.5 | 76.7 ± 5.3 | — undefined (D10) | 20.9 ± 5.4 | |
| Targeted-poisoned ML | 0.4 | 97.5 ± 1.5 | 81.6 ± 2.4 | 49.9 ± 6.4 (source rows → target class) | 16.0 ± 2.4 | |
| Evaded ML | 1 | 97.5 ± 1.5 | 6.3 ± 2.0 | 93.6 ± 2.0 (correct → wrong) | 91.3 ± 1.9 | |
| Backdoored ML (ARF) | 0.2 | 97.5 ± 1.5 | 90.7 ± 2.3 | 91.7 ± 5.3 (triggered rows → target class) | 6.8 ± 2.1 | |
| Stolen ML (ARF) | 1000 | 97.5 ± 1.5 | 97.6 ± 1.6 | 99.3 ± 0.7 (fidelity of stolen copy) | -0.0 ± 0.7 | |
| Membership-inferred ML (ARF) | 20 | 97.5 ± 1.5 | 97.5 ± 1.5 | 51.5 ± 2.5 (membership accuracy (50 = guess)) | 0.0 ± 0.0 | |
| Inverted ML (ARF) | 2000 | 97.5 ± 1.5 | 97.5 ± 1.5 | 71.7 ± 4.2 (cosine to class mean (×100)) | 0.0 ± 0.0 | |
| Clean FL | none | 97.5 ± 1.3 | 97.5 ± 1.3 | — | 0.0 ± 0.0 | |
| Malicious FL, label flip | 0.5 | 97.5 ± 1.3 | 49.4 ± 13.9 | — undefined (D10) | 48.2 ± 13.9 | |
| Malicious FL, sign flip | 0.5 | 97.5 ± 1.3 | 24.7 ± 16.0 | — undefined (D10) | 72.8 ± 16.3 | |
| Malicious FL, targeted flip | 0.5 | 97.5 ± 1.3 | 77.4 ± 2.8 | 61.4 ± 7.7 (source rows → target class) | 20.2 ± 3.0 | |
| Backdoored FL (ARF) | 0.5 | 97.5 ± 1.3 | 88.8 ± 1.8 | 87.7 ± 5.1 (triggered rows → target class) | 8.7 ± 1.7 | |
| Clean QML | none | 91.7 ± 4.2 | 91.7 ± 4.2 | — | 0.0 ± 0.0 | |
| Tampered QML | 16 | 91.7 ± 4.2 | 38.5 ± 18.2 | 63.0 ± 19.1 (correct → wrong) | 53.2 ± 18.2 | |
| Shot-biased QML | 0.4 | 91.7 ± 4.2 | 63.2 ± 0.0 | 35.6 ± 3.6 (correct → wrong) | 28.5 ± 4.2 | |
| Noisy QML | 0.1 | 91.7 ± 4.2 | 74.5 ± 11.2 | 22.8 ± 10.1 (correct → wrong) | 17.2 ± 9.8 | |
| Transpiler drift QML (QTF · TSV) | 0.8 | 91.7 ± 4.2 | 45.5 ± 14.7 | 54.9 ± 17.0 (correct → wrong) | 46.2 ± 15.2 | |
| Transpiler swap QML (QTF · TSV) | 4 | 91.7 ± 4.2 | 54.3 ± 22.6 | 44.3 ± 25.0 (correct → wrong) | 37.3 ± 21.7 | |
| Backdoored QML (ARF) | 0.2 | 91.7 ± 4.2 | 77.6 ± 5.7 | 84.2 ± 9.2 (triggered rows → target class) | 14.0 ± 6.1 | |
| Stolen QML (ARF) | 1000 | 91.7 ± 4.2 | 94.1 ± 3.2 | 94.5 ± 3.0 (fidelity of stolen copy) | -2.4 ± 2.6 | |
| Membership-inferred QML (ARF) | 20 | 91.7 ± 4.2 | 91.7 ± 4.2 | 49.9 ± 2.7 (membership accuracy (50 = guess)) | 0.0 ± 0.0 | |
| Inverted QML (ARF) | 2000 | 91.7 ± 4.2 | 91.7 ± 4.2 | 72.2 ± 7.3 (cosine to class mean (×100)) | 0.0 ± 0.0 | |

Clean QML is the project lead's QuantumNet circuit trained with Adam; QML drops are against it, not against logistic regression. FL clean accuracy is FedAvg with 0 malicious clients, not the centralised model. Stolen rows: attacked accuracy is the stolen copy's accuracy. Membership and inversion rows leave the model untouched (drop 0); their attack success is the attack's own score (METHODS D47, D48).
