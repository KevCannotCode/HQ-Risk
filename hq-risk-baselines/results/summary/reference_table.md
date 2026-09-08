# Reference systems — real numbers (breast_cancer, 5 seeds, mean ± sd)

Poisoned ML = symmetric label flip at 40% of training rows. Malicious FL = 50% label-flipping clients, iid partition. Evaded ML = FGSM ε = 1.0 (added: the only system with a defined attack-success rate).

| System | Clean accuracy (%) | Attacked accuracy (%) | Attack success (%) | Accuracy drop (pp) | HQ-Risk |
|---|---|---|---|---|---|
| Clean ML | 97.0 ± 2.0 | 97.0 ± 2.0 | — | 0.0 ± 0.0 | |
| Poisoned ML | 97.0 ± 2.0 | 74.4 ± 1.7 | — (undefined, see D10) | 22.6 ± 2.7 | |
| Evaded ML (added) | 97.0 ± 2.0 | 6.1 ± 2.1 | 93.7 ± 2.2 | 90.9 ± 3.4 | |
| Malicious FL | 97.4 ± 1.4 | 52.5 ± 7.6 | — (undefined, see D10) | 44.9 ± 8.5 | |
| Tampered QML | BLOCKED | BLOCKED | BLOCKED | BLOCKED | |

Tampered QML is blocked until the variational circuit is supplied. Malicious FL clean accuracy is the federated clean baseline (FedAvg, 0 malicious clients), not the centralised one. Attack success rate is undefined for poisoning and federated attacks until it is defined (decision D10).
