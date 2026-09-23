"""Shot manipulation: the adversary controls the returned counts and replaces a fraction of each circuit's shots
with one forged bitstring -- the one the trained head maps most strongly to the target class.
"""

import itertools

import numpy as np
import torch

from src.quantum.circuit import QuantumNet


class ShotBiasAttack:
    def forged_shots(self, shots: int, fraction: float) -> int:
        return int(round(fraction * shots))

    def target_bitstring(self, net: QuantumNet, target_label: int) -> str:
        bitstrings = ["".join(bits) for bits in itertools.product("01", repeat=net.n_qubits)]
        # bitstring index 0 is the highest qubit (Qiskit order); a 0 bit reads as Z = +1
        expectations = np.array([[1.0 if bit == "0" else -1.0 for bit in reversed(b)] for b in bitstrings])
        with torch.no_grad():
            logits = net.head(torch.tensor(expectations, dtype=torch.float64)).numpy()
        others = np.delete(logits, target_label, axis=1).max(axis=1)
        return bitstrings[int(np.argmax(logits[:, target_label] - others))]

    def falsify(self, counts: list[dict[str, int]], bitstring: str, forged: int) -> list[dict[str, int]]:
        falsified = []
        for row_counts in counts:
            row = dict(row_counts)
            row[bitstring] = row.get(bitstring, 0) + forged
            falsified.append(row)
        return falsified
