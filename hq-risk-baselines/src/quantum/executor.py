"""Inference as the notebook does it: bind the trained weights, sample the circuit on Qiskit Aer with a finite
shot budget, turn counts into per-qubit Z expectations, feed the classical head. Attacks act on this path only.
"""

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import SamplerV2

from src.quantum.circuit import QuantumNet


class ShotExecutor:
    def __init__(self, seed: int, noise_model: NoiseModel | None = None):
        options = {"backend_options": {"noise_model": noise_model}} if noise_model is not None else None
        self.sampler = SamplerV2(seed=seed, options=options)

    def counts(self, net: QuantumNet, x: np.ndarray, shots: int, circuit: QuantumCircuit | None = None) -> list[dict[str, int]]:
        measured = (circuit if circuit is not None else net.circuit).copy()
        measured.measure_all()
        values = self._parameter_values(measured, net, x)
        result = self.sampler.run([(measured, values)], shots=shots).result()[0]
        bitstrings = result.data.meas
        return [bitstrings[row].get_counts() for row in range(len(x))]

    def _parameter_values(self, circuit: QuantumCircuit, net: QuantumNet, x: np.ndarray) -> np.ndarray:
        weights = net.flat_weights()
        column = {param: ("x", i) for i, param in enumerate(net.input_params)}
        column.update({param: ("w", i) for i, param in enumerate(net.weight_params)})
        values = np.empty((len(x), circuit.num_parameters))
        for position, param in enumerate(circuit.parameters):
            source, index = column[param]
            values[:, position] = x[:, index] if source == "x" else weights[index]
        return values

    def predict(self, net: QuantumNet, counts: list[dict[str, int]]) -> np.ndarray:
        with torch.no_grad():
            logits = net.head(torch.tensor(z_expectations(counts, net.n_qubits), dtype=torch.float64))
        return logits.argmax(dim=1).numpy()


def z_expectations(counts: list[dict[str, int]], n_qubits: int) -> np.ndarray:
    expectations = np.zeros((len(counts), n_qubits))
    for row, row_counts in enumerate(counts):
        shots = sum(row_counts.values())
        for bitstring, count in row_counts.items():
            for qubit, bit in enumerate(reversed(bitstring)):
                expectations[row, qubit] += (1.0 if bit == "0" else -1.0) * count
        expectations[row] /= shots
    return expectations
