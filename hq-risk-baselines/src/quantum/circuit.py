"""The project lead's QuantumNet circuit (Qiskit notebook v6), unchanged: angle encoding, one RY/RZ variational
layer, a brick of CX, per-qubit Z readout into BatchNorm + Linear. Only the optimiser differs: Adam, not SantaQuark.
"""

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from torch import nn

from src.quantum.statevector import TorchStatevector


class QuantumNet(nn.Module):
    N_REUPLOADS = 1
    LAYERS_PER_REUPLOAD = 1

    def __init__(self, n_qubits: int, n_classes: int, seed: int):
        super().__init__()
        self.n_qubits = n_qubits
        # nn.Linear draws its init from the global RNG; fork it so the seed is local to this model.
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.quantum_weights = nn.Parameter(
                torch.randn(self.N_REUPLOADS, self.LAYERS_PER_REUPLOAD, n_qubits, 2, dtype=torch.float64) * 0.05
            )
            self.batch_norm = nn.BatchNorm1d(n_qubits, dtype=torch.float64)
            self.fc_out = nn.Linear(n_qubits, n_classes, dtype=torch.float64)
        self.circuit, self.input_params, self.weight_params = self._build_circuit()
        self.simulator = TorchStatevector(self.circuit, self.input_params, self.weight_params)

    def _build_circuit(self) -> tuple[QuantumCircuit, ParameterVector, ParameterVector]:
        circuit = QuantumCircuit(self.n_qubits)
        input_params = ParameterVector("x", self.n_qubits)
        weight_params = ParameterVector("w", self.N_REUPLOADS * self.LAYERS_PER_REUPLOAD * self.n_qubits * 2)
        w_idx = 0
        for reupload in range(self.N_REUPLOADS):
            for qubit in range(self.n_qubits):
                circuit.ry(np.pi * input_params[qubit], qubit)
                circuit.rz(np.pi * input_params[qubit], qubit)
            for _ in range(self.LAYERS_PER_REUPLOAD):
                for qubit in range(self.n_qubits):
                    circuit.ry(weight_params[w_idx], qubit)
                    circuit.rz(weight_params[w_idx + 1], qubit)
                    w_idx += 2
            start = 0 if reupload % 2 == 0 else 1
            for qubit in range(start, self.n_qubits - 1, 2):
                circuit.cx(qubit, qubit + 1)
        return circuit, input_params, weight_params

    def flat_weights(self) -> np.ndarray:
        return self.quantum_weights.detach().reshape(-1).numpy().copy()

    def head(self, expectations: torch.Tensor) -> torch.Tensor:
        return self.fc_out(self.batch_norm(expectations))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.simulator.z_expectations(x, self.quantum_weights.reshape(-1)))
