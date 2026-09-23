"""Circuit tampering by an adversary with backend or transpiler control: k extra RX(pi/2) gates inserted at
random positions on random qubits of the trained circuit. Intensity is the injected gate count.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction
from qiskit.circuit.library import RXGate


class CircuitTamperAttack:
    ANGLE = np.pi / 2

    def apply(self, circuit: QuantumCircuit, n_gates: int, rng: np.random.Generator) -> QuantumCircuit:
        tampered = circuit.copy()
        for _ in range(n_gates):
            position = int(rng.integers(0, len(tampered.data) + 1))
            qubit = tampered.qubits[int(rng.integers(0, tampered.num_qubits))]
            tampered.data.insert(position, CircuitInstruction(RXGate(self.ANGLE), (qubit,)))
        return tampered
