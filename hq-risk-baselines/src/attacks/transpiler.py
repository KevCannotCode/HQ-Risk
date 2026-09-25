"""Transpiler / supply-chain attack (TSV): the trained circuit goes through an honest Qiskit preset pass manager into
which a compromised pass has been slipped. Two passes: angle drift (every RZ in the compiled circuit shifted by
delta radians) and qubit swap (k SWAPs inserted before readout, as a routing stage that drops the final layout) (D49).
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZGate, SwapGate
from qiskit.transpiler import PassManager, TransformationPass
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


class AngleDriftPass(TransformationPass):
    def __init__(self, delta: float):
        super().__init__()
        self.delta = delta

    def run(self, dag):
        for node in dag.op_nodes():
            if node.name == "rz":
                dag.substitute_node(node, RZGate(node.op.params[0] + self.delta))
        return dag


class QubitSwapPass(TransformationPass):
    def __init__(self, pairs: list[tuple[int, int]]):
        super().__init__()
        self.pairs = pairs

    def run(self, dag):
        for a, b in self.pairs:
            dag.apply_operation_back(SwapGate(), (dag.qubits[a], dag.qubits[b]))
        return dag


class TranspilerAttack:
    ANGLE_DRIFT = "angle_drift"
    QUBIT_SWAP = "qubit_swap"
    BASIS = ["rz", "sx", "x", "cx"]
    OPTIMIZATION_LEVEL = 1

    def compile(self, circuit: QuantumCircuit, mode: str, intensity: float, rng: np.random.Generator) -> QuantumCircuit:
        manager = generate_preset_pass_manager(optimization_level=self.OPTIMIZATION_LEVEL, basis_gates=self.BASIS)
        manager.post_optimization = PassManager([self._malicious_pass(mode, intensity, circuit.num_qubits, rng)])
        return manager.run(circuit)

    def _malicious_pass(self, mode: str, intensity: float, n_qubits: int, rng: np.random.Generator) -> TransformationPass:
        if mode == self.ANGLE_DRIFT:
            return AngleDriftPass(intensity)
        if mode == self.QUBIT_SWAP:
            pairs = [tuple(int(q) for q in rng.choice(n_qubits, size=2, replace=False)) for _ in range(int(intensity))]
            return QubitSwapPass(pairs)
        raise ValueError(f"unknown transpiler attack mode {mode!r}")
