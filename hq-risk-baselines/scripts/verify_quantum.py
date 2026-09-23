"""Check the torch simulator against Qiskit's own Statevector, and that training is reproducible. Run after any S3 change."""

import sys
from pathlib import Path

import numpy as np
import torch
from qiskit.quantum_info import SparsePauliOp, Statevector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attacks.circuit_tamper import CircuitTamperAttack
from src.datasets import DatasetLoader
from src.quantum.circuit import QuantumNet
from src.quantum.encoder import PcaAngleEncoder
from src.quantum.statevector import TorchStatevector
from src.quantum.trainer import QuantumTrainer


def qiskit_z(circuit, bindings: dict, n_qubits: int) -> list[float]:
    state = Statevector(circuit.assign_parameters(bindings))
    return [state.expectation_value(SparsePauliOp("I" * (n_qubits - 1 - q) + "Z" + "I" * q)).real for q in range(n_qubits)]


def simulator_error(net: QuantumNet, circuit, x: np.ndarray) -> float:
    simulator = TorchStatevector(circuit, net.input_params, net.weight_params)
    ours = simulator.z_expectations(torch.tensor(x), net.quantum_weights.detach().reshape(-1)).numpy()
    weights = dict(zip(net.weight_params, net.flat_weights()))
    reference = [qiskit_z(circuit, {**dict(zip(net.input_params, row)), **weights}, net.n_qubits) for row in x]
    return float(np.abs(ours - np.array(reference)).max())


def main() -> None:
    net = QuantumNet(n_qubits=8, n_classes=2, seed=0)
    with torch.no_grad():
        net.quantum_weights.normal_(0.0, 1.0)
    x = np.random.default_rng(0).random((5, 8))
    tampered = CircuitTamperAttack().apply(net.circuit, 6, np.random.default_rng(0))
    for name, circuit in [("trained circuit", net.circuit), ("tampered circuit", tampered)]:
        error = simulator_error(net, circuit, x)
        print(f"{name}: max |torch - qiskit| Z expectation = {error:.2e}")
        assert error < 1e-10

    data = DatasetLoader().load("breast_cancer", 0)
    features = PcaAngleEncoder(8, 0).fit(data.x_train).transform(data.x_train)
    weights = [QuantumTrainer(2, 0.05, 16, 0).fit(QuantumNet(8, 2, 0), features, data.y_train).flat_weights() for _ in range(2)]
    print(f"training twice, same seed: max weight difference = {np.abs(weights[0] - weights[1]).max():.2e}")
    assert np.array_equal(weights[0], weights[1])
    print("OK")


if __name__ == "__main__":
    main()
