"""Exact, differentiable statevector simulation of a Qiskit circuit in torch, so Adam gets analytic gradients
(what parameter-shift would estimate). Reads the gates off the Qiskit circuit itself -- one circuit definition.
"""

import torch
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


class TorchStatevector:
    SINGLE_QUBIT = ("rx", "ry", "rz")
    IGNORED = ("barrier", "measure")

    def __init__(self, circuit: QuantumCircuit, input_params: ParameterVector, weight_params: ParameterVector):
        self.n_qubits = circuit.num_qubits
        sources = {param: ("x", i) for i, param in enumerate(input_params)}
        sources.update({param: ("w", i) for i, param in enumerate(weight_params)})
        self.operations = [self._compile(instruction, circuit, sources) for instruction in circuit.data]
        self.operations = [operation for operation in self.operations if operation is not None]

    def _compile(self, instruction, circuit: QuantumCircuit, sources: dict):
        name = instruction.operation.name
        qubits = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
        if name in self.IGNORED:
            return None
        if name == "cx":
            return ("cx", qubits, None)
        if name not in self.SINGLE_QUBIT:
            raise ValueError(f"gate {name!r} not supported by the torch simulator")
        return (name, qubits, self._angle(instruction.operation.params[0], sources))

    def _angle(self, expression, sources: dict):
        if not hasattr(expression, "parameters") or not expression.parameters:
            return ("const", None, float(expression))
        (param,) = expression.parameters
        coefficient = float(expression.gradient(param))
        if abs(float(expression.bind({param: 0.0}))) > 1e-12:
            raise ValueError(f"angle {expression} is not linear through zero")
        source, index = sources[param]
        return (source, index, coefficient)

    def z_expectations(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        state = torch.zeros(batch, 2**self.n_qubits, dtype=torch.complex128)
        state[:, 0] = 1.0
        for name, qubits, angle in self.operations:
            if name == "cx":
                state = self._cx(state, *qubits)
            else:
                state = self._rotate(state, name, qubits[0], self._angles(angle, x, weights, batch))
        probabilities = state.abs() ** 2
        return torch.stack([self._z(probabilities, qubit) for qubit in range(self.n_qubits)], dim=1)

    def _angles(self, angle, x: torch.Tensor, weights: torch.Tensor, batch: int) -> torch.Tensor:
        source, index, coefficient = angle
        if source == "const":
            return torch.full((batch,), coefficient, dtype=torch.float64)
        if source == "x":
            return coefficient * x[:, index].to(torch.float64)
        return (coefficient * weights[index]).expand(batch)

    def _rotate(self, state: torch.Tensor, name: str, qubit: int, theta: torch.Tensor) -> torch.Tensor:
        c, s = torch.cos(theta / 2).to(torch.complex128), torch.sin(theta / 2).to(torch.complex128)
        if name == "ry":
            matrix = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
        elif name == "rx":
            matrix = torch.stack([torch.stack([c, -1j * s], -1), torch.stack([-1j * s, c], -1)], -2)
        else:
            phase = torch.exp(-0.5j * theta.to(torch.complex128))
            zero = torch.zeros_like(phase)
            matrix = torch.stack([torch.stack([phase, zero], -1), torch.stack([zero, phase.conj()], -1)], -2)
        # Qiskit is little-endian: qubit q is bit q of the basis index.
        view = state.reshape(state.shape[0], 2 ** (self.n_qubits - 1 - qubit), 2, 2**qubit)
        return torch.einsum("bij,bajc->baic", matrix, view).reshape(state.shape)

    def _cx(self, state: torch.Tensor, control: int, target: int) -> torch.Tensor:
        index = torch.arange(2**self.n_qubits)
        flipped = torch.where((index >> control) & 1 == 1, index ^ (1 << target), index)
        return state[:, flipped]

    def _z(self, probabilities: torch.Tensor, qubit: int) -> torch.Tensor:
        bit = (torch.arange(2**self.n_qubits) >> qubit) & 1
        return probabilities @ (1.0 - 2.0 * bit).to(torch.float64)
