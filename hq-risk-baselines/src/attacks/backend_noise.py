"""Noise increase: depolarizing error of probability p on every gate (1- and 2-qubit) plus symmetric readout
error p on every qubit. Intensity is p.
"""

from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error


class BackendNoiseAttack:
    SINGLE_QUBIT_GATES = ["rx", "ry", "rz"]
    TWO_QUBIT_GATES = ["cx"]

    def noise_model(self, probability: float) -> NoiseModel:
        model = NoiseModel()
        model.add_all_qubit_quantum_error(depolarizing_error(probability, 1), self.SINGLE_QUBIT_GATES)
        model.add_all_qubit_quantum_error(depolarizing_error(probability, 2), self.TWO_QUBIT_GATES)
        model.add_all_qubit_readout_error(ReadoutError([[1 - probability, probability], [probability, 1 - probability]]))
        return model
