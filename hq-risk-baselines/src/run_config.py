"""One experiment configuration. Its hash is the run_id, so a field added here must not change existing ids."""

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RunConfig:
    scenario: str
    dataset: str
    model: str
    attack: str
    attack_mode: str
    intensity: float
    seed: int
    n_clients: int | None = None
    partition: str | None = None
    rounds: int | None = None
    local_epochs: int | None = None
    learning_rate: float | None = None
    batch_size: int | None = None
    dirichlet_alpha: float | None = None
    n_qubits: int | None = None
    shots: int | None = None
    epochs: int | None = None
    # Combined runs only (attack == COMBINED): ((attack, attack_mode, level), ...) applied in that order.
    factors: tuple[tuple[str, str, float], ...] | None = None

    S1 = "s1"
    S2 = "s2"
    S3 = "s3"
    POISONING = "poisoning"
    EVASION = "evasion"
    BYZANTINE = "byzantine"
    CIRCUIT_TAMPER = "circuit_tamper"
    SHOT_BIAS = "shot_bias"
    NOISE = "noise"
    TRANSPILER = "transpiler"
    BACKDOOR = "backdoor"
    EXTRACTION = "extraction"
    MEMBERSHIP = "membership"
    INVERSION = "inversion"
    COMBINED = "combined"
    # Attacks that only query the deployed model; they never change what it predicts.
    QUERY_ATTACKS = (EXTRACTION, MEMBERSHIP, INVERSION)
    ATTACKS = (POISONING, EVASION, BYZANTINE, CIRCUIT_TAMPER, SHOT_BIAS, NOISE, TRANSPILER, BACKDOOR, *QUERY_ATTACKS)
    # Added after the first runs; left out of the hash when unset so every run_id already in runs.csv is unchanged.
    _OPTIONAL_FIELDS = ("n_qubits", "shots", "epochs", "factors")

    def factor_level(self, attack: str, attack_mode: str) -> float:
        """Level of one factor of a combined run; 0 when the factor is absent."""
        for factor_attack, factor_mode, level in self.factors or ():
            if (factor_attack, factor_mode) == (attack, attack_mode):
                return float(level)
        return 0.0

    def factors_text(self) -> str | None:
        if self.factors is None:
            return None
        return ";".join(f"{attack}:{mode}={level:g}" for attack, mode, level in self.factors)

    def config_hash(self) -> str:
        fields = asdict(self)
        fields.pop("seed")
        return self._digest(fields)

    def run_id(self) -> str:
        return self._digest(asdict(self))

    def _digest(self, fields: dict) -> str:
        fields = {key: value for key, value in fields.items() if not (key in self._OPTIONAL_FIELDS and value is None)}
        return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:12]
