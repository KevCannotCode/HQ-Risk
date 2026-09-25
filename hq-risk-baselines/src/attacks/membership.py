"""Membership inference (infer): confidence-threshold attack. The score is the victim's probability of the true
label; the attacker knows k members and k non-members, picks the threshold that separates them best, and is scored
on a balanced set of the remaining rows. Intensity is k (D47).
"""

import numpy as np


class MembershipInferenceAttack:
    def scores(self, victim, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return victim.predict_proba(x)[np.arange(len(y)), y]

    def accuracy(self, member_scores: np.ndarray, outsider_scores: np.ndarray, n_known: int, rng: np.random.Generator) -> float:
        n = min(len(member_scores), len(outsider_scores))
        if n_known >= n:
            raise ValueError(f"attacker knows {n_known} rows per side but only {n} exist")
        members = rng.permutation(member_scores)[:n]
        outsiders = rng.permutation(outsider_scores)[:n]
        threshold = self._best_threshold(members[:n_known], outsiders[:n_known])
        hits = np.r_[members[n_known:] >= threshold, outsiders[n_known:] < threshold]
        return float(np.mean(hits))

    def _best_threshold(self, members: np.ndarray, outsiders: np.ndarray) -> float:
        candidates = np.unique(np.r_[members, outsiders])
        accuracy = [np.mean(np.r_[members >= t, outsiders < t]) for t in candidates]
        return float(candidates[int(np.argmax(accuracy))])
