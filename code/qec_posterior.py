"""Exact logical-class decoding for the phenomenological repetition code.

The hidden state is the cumulative data-error bit string after each noisy
round. The forward recursion sums the likelihood of every fault history that
ends in each state. The noiseless final syndrome selects two complementary
states. Their relative posterior weights determine the logical prediction.
"""

import numpy as np

from qec_gen_data import ERROR_RATES


class ExactLogicalDecoder:
    """Binary logical-class MAP decoder for the generator's T = d rounds."""

    def __init__(self, d, p):
        self.d = d
        self.p = p
        self.n_detectors = (d + 1) * (d - 1)

        data_states = np.arange(1 << d)
        syndromes = (data_states ^ (data_states >> 1)) & ((1 << (d - 1)) - 1)
        self.syndrome_bits = 1 << np.arange(d - 1)
        self.final_states = np.empty((1 << (d - 1), 2), dtype=np.intp)
        self.final_states[syndromes, data_states & 1] = data_states

        measured = np.arange(1 << (d - 1))
        differences = measured[:, None] ^ syndromes[None, :]
        bit_counts = np.array([value.bit_count() for value in data_states.tolist()])
        n_wrong = bit_counts[differences]
        self.likelihood = p ** n_wrong * (1 - p) ** (d - 1 - n_wrong)
        self.first_round = self.likelihood * (p ** bit_counts * (1 - p) ** (d - bit_counts))

        # Eight detectors at d=3 make a complete lookup table inexpensive.
        self.lookup = None
        if self.n_detectors <= 8:
            detector_bits = 1 << np.arange(self.n_detectors)
            patterns = (np.arange(1 << self.n_detectors)[:, None] & detector_bits) != 0
            self.lookup = self._probabilities(patterns)

    def _probabilities(self, detectors):
        d, p = self.d, self.p
        measurements = np.bitwise_xor.accumulate(detectors.reshape(-1, d + 1, d - 1), axis=1)
        measured_syndromes = measurements @ self.syndrome_bits
        n_shots = len(detectors)
        weights = self.first_round[measured_syndromes[:, 0]]
        scratch = np.empty_like(weights)
        flipped_term = np.empty_like(weights)
        for t in range(1, d):
            # The transition factors into d independent binary symmetric
            # channels. Apply each one to neighboring pairs of states
            # differing at its qubit, without forming a 2^d by 2^d matrix.
            for q in range(d):
                pairs = weights.reshape(n_shots, -1, 2, 1 << q)
                updated = scratch.reshape(pairs.shape)
                np.multiply(pairs, 1 - p, out=updated)
                flipped = flipped_term.reshape(pairs.shape)
                np.multiply(pairs[:, :, ::-1, :], p, out=flipped)
                updated += flipped
                weights, scratch = scratch, weights
            weights *= self.likelihood[measured_syndromes[:, t]]

        candidates = self.final_states[measured_syndromes[:, -1]]
        logical = np.take_along_axis(weights, candidates, axis=1)
        return logical / logical.sum(axis=1, keepdims=True)

    def logical_probabilities(self, detectors):
        """Return a row of [Pr(logical=0 | data), Pr(logical=1 | data)] per shot."""
        detectors = np.asarray(detectors)
        if detectors.ndim != 2 or detectors.shape[1] != self.n_detectors:
            raise ValueError(f"expected detector shots with {self.n_detectors} columns")
        if self.lookup is not None:
            return self.lookup[np.packbits(detectors, axis=1, bitorder="little")[:, 0]]

        result = np.empty((len(detectors), 2))
        for start in range(0, len(detectors), 4096):
            batch = detectors[start:start + 4096]
            result[start:start + len(batch)] = self._probabilities(batch)
        return result

    def predict(self, detectors):
        """Return the more probable class, breaking numerical ties toward zero."""
        probabilities = self.logical_probabilities(detectors)
        return (probabilities[:, 1] - probabilities[:, 0] > 1e-14).astype(np.uint8)


def validate_exact_posterior():
    """Cross-check every d=3 detector pattern by enumerating 2^15 faults."""
    d = 3
    n_stab = d - 1
    n_det = (d + 1) * n_stab
    n_faults = d * (d + n_stab)
    history = (np.arange(1 << n_faults)[:, None] >> np.arange(n_faults)) & 1
    data_faults = history[:, :d * d].reshape(-1, d, d)
    readout_faults = history[:, d * d:].reshape(-1, d, n_stab)
    state = np.bitwise_xor.accumulate(data_faults, axis=1)
    true_syndrome = state[:, :, :-1] ^ state[:, :, 1:]
    readout = np.concatenate([true_syndrome ^ readout_faults, true_syndrome[:, -1:]], axis=1)
    detectors = np.diff(readout, axis=1, prepend=0) != 0
    detectors = detectors.reshape(-1, n_det)
    detector_bits = 1 << np.arange(n_det)
    syndrome_ids = detectors @ detector_bits
    logical = state[:, -1, 0]
    fault_counts = history.sum(axis=1)

    patterns = (np.arange(1 << n_det)[:, None] & detector_bits) != 0
    max_posterior_error = 0.0
    minimum_risks = {}
    for p in ERROR_RATES:
        probabilities = p ** fault_counts * (1 - p) ** (n_faults - fault_counts)
        joint = np.bincount(2 * syndrome_ids + logical, weights=probabilities,
                            minlength=2 * len(patterns)).reshape(-1, 2)
        expected = joint / joint.sum(axis=1, keepdims=True)
        decoder = ExactLogicalDecoder(d, p)
        actual = decoder.logical_probabilities(patterns)
        max_posterior_error = max(max_posterior_error, float(np.max(np.abs(actual - expected))))
        risk = joint[np.arange(len(patterns)), 1 - decoder.predict(patterns)].sum()
        minimum_risks[f"{p:g}"] = float(joint.min(axis=1).sum())
        if not np.isclose(risk, minimum_risks[f"{p:g}"], atol=1e-14, rtol=0):
            raise AssertionError(f"d=3 logical-class predictions fail enumeration at p={p}")

    if max_posterior_error > 1e-12:
        raise AssertionError(f"d=3 posterior disagrees with enumeration: {max_posterior_error}")
    return {"distance": d, "fault_histories": len(history), "detector_patterns": len(patterns),
            "physical_error_rates": ERROR_RATES, "max_posterior_abs_error": max_posterior_error,
            "expected_logical_error_rate": minimum_risks}
