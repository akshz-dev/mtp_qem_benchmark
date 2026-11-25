"""Match detector events for the phenomenological repetition-code memory."""

from functools import lru_cache
from itertools import combinations

import numpy as np


def detector_graph_edges(d):
    """Yield (detector, detector or None for boundary, logical-Z flip).

    Equal data and measurement fault probabilities make all physical edges
    equally costly. The final perfect readout has only vertical edges from
    measurement faults in the preceding round.
    """
    n_stab = d - 1
    for t in range(d):
        first = t * n_stab
        yield first, None, 1
        for j in range(n_stab - 1):
            yield first + j, first + j + 1, 0
        yield first + n_stab - 1, None, 0
        for j in range(n_stab):
            yield first + j, first + n_stab + j, 0


def _matching_graph(d):
    from pymatching import Matching

    graph = Matching()
    for u, v, logical in detector_graph_edges(d):
        faults = {0} if logical else set()
        if v is None:
            graph.add_boundary_edge(u, fault_ids=faults)
        else:
            graph.add_edge(u, v, fault_ids=faults)
    return graph


def _detector_batch(detectors, d):
    data = np.asarray(detectors, dtype=np.uint8)
    if d < 3 or data.ndim != 2 or data.shape[1] != (d + 1) * (d - 1):
        raise ValueError("Expected a detector matrix with (d + 1)(d - 1) columns for d >= 3")
    return np.ascontiguousarray(data)


def decode_mwpm(detectors, d):
    """Return minimum-weight matching's predicted logical flip for each shot."""
    data = _detector_batch(detectors, d)
    return _matching_graph(d).decode_batch(data)[:, 0]


@lru_cache(maxsize=None)
def _greedy_schedule(d):
    """Order pairs and boundary moves for the grid after peeling the final row."""
    n_stab = d - 1
    nodes = d * n_stab
    boundaries = []
    for u in range(nodes):
        j = u % n_stab
        boundaries.append((min(j + 1, n_stab - j), u, -1, j + 1 < n_stab - j))

    moves = list(boundaries)
    for u, v in combinations(range(nodes), 2):
        tu, ju = divmod(u, n_stab)
        tv, jv = divmod(v, n_stab)
        move = (tv - tu + abs(jv - ju), u, v, False)
        # This pair can never be selected if either endpoint's independent
        # boundary move would already have removed that endpoint.
        if move < boundaries[u] and move < boundaries[v]:
            moves.append(move)
    return sorted(moves)


def _decode_greedy(detectors, d, *, return_weights=False):
    data = _detector_batch(detectors, d)
    n_stab = d - 1
    final = data[:, -n_stab:].T
    remaining = data[:, :-n_stab].T.copy(order="C")
    # Each final-row defect fixes the use of its sole incident edge. That
    # edge toggles the preceding row and carries no logical flip.
    remaining[-n_stab:] ^= final
    parity = np.zeros(len(data), dtype=np.uint8)
    weights = final.sum(axis=0, dtype=np.int16) if return_weights else None
    for weight, u, v, left_boundary in _greedy_schedule(d):
        if v == -1:
            if left_boundary:
                parity ^= remaining[u]
            if return_weights:
                weights += weight * remaining[u]
            remaining[u].fill(0)
        else:
            paired = remaining[u] & remaining[v]
            if return_weights:
                weights += weight * paired
            remaining[u] ^= paired
            remaining[v] ^= paired
    return (parity, weights) if return_weights else parity


def decode_greedy(detectors, d):
    """Greedily pair nearest defects or send them to a spatial boundary."""
    return _decode_greedy(detectors, d)


def _simulate_faults(faults, d):
    """Map explicit data and measurement fault histories to detector events."""
    data = faults[:, :d * d].reshape(-1, d, d)
    measurement = faults[:, d * d:].reshape(-1, d, d - 1)
    state = np.bitwise_xor.accumulate(data, axis=1)
    checks = state[:, :, :-1] ^ state[:, :, 1:]
    noisy_checks = checks ^ measurement
    readouts = np.concatenate((noisy_checks, checks[:, -1:]), axis=1)
    detectors = readouts.copy()
    detectors[:, 1:] ^= readouts[:, :-1]
    return detectors.reshape(len(faults), -1), state[:, -1, 0]


def validate_matching():
    """Check every graph edge against one simulated fault, then exhaust d=3."""
    from qec_gen_data import ERROR_RATES

    single_fault_checks = {}
    for d in (3, 5, 7, 9):
        n_faults = d * d + d * (d - 1)
        detectors, logical = _simulate_faults(np.eye(n_faults, dtype=np.uint8), d)
        observed = set()
        for row, bit in zip(detectors, logical):
            indices = np.flatnonzero(row).tolist()
            v = indices[1] if len(indices) == 2 else None
            observed.add((indices[0], v, int(bit)))
        edges = set(detector_graph_edges(d))
        if len(observed) != n_faults or observed != edges:
            raise AssertionError(f"Distance-{d} detector graph disagrees with single faults")
        single_fault_checks[str(d)] = n_faults

    d = 3
    n_faults = single_fault_checks[str(d)]
    patterns = np.arange(1 << n_faults, dtype=np.uint32)
    faults = ((patterns[:, None] >> np.arange(n_faults)) & 1).astype(np.uint8)
    detectors, logical = _simulate_faults(faults, d)
    n_detectors = detectors.shape[1]
    n_syndromes = 1 << n_detectors
    syndrome_code = detectors @ (1 << np.arange(n_detectors))
    fault_counts = faults.sum(axis=1).astype(int)
    lowest_cost = np.full((n_syndromes, 2), n_faults + 1)
    np.minimum.at(lowest_cost, (syndrome_code, logical), fault_counts)
    syndromes = np.arange(n_syndromes)
    shots = ((syndromes[:, None] >> np.arange(n_detectors)) & 1).astype(np.uint8)
    prediction, weights = _matching_graph(d).decode_batch(shots, return_weights=True)
    optimum = lowest_cost.min(axis=1)
    greedy_parity, greedy_weights = _decode_greedy(shots, d, return_weights=True)
    cost_mismatches = int(np.count_nonzero(weights != optimum))
    class_mismatches = int(np.count_nonzero(lowest_cost[syndromes, prediction[:, 0]] != optimum))
    invalid_greedy = np.count_nonzero(greedy_weights < lowest_cost[syndromes, greedy_parity])
    if cost_mismatches or class_mismatches or invalid_greedy:
        raise AssertionError("Decoder output disagrees with the exhaustive physical fault model")

    # Sum the physical fault probabilities by (syndrome, logical class).
    # Min and max select between classes only when both have minimum cost.
    history_counts = np.zeros((n_syndromes, 2, n_faults + 1), dtype=np.int64)
    np.add.at(history_counts, (syndrome_code, logical, fault_counts), 1)
    possible = lowest_cost == optimum[:, None]
    ties = possible.all(axis=1)
    counts = np.arange(n_faults + 1)
    risk_by_p = {}
    for p in ERROR_RATES:
        joint = history_counts @ (p ** counts * (1 - p) ** (n_faults - counts))
        wrong_class = joint[:, ::-1]
        risk_by_p[f"{p:g}"] = {
            "pymatching": float(wrong_class[syndromes, prediction[:, 0]].sum()),
            "min_across_mwpm_tie_choices": float(np.min(np.where(possible, wrong_class, np.inf), axis=1).sum()),
            "max_across_mwpm_tie_choices": float(np.max(np.where(possible, wrong_class, -np.inf), axis=1).sum()),
            "logical_class_map": float(wrong_class.min(axis=1).sum()),
            "tied_syndrome_probability": float(joint[ties].sum()),
        }
    return {
        "backend": "PyMatching Sparse Blossom",
        "single_fault_checks_by_distance": single_fault_checks,
        "exhaustive_distance": d,
        "n_fault_histories": len(faults),
        "n_syndromes": n_syndromes,
        "mwpm_cost_mismatches": cost_mismatches,
        "mwpm_logical_class_mismatches": class_mismatches,
        "syndromes_with_tied_logical_classes": int(np.count_nonzero(ties)),
        "greedy_optimal_syndromes": int(np.count_nonzero(greedy_weights == optimum)),
        "analytic_logical_error_rate_by_p": risk_by_p,
    }
