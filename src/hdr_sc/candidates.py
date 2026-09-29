from __future__ import annotations

import itertools
import math
import random


def generate_candidates(
    pool_size: int,
    support_size: int,
    max_candidates: int,
    seed: int,
) -> list[tuple[int, ...]]:
    """Generate unique support candidates deterministically.

    The complete combination space is enumerated when it fits within the
    search budget. Otherwise, candidates are sampled uniformly without
    replacement and returned in lexicographic order.
    """

    if pool_size < 1:
        raise ValueError("pool_size must be at least 1")
    if not 1 <= support_size <= pool_size:
        raise ValueError("support_size must be between 1 and pool_size")
    if max_candidates < 1:
        raise ValueError("max_candidates must be at least 1")

    total = math.comb(pool_size, support_size)
    if total <= max_candidates:
        return list(itertools.combinations(range(pool_size), support_size))

    rng = random.Random(int(seed))
    candidates: set[tuple[int, ...]] = set()
    population = list(range(pool_size))
    while len(candidates) < max_candidates:
        candidates.add(tuple(sorted(rng.sample(population, support_size))))
    return sorted(candidates)

