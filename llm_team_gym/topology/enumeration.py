"""
Exhaustive topology enumeration for LLM-TeamGym.

Enumerates all valid Worker-Validator bipartite connection patterns (shapes)
for small-scale teams (N <= 8) using exact canonical form computation.

A team has N agents: 1 Thinker (output) + W Workers + V Validators,
where W + V = N - 1, W >= 1, V >= 1.

The variable part is a binary matrix B of shape (W, V) where B[w,v] = 1
means Worker w connects to Validator v.

Two matrices are equivalent under independent row permutations and column
permutations (role-preserving isomorphism). A "shape" is an equivalence
class of matrices under these permutations.

Validity modes:
  - RELAXED: every row sum >= 1 (no idle Worker). Zero-column Validators allowed.
  - STRICT:  relaxed + every column sum >= 1 (no idle Validator).

Verified counts (from thesis):
  W=4, V=1 ->  1 shape
  W=4, V=2 ->  9 shapes (relaxed)
  W=4, V=3 -> 51 shapes (relaxed)
  Total four-penguin relaxed: 61 shapes
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np


def canonical_form(B: np.ndarray) -> tuple:
    """Compute the canonical representative of a binary matrix under
    independent row and column permutations.

    The canonical form is the lexicographically smallest matrix obtainable
    by applying all possible row permutations and all possible column
    permutations independently.

    Parameters
    ----------
    B : np.ndarray
        Binary matrix of shape (W, V).

    Returns
    -------
    tuple
        Flattened canonical matrix as a tuple (for hashing).
    """
    W, V = B.shape
    best: Optional[tuple] = None

    # Generate all row permutations and column permutations.
    # For each combination, permute the matrix and track the lex-smallest.
    row_perms = list(itertools.permutations(range(W)))
    col_perms = list(itertools.permutations(range(V)))

    for rp in row_perms:
        # Apply row permutation once.
        row_permuted = B[list(rp), :]
        for cp in col_perms:
            # Apply column permutation.
            permuted = row_permuted[:, list(cp)]
            flat = tuple(permuted.ravel())
            if best is None or flat < best:
                best = flat

    return best


def _is_valid(B: np.ndarray, relaxed: bool = True) -> bool:
    """Check if a binary matrix represents a valid topology.

    Parameters
    ----------
    B : np.ndarray
        Binary matrix of shape (W, V).
    relaxed : bool
        If True, only require every row sum >= 1 (Workers active).
        If False, also require every column sum >= 1 (Validators active).

    Returns
    -------
    bool
    """
    # Every Worker must connect to at least one Validator.
    if not np.all(B.sum(axis=1) >= 1):
        return False
    # Strict mode: every Validator must be connected to at least one Worker.
    if not relaxed:
        if not np.all(B.sum(axis=0) >= 1):
            return False
    return True


def enumerate_shapes(W: int, V: int, relaxed: bool = True) -> List[np.ndarray]:
    """Enumerate all distinct shapes (equivalence classes) of valid
    Worker-Validator bipartite connection matrices.

    Parameters
    ----------
    W : int
        Number of Workers (>= 1).
    V : int
        Number of Validators (>= 1).
    relaxed : bool
        Validity mode. True = relaxed (row sums >= 1 only).

    Returns
    -------
    List[np.ndarray]
        List of canonical representative matrices, each of shape (W, V).
    """
    if W < 1 or V < 1:
        raise ValueError(f"Need W >= 1 and V >= 1, got W={W}, V={V}")

    total_bits = W * V
    seen: set = set()
    shapes: List[np.ndarray] = []

    for code in range(1, 2**total_bits):
        # Build binary matrix from integer code.
        bits = [(code >> i) & 1 for i in range(total_bits)]
        B = np.array(bits, dtype=np.int8).reshape(W, V)

        if not _is_valid(B, relaxed=relaxed):
            continue

        canon = canonical_form(B)
        if canon not in seen:
            seen.add(canon)
            shapes.append(np.array(canon, dtype=np.int8).reshape(W, V))

    return shapes


def count_shapes(W: int, V: int, relaxed: bool = True) -> int:
    """Count the number of distinct shapes for given (W, V).

    Parameters
    ----------
    W : int
        Number of Workers.
    V : int
        Number of Validators.
    relaxed : bool
        Validity mode.

    Returns
    -------
    int
    """
    return len(enumerate_shapes(W, V, relaxed=relaxed))


def count_all_shapes(
    max_N: int = 8, relaxed: bool = True
) -> Dict[Tuple[int, int], int]:
    """Count shapes for all valid (W, V) pairs with W + V + 1 <= max_N.

    Parameters
    ----------
    max_N : int
        Maximum team size (Thinker + Workers + Validators).
    relaxed : bool
        Validity mode.

    Returns
    -------
    dict
        Mapping (W, V) -> count for all valid pairs.
    """
    results: Dict[Tuple[int, int], int] = {}
    for N in range(3, max_N + 1):
        # N = 1 (Thinker) + W + V, so W + V = N - 1 with W >= 1, V >= 1.
        for W in range(1, N - 1):
            V = N - 1 - W
            if V < 1:
                continue
            results[(W, V)] = count_shapes(W, V, relaxed=relaxed)
    return results


def shape_to_topology(shape: np.ndarray) -> dict:
    """Convert a canonical shape matrix to a topology descriptor.

    Parameters
    ----------
    shape : np.ndarray
        Binary matrix of shape (W, V).

    Returns
    -------
    dict
        Topology descriptor with keys:
        - 'W': number of Workers
        - 'V': number of Validators
        - 'N': total team size (1 + W + V)
        - 'matrix': the shape matrix as a list of lists
        - 'worker_degrees': list of row sums (connections per Worker)
        - 'validator_degrees': list of column sums (connections per Validator)
        - 'density': fraction of possible edges present
        - 'edges': list of (worker_idx, validator_idx) pairs
    """
    W, V = shape.shape
    row_sums = shape.sum(axis=1).tolist()
    col_sums = shape.sum(axis=0).tolist()
    edges = list(zip(*np.where(shape == 1)))

    return {
        "W": W,
        "V": V,
        "N": 1 + W + V,
        "matrix": shape.tolist(),
        "worker_degrees": row_sums,
        "validator_degrees": col_sums,
        "density": float(shape.sum()) / (W * V),
        "edges": [(int(r), int(c)) for r, c in edges],
    }


def print_count_table(max_N: int = 8, relaxed: bool = True) -> None:
    """Pretty-print the shape count table (thesis Table 1).

    Rows are indexed by W, columns by V. Only cells where W + V + 1 <= max_N
    are populated.

    Parameters
    ----------
    max_N : int
        Maximum team size.
    relaxed : bool
        Validity mode.
    """
    mode_label = "relaxed" if relaxed else "strict"
    counts = count_all_shapes(max_N=max_N, relaxed=relaxed)

    max_W = max(w for w, v in counts) if counts else 1
    max_V = max(v for w, v in counts) if counts else 1

    # Header.
    header = f"{'W\\V':>6}"
    for v in range(1, max_V + 1):
        header += f"  V={v:>2}"
    header += "  | Row Total"
    print(f"\nShape count table ({mode_label}, max_N={max_N}):")
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    grand_total = 0
    for w in range(1, max_W + 1):
        row_str = f"W={w:>2}  "
        row_total = 0
        for v in range(1, max_V + 1):
            if (w, v) in counts:
                c = counts[(w, v)]
                row_str += f"  {c:>4}"
                row_total += c
            else:
                row_str += "     -"
        row_str += f"  | {row_total:>9}"
        print(row_str)
        grand_total += row_total

    print("-" * len(header))
    print(f"{'Grand total':>6}: {grand_total}")
    print()


def _validate_known_counts() -> bool:
    """Validate against known correct counts from the thesis.

    Returns True if all checks pass, raises AssertionError otherwise.
    """
    checks = [
        (1, 1, True, 1, "W=1,V=1 relaxed"),
        (2, 1, True, 1, "W=2,V=1 relaxed (star)"),
        (3, 1, True, 1, "W=3,V=1 relaxed (star)"),
        (4, 1, True, 1, "W=4,V=1 relaxed (star)"),
        (4, 2, True, 9, "W=4,V=2 relaxed"),
        (4, 3, True, 51, "W=4,V=3 relaxed"),
        (1, 1, False, 1, "W=1,V=1 strict"),
    ]

    all_ok = True
    for W, V, relaxed, expected, label in checks:
        actual = count_shapes(W, V, relaxed=relaxed)
        status = "PASS" if actual == expected else "FAIL"
        if actual != expected:
            all_ok = False
        print(f"  [{status}] {label}: expected={expected}, actual={actual}")

    # Four-penguin total (relaxed).
    total_4p = sum(
        count_shapes(4, v, relaxed=True) for v in range(1, 4)
    )
    status = "PASS" if total_4p == 61 else "FAIL"
    if total_4p != 61:
        all_ok = False
    print(f"  [{status}] Four-penguin total (relaxed): expected=61, actual={total_4p}")

    return all_ok


if __name__ == "__main__":
    print("=" * 60)
    print("LLM-TeamGym Topology Enumeration")
    print("=" * 60)

    # Validate known counts.
    print("\nValidating against known counts...")
    ok = _validate_known_counts()
    if ok:
        print("\nAll validations PASSED.\n")
    else:
        print("\nSome validations FAILED!\n")

    # Print relaxed count table.
    print_count_table(max_N=8, relaxed=True)

    # Print strict count table.
    print_count_table(max_N=8, relaxed=False)
