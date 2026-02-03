from typing import Union, List, Set, FrozenSet, Tuple, Iterable, Dict, Optional, Any
import numpy as np
import scipy.sparse as sp

def bit_low(col_bits: int) -> int:
    """Highest set-bit index (PH 'low'), or -1 if col_bits == 0."""
    if col_bits == 0:
        return -1
    return col_bits.bit_length() - 1


def bitset_to_indices(bits: int) -> List[int]:
    """Convert int bitset to sorted list of indices where bit=1."""
    out: List[int] = []
    while bits:
        lsb = bits & -bits
        idx = lsb.bit_length() - 1
        out.append(idx)
        bits ^= lsb
    return out


def _cols_from_sparse_F2(D: sp.spmatrix) -> List[int]:
    """Convert sparse matrix D to list of column bitsets over rows (F2)."""
    D = D.tocsc(copy=False)
    n_cols = D.shape[1]
    cols: List[int] = [0] * n_cols

    indptr = D.indptr
    indices = D.indices

    for j in range(n_cols):
        start, end = indptr[j], indptr[j + 1]
        row_idxs = indices[start:end]
        bits = 0
        # XOR also cancels duplicate row indices mod 2 (shouldn't happen, but safe)
        for r in row_idxs:
            bits ^= (1 << int(r))
        cols[j] = bits
    return cols


def _cols_from_dense_F2(D: np.ndarray) -> List[int]:
    """Convert dense matrix D to list of column bitsets over rows (F2)."""
    A = np.asarray(D)
    n_cols = A.shape[1]
    cols: List[int] = [0] * n_cols
    for j in range(n_cols):
        nz = np.nonzero(A[:, j])[0]
        bits = 0
        for r in nz:
            bits ^= (1 << int(r))
        cols[j] = bits
    return cols


def reduce_boundary_matrix_F2(
    D: Union[sp.spmatrix, np.ndarray, List[List[int]]],
) -> Tuple[int, Dict[int, int], List[int], List[int]]:
    """
    Reduce a boundary matrix over F2 using XOR elimination (PH-style column reduction),
    tracking the combination of original columns that produce each reduced column.

    Returns:
        rank: rank(D) over F2
        pivot_of_low: mapping low_row -> pivot_col index
        reduced_cols: list of reduced column bitsets (over rows)
        combo_cols: list of combination bitsets (over original columns)
    """
    if sp.issparse(D):
        cols = _cols_from_sparse_F2(D)
    else:
        cols = _cols_from_dense_F2(np.asarray(D))

    n = len(cols)
    reduced_cols = cols[:]
    combo_cols = [(1 << j) for j in range(n)]  # identity tracking

    pivot_of_low: Dict[int, int] = {}

    for j in range(n):
        col = reduced_cols[j]
        comb = combo_cols[j]

        low = bit_low(col)
        while low != -1 and low in pivot_of_low:
            p = pivot_of_low[low]
            col ^= reduced_cols[p]
            comb ^= combo_cols[p]
            low = bit_low(col)

        reduced_cols[j] = col
        combo_cols[j] = comb

        if low != -1:
            pivot_of_low[low] = j

    rank = len(pivot_of_low)
    return rank, pivot_of_low, reduced_cols, combo_cols


def kernel_basis_F2(D: Union[sp.spmatrix, np.ndarray, List[List[int]]]) -> List[int]:
    """
    Basis for ker(D) over F2.
    Each basis vector is a bitset over columns of D.
    """
    _, _, reduced_cols, combo_cols = reduce_boundary_matrix_F2(D)
    basis = [combo_cols[j] for j, col in enumerate(reduced_cols) if col == 0 and combo_cols[j] != 0]
    return basis


def image_basis_F2(D: Union[sp.spmatrix, np.ndarray, List[List[int]]]) -> List[int]:
    """
    Basis for im(D) over F2 (column space).
    Each basis vector is a bitset over rows of D.
    """
    _, pivot_of_low, reduced_cols, _ = reduce_boundary_matrix_F2(D)
    pivot_cols = sorted(pivot_of_low.values())
    return [reduced_cols[j] for j in pivot_cols]


def image_basis_with_generators_F2(
    D: Union[sp.spmatrix, np.ndarray, List[List[int]]]
) -> Tuple[List[int], List[int]]:
    """
    Basis for im(D) over F2 plus generator vectors over columns.
    Returns (image_vectors_over_rows, generator_vectors_over_cols) such that D*gen = image_vec.
    """
    _, pivot_of_low, reduced_cols, combo_cols = reduce_boundary_matrix_F2(D)
    pivot_cols = sorted(pivot_of_low.values())
    image_vecs = [reduced_cols[j] for j in pivot_cols]
    gens = [combo_cols[j] for j in pivot_cols]
    return image_vecs, gens