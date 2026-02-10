from typing import TYPE_CHECKING, Union, List, Set, FrozenSet, Tuple, Iterable, Dict, Optional, Any
import numpy as np
import scipy.sparse as sp

from combinatorics import Chain
if TYPE_CHECKING:
    from complex import SimplicialComplex
from linear_algebra import reduce_boundary_matrix_F2, kernel_basis_F2, image_basis_F2, bitset_to_indices

class Metric:
    """
    Computation helper attached to a SimplicialComplex.
    Currently supports homology computations over F2 using boundary matrices.
    """
    def __init__(self, complex_ref: "SimplicialComplex"):
        self.sc = complex_ref

    # ---- Boundary matrices ----
    def boundary_matrix(self, k: int) -> sp.csc_matrix:
        """
        Boundary matrix D_k : C_k -> C_{k-1} over F2 (entries 0/1).
        Rows index (k-1)-simplices, columns index k-simplices.

        Note: ordering is deterministic via .ordered().
        """
        if k < 1:
            raise ValueError("k must be >= 1.")
        ordered = self.sc.ordered()
        ks = [s for s in ordered if s.dim == k]
        km1 = [s for s in ordered if s.dim == k-1]
        row_idx = {s: i for i, s in enumerate(km1)}

        cols = []
        for sigma in ks:
            bits = 0
            for facet in sigma.facets():
                bits ^= (1 << row_idx[facet])
            cols.append(bits)

        # convert bitset columns to CSC sparse
        # build lists of (row, col) for nonzeros
        rows, cols_idx = [], []
        for j, bits in enumerate(cols):
            for r in bitset_to_indices(bits):
                rows.append(r)
                cols_idx.append(j)

        data = np.ones(len(rows), dtype=np.uint8)
        D = sp.csc_matrix((data, (rows, cols_idx)), shape=(len(km1), len(ks)))
        return D

    # ---- Linear algebra over F2 ----
    def rank_F2(self, D: Union[np.ndarray, sp.spmatrix]) -> int:
        rank, _, _, _ = reduce_boundary_matrix_F2(D)
        return rank

    def kernel_F2(self, D: Union[np.ndarray, sp.spmatrix]) -> List[int]:
        return kernel_basis_F2(D)

    def image_F2(self, D: Union[np.ndarray, sp.spmatrix]) -> List[int]:
        return image_basis_F2(D)

    # ---- Cycles / boundaries as Chains of simplices ----
    def kernel_as_chains(self, k: int) -> List[Chain]:
        """
        Return a list of k-cycles (kernel basis vectors) as Chains of k-simplices.
        Note: these are cycles, not yet reduced modulo boundaries.
        """
        ks = self.sc.ksimplices(k).ordered()
        Dk = self.boundary_matrix(k)
        ker = kernel_basis_F2(Dk)

        chains: List[Chain] = []
        for vbits in ker:
            idxs = bitset_to_indices(vbits)
            ch = Chain([ks[i] for i in idxs])
            chains.append(ch)
        return chains

    def image_as_chains(self, k: int) -> List[Chain]:
        """
        Return a basis for im(D_k) as Chains of (k-1)-simplices (since im(D_k) subset of C_{k-1}).
        """
        km1 = self.sc.ksimplices(k - 1).ordered()
        Dk = self.boundary_matrix(k)
        img = image_basis_F2(Dk)

        chains: List[Chain] = []
        for row_bits in img:
            idxs = bitset_to_indices(row_bits)
            ch = Chain([km1[i] for i in idxs])
            chains.append(ch)
        return chains

    # ---- Betti numbers ----
    def betti_numbers(self, max_dim: Optional[int] = None) -> Dict[int, int]:
        """
        Compute Betti numbers β_k over F2 using:
            β_k = n_k - rank(D_k) - rank(D_{k+1})
        where D_0 is the zero map.

        Args:
            max_dim: maximum k to compute. Defaults to complex.dim.
            sparse: convert boundary matrices to sparse (CSC) before reduction.

        Returns:
            dict: {k: β_k}
        """
        if max_dim is None:
            max_dim = max(0, self.sc.dim)

        # Precompute simplex counts
        n = {k: len(self.sc.ksimplices(k)) for k in range(0, max_dim + 1)}

        # Ranks of boundary maps
        rankD: Dict[int, int] = {0: 0}
        for k in range(1, max_dim + 1):
            Dk = self.boundary_matrix(k)
            rankD[k] = self.rank_F2(Dk)

        # Need rank(D_{max_dim+1}) = 0 by definition (no higher simplices)
        rankD[max_dim + 1] = 0

        betti: Dict[int, int] = {}
        for k in range(0, max_dim + 1):
            betti[k] = n[k] - rankD.get(k, 0) - rankD.get(k + 1, 0)

        return betti
    
    # ---- Persistence Pairs ----
    def persistence_pairs(self, max_dim: Optional[int] = None):
        """
        Return persistence intervals by dimension as:
           intervals[d] = list of (birth, death) in H_d
        """
        if max_dim is None:
            max_dim = max(0, self.sc.dim)

        ordered = self.sc.ordered()

        # filtration times for simplices
        t = {s: s.distance for s in ordered}

        intervals = {d: [] for d in range(max_dim + 1)}

        for k in range(1, max_dim + 1):
            Dk = self.boundary_matrix(k)
            rank, pivot_of_low, reduced_cols, combo = reduce_boundary_matrix_F2(Dk)

            km1 = [s for s in ordered if s.dim == k-1]
            ks  = [s for s in ordered if s.dim == k]

            # each pivot low row i paired with pivot column j kills H_{k-1} class
            for low_i, col_j in pivot_of_low.items():
                birth = t[km1[low_i]]
                death = t[ks[col_j]]
                intervals[k-1].append((birth, death))

        return intervals