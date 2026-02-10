from typing import TYPE_CHECKING, Union, List, Set, FrozenSet, Tuple, Iterable, Dict, Optional, Any
import math
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
        n_rows, n_cols = len(km1), len(ks)

        cols = []
        for sigma in ks:
            bits = 0
            for facet in sigma.facets():
                bits ^= (1 << row_idx[facet])
            cols.append(bits)

        rr: List[int] = []
        cc: List[int] = []
        for j, sigma in enumerate(ks):
            for facet in sigma.facets():
                rr.append(row_idx[facet])
                cc.append(j)

        data = np.ones(len(rr), dtype=np.uint8)
        return sp.csc_matrix((data, (rr, cc)), shape=(n_rows, n_cols))

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
    def persistence_intervals(
        self,
        max_dim: Optional[int] = None,
        include_simplices: bool = False,
    ):
        """
        Compute persistence intervals over F2.

        Returns:
          If include_simplices=False:
            {d: [(birth, death), ...]}
          If include_simplices=True:
            {d: [(birth, death, birth_simplex, death_simplex_or_None), ...]}

        Infinity intervals are represented with math.inf and death_simplex_or_None = None.
        """

        ordered = self.sc.ordered()
        t = {s: s.distance for s in ordered}

        if max_dim is None:
            max_dim = max(0, self.sc.dim)

        # Simplices by dimension in filtration order
        by_dim: Dict[int, List] = {d: [] for d in range(max_dim + 2)}
        for s in ordered:
            if 0 <= s.dim <= max_dim + 1:
                by_dim[s.dim].append(s)

        intervals: Dict[int, List[tuple]] = {d: [] for d in range(max_dim + 1)}

        # Births: H0 => every vertex; Hd (d>=1) => zero columns in reduced D_d
        births: Dict[int, Set] = {d: set() for d in range(max_dim + 1)}
        for v in by_dim.get(0, []):
            births[0].add(v)

        for d in range(1, max_dim + 1):
            if len(by_dim.get(d, [])) == 0 or len(by_dim.get(d - 1, [])) == 0:
                continue
            Dd = self.boundary_matrix(d)
            _, _, reduced_cols, _ = reduce_boundary_matrix_F2(Dd)
            d_simplices = by_dim[d]
            for j, col_bits in enumerate(reduced_cols):
                if col_bits == 0:
                    births[d].add(d_simplices[j])

        # Deaths/pairings from D_{d+1}: each low(row) paired with a (d+1)-simplex column
        paired_births: Dict[int, Set] = {d: set() for d in range(max_dim + 1)}
        for d in range(0, max_dim + 1):
            if len(by_dim.get(d + 1, [])) == 0 or len(by_dim.get(d, [])) == 0:
                continue
            Ddp1 = self.boundary_matrix(d + 1)
            _, pivot_of_low, _, _ = reduce_boundary_matrix_F2(Ddp1)

            d_simplices = by_dim[d]
            dp1_simplices = by_dim[d + 1]

            for low_i, col_j in pivot_of_low.items():
                birth_s = d_simplices[low_i]
                death_s = dp1_simplices[col_j]
                paired_births[d].add(birth_s)

                birth = t[birth_s]
                death = t[death_s]
                if include_simplices:
                    intervals[d].append((birth, death, birth_s, death_s))
                else:
                    intervals[d].append((birth, death))

        # Infinity intervals for unpaired births
        for d in range(0, max_dim + 1):
            for b in births[d]:
                if b not in paired_births[d]:
                    birth = t[b]
                    if include_simplices:
                        intervals[d].append((birth, math.inf, b, None))
                    else:
                        intervals[d].append((birth, math.inf))

        for d in intervals:
            intervals[d].sort(key=lambda x: (x[0], x[1]))
        return intervals