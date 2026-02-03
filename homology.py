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
    def boundary_matrix(self, k: int, sparse: bool = False) -> Union[np.ndarray, sp.spmatrix]:
        """
        Return D_k : C_k -> C_{k-1}.
        Uses the complex's ordering of simplices (deterministic).
        """
        D = self.sc.k_boundary_matrix(k)
        if sparse:
            return sp.csc_matrix(D)
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
    def kernel_as_chains(self, k: int, sparse: bool = True) -> List[Chain]:
        """
        Return a list of k-cycles (kernel basis vectors) as Chains of k-simplices.
        Note: these are cycles, not yet reduced modulo boundaries.
        """
        ks = self.sc.ksimplices(k).ordered()
        Dk = self.boundary_matrix(k, sparse=sparse)
        ker = kernel_basis_F2(Dk)

        chains: List[Chain] = []
        for vbits in ker:
            idxs = bitset_to_indices(vbits)
            ch = Chain([ks[i] for i in idxs])
            chains.append(ch)
        return chains

    def image_as_chains(self, k: int, sparse: bool = True) -> List[Chain]:
        """
        Return a basis for im(D_k) as Chains of (k-1)-simplices (since im(D_k) subset of C_{k-1}).
        """
        km1 = self.sc.ksimplices(k - 1).ordered()
        Dk = self.boundary_matrix(k, sparse=sparse)
        img = image_basis_F2(Dk)

        chains: List[Chain] = []
        for row_bits in img:
            idxs = bitset_to_indices(row_bits)
            ch = Chain([km1[i] for i in idxs])
            chains.append(ch)
        return chains

    # ---- Betti numbers ----
    def betti_numbers(self, max_dim: Optional[int] = None, sparse: bool = True) -> Dict[int, int]:
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
            Dk = self.boundary_matrix(k, sparse=sparse)
            rankD[k] = self.rank_F2(Dk)

        # Need rank(D_{max_dim+1}) = 0 by definition (no higher simplices)
        rankD[max_dim + 1] = 0

        betti: Dict[int, int] = {}
        for k in range(0, max_dim + 1):
            betti[k] = n[k] - rankD.get(k, 0) - rankD.get(k + 1, 0)

        return betti
