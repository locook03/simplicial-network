from typing import TYPE_CHECKING, Union, List, Set, FrozenSet, Tuple, Iterable, Dict, Optional, Any
import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import scipy.sparse as sp

from combinatorics import Chain
if TYPE_CHECKING:
    from combinatorics import Simplex
    from complex import SimplicialComplex
from linear_algebra import reduce_boundary_matrix_F2, kernel_basis_F2, image_basis_F2, bitset_to_indices

@dataclass
class Persistence:
    """
    Dataclass to hold persistence information.

    interval: should have structure [(birth, death, birth_simplex, death_simplex_or_None), ...]
    """
    interval: List[Tuple[float,float,Simplex,Simplex]]

    def __post_init__(self):
        for d in self.interval:
            self._validate(d)
    def _validate(self, item):
        if len(item) != 4: raise ValueError("Persistence Collection must be a collection of tuples with length 4: [(birth, death, birth_simplex, death_simplex_or_None), ...]")
    def sort(self):
        self.interval.sort(key=lambda t: (t[0],-t[1], t[2].dim))

    # ---- Persistence Metrics ----
    def entropy(self) -> float:
        b, d, bs, ds = [np.array(x) for x in zip(*self.interval)]
        fin = np.isfinite(d)
        p = d[fin] - b[fin]
        nonzero = p > 0
        p = p[nonzero]
        l = np.sum(p)
        return float(-np.sum(p/l * np.log2(p/l)))
    def total(self) -> float:
        b, d, bs, ds = list(zip(*self.interval))
        b, d = np.asarray(b), np.asarray(d)
        fin = np.isfinite(d)
        return float(np.sum(d[fin] - b[fin]))

    # ---- Persistence Graph Utils ----
    def persistence_diagram(self) -> Tuple[List[float], List[float], List[int]]:
        self.sort()
        intervals = self.interval
        intervals = [(b,d,bs,ds) for b,d,bs,ds in intervals if np.isfinite(b) and np.isfinite(d) and d-b > 0]
        b, d, bs, ds = list(zip(*intervals))
        dim = [s.dim for s in bs]
        return b, d, dim
    def persistence_landscape(self, npoints: int=1000) -> Tuple[np.ndarray, np.ndarray, list]:
        b, d, dim = self.persistence_diagram()
        b = np.asarray(b).reshape(-1, 1)
        d = np.asarray(d).reshape(-1, 1)

        j = len(b)
        l = np.zeros((j,npoints))
        x = np.linspace(0,max(d),npoints)
        for n, t in enumerate(x):
            t_vec = np.full(j, t).reshape(-1, 1)
            l[:,n] = np.max([np.zeros((j,1)), np.min([t_vec-b, d-t_vec], axis=0)], axis=0).ravel()
        return x, l, dim


@dataclass(frozen=True)
class Homology:
    """
    Computation helper attached to a SimplicialComplex.
    Currently supports homology computations over F2 using boundary matrices.
    """
    sc: "SimplicialComplex"
    max_dim: Optional[int] = 3

    @cached_property
    def persistence(self) -> Persistence:
        return persistence_interval(self.sc, self.max_dim)

    def betti_numbers(self):
        return betti_numbers(self.sc, self.max_dim)

# ---- Boundary matrices ----
def boundary_matrix(sc: "SimplicialComplex", k: int) -> sp.csc_matrix:
    """
    Boundary matrix D_k : C_k -> C_{k-1} over F2 (entries 0/1).
    Rows index (k-1)-simplices, columns index k-simplices.

    Note: ordering is deterministic via .ordered().
    """
    if k < 1:
        raise ValueError("k must be >= 1.")
    ordered = sc.ordered()
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
def rank_F2(D: Union[np.ndarray, sp.spmatrix]) -> int:
    rank, _, _, _ = reduce_boundary_matrix_F2(D)
    return rank

def kernel_F2( D: Union[np.ndarray, sp.spmatrix]) -> List[int]:
    return kernel_basis_F2(D)

def image_F2(D: Union[np.ndarray, sp.spmatrix]) -> List[int]:
    return image_basis_F2(D)

# ---- Cycles / boundaries as Chains of simplices ----
def kernel_as_chains(sc: "SimplicialComplex", k: int) -> List[Chain]:
    """
    Return a list of k-cycles (kernel basis vectors) as Chains of k-simplices.
    Note: these are cycles, not yet reduced modulo boundaries.
    """
    ks = sc.ksimplices(k).ordered()
    Dk = boundary_matrix(sc, k)
    ker = kernel_basis_F2(Dk)

    chains: List[Chain] = []
    for vbits in ker:
        idxs = bitset_to_indices(vbits)
        ch = Chain([ks[i] for i in idxs])
        chains.append(ch)
    return chains

def image_as_chains(sc: "SimplicialComplex", k: int) -> List[Chain]:
    """
    Return a basis for im(D_k) as Chains of (k-1)-simplices (since im(D_k) subset of C_{k-1}).
    """
    km1 = sc.ksimplices(k - 1).ordered()
    Dk = boundary_matrix(sc, k)
    img = image_basis_F2(Dk)

    chains: List[Chain] = []
    for row_bits in img:
        idxs = bitset_to_indices(row_bits)
        ch = Chain([km1[i] for i in idxs])
        chains.append(ch)
    return chains

# ---- Betti numbers ----
def betti_numbers(sc: "SimplicialComplex", max_dim: Optional[int] = None) -> Dict[int, int]:
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
        max_dim = max(0, sc.dim)

    # Precompute simplex counts
    n = {k: len(sc.ksimplices(k)) for k in range(0, max_dim + 1)}

    # Ranks of boundary maps
    rankD: Dict[int, int] = {0: 0}
    for k in range(1, max_dim + 1):
        Dk = boundary_matrix(sc, k)
        rankD[k] = rank_F2(Dk)

    # Need rank(D_{max_dim+1}) = 0 by definition (no higher simplices)
    rankD[max_dim + 1] = 0

    betti: Dict[int, int] = {}
    for k in range(0, max_dim + 1):
        betti[k] = n[k] - rankD.get(k, 0) - rankD.get(k + 1, 0)

    return betti

# ---- Persistence Pairs ----
def persistence_interval(
    sc: "SimplicialComplex",
    max_dim: Optional[int] = None
) -> Persistence:
    """
    Compute persistence intervals over F2.

    Returns:
        Persistence

    Infinity intervals are represented with math.inf and death_simplex_or_None = None.
    """

    ordered = sc.ordered()
    t = {s: s.distance for s in ordered}

    if max_dim is None:
        max_dim = max(0, sc.dim)

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
        Dd = boundary_matrix(sc, d)
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
        Ddp1 = boundary_matrix(sc, d + 1)
        _, pivot_of_low, _, _ = reduce_boundary_matrix_F2(Ddp1)

        d_simplices = by_dim[d]
        dp1_simplices = by_dim[d + 1]

        for low_i, col_j in pivot_of_low.items():
            birth_s = d_simplices[low_i]
            death_s = dp1_simplices[col_j]
            paired_births[d].add(birth_s)

            birth = t[birth_s]
            death = t[death_s]
            intervals[d].append((birth, death, birth_s, death_s))

    # Infinity intervals for unpaired births
    for d in range(0, max_dim + 1):
        for b in births[d]:
            if b not in paired_births[d]:
                birth = t[b]
                intervals[d].append((birth, math.inf, b, None))

    # Don't need dimension, already encoded in birth simplex.
    intervals = [interval for dim_intervals in list(intervals.values()) for interval in dim_intervals]
    return Persistence(intervals)

