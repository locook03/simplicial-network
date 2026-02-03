from typing import Union, List, Set, FrozenSet, Tuple, Iterable, Dict, Optional, Any
import numpy as np

from combinatorics import Point, PointSet, Simplex, SimplexSet, Chain, find_max_simplices
from homology import Metric

class SimplicialComplex(SimplexSet):
    """
    Similar to SimplexSet, but ensures membership of only maximum simplices.

    Attributes:
        metric: Metric helper for homology computations (over F2).
    """
    def __init__(self, iterable: Iterable[Simplex] = None):
        corrected = SimplexSet()
        if iterable:
            for item in iterable:
                corrected.add(self._validate(item))
            corrected = find_max_simplices(corrected)
        super(PointSet, self).__init__(corrected)

        # Attach metric helper
        self.metric = Metric(self)

    def __repr__(self):
        rep = "SimplicialComplex{" + f"{','.join([str(v) for v in self.ordered()])}" + "}"
        return rep

    def add(self, element: Simplex):
        element = self._validate(element)
        if element <= self:
            return
        to_remove = SimplexSet()
        # if existing simplex is contained in new element, remove it
        for simplex in self:
            if simplex < element:
                to_remove.add(simplex)
        for simplex in to_remove:
            super(PointSet, self).remove(simplex)
        super(PointSet, self).add(element)

    @property
    def max_simplices(self):
        return self

    def k_boundary_matrix(self, k: int) -> np.ndarray:
        """
        Boundary matrix D_k : C_k -> C_{k-1} over F2 (entries 0/1).
        Rows index (k-1)-simplices, columns index k-simplices.

        Note: ordering is deterministic via .ordered().
        """
        if k < 1:
            raise ValueError("k must be >= 1.")

        ksimplices = self.ksimplices(k).ordered()
        km1simplices = self.ksimplices(k - 1).ordered()

        nk = len(ksimplices)
        nkm1 = len(km1simplices)

        col_idx = {s: j for j, s in enumerate(ksimplices)}
        row_idx = {s: i for i, s in enumerate(km1simplices)}

        d = np.zeros((nkm1, nk), dtype=np.uint8)
        for ksimplex in ksimplices:
            j = col_idx[ksimplex]
            for facet in ksimplex.facets():
                i = row_idx[facet]
                d[i, j] = 1
        return d
    

def adjMat_to_simpComplex(mat, columns=None) -> SimplicialComplex:
    """
    Build a 1-skeleton simplicial complex (vertices + edges) from an adjacency matrix.
    """
    if columns is None:
        columns = range(0, mat.shape[0])
    if not len(columns) == mat.shape[0]:
        raise ValueError("columns length must match mat.shape[0]")

    simplices: Set[Simplex] = set()

    # vertices
    for i in range(mat.shape[0]):
        simplices.add(Simplex([columns[i]]))

    # edges
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if i < j and mat[i, j] != 0:
                simplices.add(Simplex([columns[i], columns[j]], distance=float(mat[i, j])))

    return SimplicialComplex(simplices)


if __name__ == "__main__":
    mat = np.array([[0, 1, 0, 1, 0],
                    [1, 0, 1, 0, 0],
                    [0, 1, 0, 1, 0],
                    [1, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0]])

    sc = adjMat_to_simpComplex(mat, columns=list("abcde"))

    print("SC:", sc)
    print("1-simplices:", sc.ksimplices(1))
    print("D1:\n", sc.k_boundary_matrix(1))

    # Example: Betti numbers on the clique complex induced by max_simplices procedure
    print("Betti (up to dim 2):", sc.metric.betti_numbers(max_dim=2))