from typing import Union, List, Set, FrozenSet, Tuple, Iterable, Dict, Optional, Any
import numpy as np

from combinatorics import Point, PointSet, Simplex, SimplicialComplex, Chain, find_supersets
from homology import Metric


class CliqueComplex(SimplicialComplex):
    """
    Similar to SimplicialComplex, but ensures membership of only maximum simplices.

    Attributes:
        metric: Metric helper for homology computations (over F2).
    """
    def __init__(self, iterable: Iterable[Simplex] = None):
        corrected = SimplicialComplex()
        if iterable:
            for item in iterable:
                corrected.add(self._validate(item))
            corrected = find_supersets(corrected)
        super(PointSet, self).__init__(corrected)

        # Attach metric helper
        self.metric = Metric(self)

    def __repr__(self):
        rep = "CliqueComplex{" + f"{','.join([str(v) for v in self.ordered()])}" + "}"
        return rep

    def add(self, element: Simplex):
        element = self._validate(element)
        if element <= self:
            return
        
        to_remove = SimplicialComplex()
        to_add = SimplicialComplex()
        maximal_set = find_supersets(self.ksimplices(element.dim).add(element))
        # if existing simplex is contained in new element, remove it
        for maximal_simplex in maximal_set:
            to_add.add(maximal_simplex)
            for simplex in self:
                if simplex < maximal_simplex:
                    to_remove.add(simplex)
        for simplex in to_remove:
            super(PointSet, self).remove(simplex)
        for simplex in to_add:
            super(PointSet, self).add(simplex)

    def remove(self, element: Simplex):
        return NotImplemented


class CliqueFiltration:
    def __init__(self, complex: SimplicialComplex):
        self.vertices = complex.ksimplices(0)
        self.edges = complex.ksimplices(1)

        # Ensure all edges have distances
        if not np.all([edge.distance != float('inf') for edge in self.edges]):
            raise ValueError("All edges provided for filtration must have an explicit, non-infinity distance.")
        # Make all vertices have distance 0
        for vert in self.vertices:
            vert.distance = 0
        
        complex = self.vertices | self.edges
        self.full_complex = find_supersets(complex, max_only=False)

        
def rips_filtration(mat, columns=None) -> Metric:
    if columns is None:
        columns = range(0, mat.shape[0])
    if not len(columns) == mat.shape[0]:
        raise ValueError("columns length must match mat.shape[0]")

    simplices = SimplicialComplex()

    # vertices
    for i in range(mat.shape[0]):
        simplices.add(Simplex([columns[i]], distance=float(0)))

    # edges
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if i < j and mat[i, j] != 0:
                simplices.add(Simplex([columns[i], columns[j]], distance=float(mat[i, j])))

    full_complex = simplices | find_supersets(simplices, max_only=False)
    full_complex.metric = Metric(full_complex)
    return full_complex.metric


def adj_mat_to_clique_complex(mat, columns=None) -> CliqueComplex:
    """
    Build a clique complex from an adjacency matrix by adding 1-skeleton simplices.
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

    return CliqueComplex(simplices)


if __name__ == "__main__":
    # mat = np.array([[0, 1, 1, 1, 0],
    #                 [1, 0, 1, 1, 0],
    #                 [0, 1, 0, 1, 0],
    #                 [1, 0, 1, 0, 0],
    #                 [0, 0, 0, 0, 0]])

    # clique_complex = adj_mat_to_clique_complex(mat, columns=list("abcde"))

    # print("Clique Complex:", clique_complex)
    # print("1-simplices:", clique_complex.ksimplices(1))
    # print("D1:\n", clique_complex.k_boundary_matrix(1))

    # # Example: Betti numbers on the clique complex induced by max_simplices procedure
    # print("Betti (up to dim 2):", clique_complex.metric.betti_numbers(max_dim=2))

    mat = np.array([[0, 1, 4, 2],
                    [1, 0, 2, 5],
                    [4, 2, 0, 2],
                    [2, 5, 2, 0]])
    columns = list("abcd")
    metric = rips_filtration(mat, columns=columns)
    print(adj_mat_to_clique_complex(mat, columns=columns))
    pp = metric.persistence_pairs()
    print(pp)
    print(metric.betti_numbers())
    print("done")
    
