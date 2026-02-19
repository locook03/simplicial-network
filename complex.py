from typing import Union, List, Set, FrozenSet, Tuple, Iterable, Dict, Optional, Any
import numpy as np

from combinatorics import Point, PointSet, Simplex, SimplicialComplex, Chain, grow_skeleton
from homology import Homology



def rips_filtration(mat, max_dim: int = 3, columns = None) -> Homology:
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

    full_complex = simplices | grow_skeleton(simplices.ksimplices(1), max_dim=max_dim)
    return Homology(full_complex)


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
    homology = rips_filtration(mat, columns=columns)
    births, deaths, dim = homology.persistence.persistence_diagram()
    t, L, dims = homology.persistence.persistence_landscape()
    
    import matplotlib.pyplot as plt
    colors = {0: 'blue', 1: 'red', 2: 'green'}
    c = [colors[d] for d in dim]
    scatter = plt.scatter(births, deaths, c=c)
    plt.show()
    plt.figure()

    plt.plot(t, L.T)
    plt.legend(dims)
    plt.show()
    
    print(f"Betti numbers: {homology.betti_numbers()}")
    print(f"PE: {homology.persistence.entropy()}")

    print("done")
    
