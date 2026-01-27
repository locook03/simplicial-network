from typing import Union, List, Self, Type, Set, FrozenSet, Tuple, Iterable
from dataclasses import dataclass
import numpy as np
import itertools

class Graph:
    def __init__(self, matrix):
        self.matrix = self._validate_matrix(matrix)
        self.nvertices = self.matrix.shape[1]

    def _validate_matrix(self, matrix):
        """
        Checks if matrix is numpylizable (homogenous) and square (and 2D)
        """
        matrix = np.array(matrix)
        if matrix.ndim != 2:
            raise ValueError(f"The matrix should be 2D. The given matrix is {matrix.ndim}D.")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"The matrix should be square. The given matrix is {matrix.shape}.")
        
        return matrix

class UndirectedGraph(Graph): #undirected (symmetric), unweighted, and no self-loops
    def __init__(self, adj_matrix):
        super().__init__(adj_matrix)

        self.simplexes_calculated = False
        self.max_simplices_dim = {0: [], 1: [], 2:[]} # dim (int): [{vertex (int),},]
        self.max_simplex_dim = 0

        self._validate_adjacent()
        self.create_skeleton()

    def _validate_adjacent(self):
        """
        Checks if matrix is integers, 1 or 0, symmetric, and not self-looping (0 diagonal)
        """
        if self.matrix.dtype.kind not in {'i', 'u'}:
            raise ValueError(f"The adjacency matrix should be of integer type. The given matrix is of {self.matrix.dtype} type.")
        if not np.all(np.isin(self.matrix, [0,1])):
            raise ValueError(f"The adjacency matrix should be composed of only 0 and 1.")
        if not np.array_equal(self.matrix, self.matrix.T):
            raise ValueError(f"The adjacency matrix should be symmetrical.")
        if not (np.diag(self.matrix) == 0).all():
            raise ValueError(f"The adjacency matrix should not be self-looping. The diagonal should only have zeros.")
        return
    
    def create_skeleton(self):
        # Create vertices
        self.max_simplices_dim[0] = [set(range(0, self.nvertices))]
        
        # Create edge of vertices for each 1 in Adjacency matrix
        for i in range(0, self.matrix.shape[0]):
            for j in range(0, self.matrix.shape[1]):
                if i < j and self.matrix[i,j] == 1:
                    self.max_simplices_dim[1].append({i,j})
                    self.max_simplex_dim = 1
        
        # Remove vertices that are in edges (no longer max simplices)
        v_in_edges = set().union(*self.max_simplices_dim[1])
        new_v = self.max_simplices_dim[0][0] - v_in_edges
        self.max_simplices_dim[0][0] =new_v
        # Turn dim 0 back into a list of all vertices: [{vertex},] from [{vertex,}]
        self.max_simplices_dim[0] = [set([i]) for i in list(self.max_simplices_dim[0][0])]
    
    def calculate_simplices(self):
        dim = 2
        # Iterate to next dim if there exists at least {dim} simplices in prev dim (min needed for a higher-dim simplex)
        while len(self.max_simplices_dim[dim-1]) > dim:
            dm1_simplices = self.max_simplices_dim[dim-1].copy() # copy of dim-1 simplices (facets of next face)
            possible_d_simplices = itertools.combinations(range(0,self.nvertices), dim+1) # combinatorial number of possible faces
            # Iterate over possible faces, checking if required facets exist
            for ps in possible_d_simplices:
                needed_dm1_simplices = list(itertools.combinations(ps, dim))
                form_new_simplex = np.all([set(needed_dm1_simplex) in dm1_simplices for needed_dm1_simplex in needed_dm1_simplices])
                # If all required facets are in the lower dim, add the new face
                if form_new_simplex:
                    if dim not in self.max_simplices_dim.keys(): self.max_simplices_dim[dim] = []
                    self.max_simplices_dim[dim].append(set(ps))
                    self.max_simplex_dim = dim
                    # Remove facets from max simplex list, could've already been removed in prev iteration
                    for needed_dm1_simplex in needed_dm1_simplices:
                        try:
                            self.max_simplices_dim[dim-1].remove(set(needed_dm1_simplex))
                        except ValueError:
                            continue
            dim += 1
        self.simplexes_calculated = True
    
    def get_max_simplices(self):
        if not self.simplexes_calculated:
            raise RuntimeError("Run UndirectedGraph().calculate_simplices() before calling this method.")
        ms = []
        for d in self.max_simplices_dim.values():
            ms.extend(d)
        return ms
        
    def create_simplicial_complex(self):
        if not self.simplexes_calculated:
            raise RuntimeError("Run UndirectedGraph().calculate_simplices() before calling this method.")
        max_simplices = self.get_max_simplices()
        return SimplicialComplex(max_simplices)

@dataclass(frozen=True)
class Simplex:
    verts: FrozenSet

    def __post_init__(self):
        """Make sure verts is a frozenset"""
        if not isinstance(self.verts, Iterable):
            raise TypeError(f"Given verts must be of type Iterable, got: {type(self.verts)}")
        object.__setattr__(self, "verts", frozenset(self.verts))
        
    def __repr__(self):
        rep = "Simplex{" + f"{",".join([str(v) for v in self.ordered()])}" + "}"
        return rep

    def __iter__(self):
        """Return facets when iterating"""
        return iter(self.facets())
    
    def __len__(self):
        return len(self.verts)
    
    def __contains__(self, s):
        """Returns if simplex is within this simplex"""
        if isinstance(s, Simplex): return s.verts in self.verts
        elif isinstance(s, Iterable): return frozenset(s) in self.verts
        else: return NotImplemented
        
    # Comparisons reveal subset information. If < or <= is true, it is a subset of another.
    # If > or >= is true, it is a superset of another.
    def __lt__(self, s):
        if not isinstance(s, Simplex): return NotImplemented
        return self.verts.issubset(s.verts) and self != s
    def __le__(self, s):
        if not isinstance(s, Simplex): return NotImplemented
        return self.verts.issubset(s.verts)
    def __gt__(self, s):
        if not isinstance(s, Simplex): return NotImplemented
        return self.verts.issuperset(s.verts) and self != s
    def __ge__(self, s):
        if not isinstance(s, Simplex): return NotImplemented
        return self.verts.issuperset(s.verts)

    @property
    def dim(self): return len(self.verts) - 1

    def facets(self) -> set[Simplex]:
        return self.ndfacets(self.dim-1)
    
    def ndfacets(self, facet_dim: int) -> set[Simplex]:
        if facet_dim > self.dim: raise ValueError(f"Dimension of facet should be less than dimension of simplex. Given {facet_dim}, Expected <= {self.dim}.")
        ndfacets = {Simplex(verts) for verts in list(itertools.combinations(self.verts, facet_dim+1))}
        return ndfacets
    
    def ordered(self) -> Tuple:
        """Orders vertices into a list."""
        return sorted(tuple(self.verts))


class SimplicialComplex():
    def __init__(self, simplices: Union[Simplex,Set[Simplex]]):
        if isinstance(simplices, Simplex): given_simplices = set(simplices)
        else: given_simplices = simplices

        self.verts, self.edges = self._reduce_to_1skeleton(given_simplices)
        # self.simplex_indices = self._idx_simplices_from_1skeleton(self.verts, self.edges)
        self.max_simplices = self._max_simplices_from_1skeleton(self.verts, self.edges)
        self.max_dim = np.max([simplex.dim for simplex in self.max_simplices])

    def __contains__(self, s):
        if not isinstance(s, Simplex): raise NotImplemented
        for simplex in self.max_simplices:
            if s <= simplex:
                return True
        return False

    def _reduce_to_1skeleton(self, simplices: Set[Simplex]) -> Tuple[Set[Simplex], Set[Simplex]]:
        """Reduce simplices to 1-skeleton. Returns verts, edges"""
        verts = set()
        edges = set()
        for simplex in simplices:
            for vert in simplex.verts:
                verts.add(Simplex([vert]))
            if simplex.dim > 0:
                for vs in list(itertools.combinations(simplex.verts, 2)):
                    edges.add(Simplex(vs))
        return verts, edges                

    # def _idx_simplices_from_1skeleton(self, verts: Set[Simplex], edges: Set[Simplex]) -> dict[int: dict[Simplex: int]]:
    #     """Builds index dictionary. Returns dictionary {dimension[int] : {simplex[Simplex]: idx[int]}}."""
    #     simplex_indices_set = {0: verts, 1: edges}
    #     d = 1 # dimension
    #     faces_d = edges.copy()
    #     while len(faces_d) > d + 2:
    #         # Find all d+1 simplices
    #         faces_dp1 = set()
    #         total_vert_list = [list(vert.verts)[0] for vert in self.verts]
    #         possible_dp1_simplices = Simplex(total_vert_list).ndfacets(d+1)
    #         for ps in possible_dp1_simplices:
    #             needed_d_simplices = ps.facets()
    #             if needed_d_simplices < faces_d: # If needed_d_simplices is subset of faces_d
    #                 faces_dp1.add(ps)
            
    #         simplex_indices_set[d+1] = faces_dp1
    #         d+=1
    #         faces_d = faces_dp1

    #     simplex_indices = {}
    #     i = 0
    #     for k, simplex_set in simplex_indices_set.items():
    #         simplex_indices[k] = {}
    #         ordered_simplices = order_simplex_set(simplex_set) # sort for consistency
    #         for simplex in ordered_simplices:
    #             simplex_indices[k][simplex] = i
    #             i+=1

    #     return simplex_indices

    def _max_simplices_from_1skeleton(self, verts: Set[Simplex], edges: Set[Simplex]) -> Set[Simplex]:
        """Finds max simplices from vertices and edges (searches upwards)."""
        # Builds upwards
        max_simplices = verts | edges # set of max simplices
        d = 1 # dimension
        faces_d = edges.copy()
        while len(faces_d) > d + 1:
            for face in faces_d:
                max_simplices.add(face)
                for facet in face:
                    max_simplices.discard(facet)

            # Find all d+1 simplices
            faces_dp1 = set()
            total_vert_list = [list(vert.verts)[0] for vert in self.verts]
            possible_dp1_simplices = Simplex(total_vert_list).ndfacets(d+1)
            for ps in possible_dp1_simplices:
                needed_d_simplices = ps.facets()
                if needed_d_simplices < faces_d: # If needed_d_simplices is subset of faces_d
                    faces_dp1.add(ps)
            
            d+=1
            faces_d = faces_dp1

        # Add final simplices
        for face in faces_d:
                max_simplices.add(face)
                for facet in face:
                    max_simplices.discard(facet)

        return max_simplices
    
    def validate_simplices_exist(self, simplices: Set[Simplex]) -> bool:
        """Returns boolean whether the given simplices are in the complex."""
        for simplex in simplices:
            if simplex not in self:
                return False
        return True

    def vertex_scheme(self):
        """Return vertex scheme of graph: Set of all simplices."""
        return self.closure(self.max_simplices)
    
    def ksimplices(self, k: int):
        """Return all k-dimensional simplices in the complex."""
        ksimplices = set()
        for simplex in self.max_simplices:
            if simplex.dim < k:
                continue
            elif simplex.dim == k:
                ksimplices.add(simplex)
            else:
                ksimplices = ksimplices | simplex.ndfacets(k)
        return ksimplices
    
    def facets(self, simplices: Union[Simplex, Set[Simplex]]) -> Set[Simplex]:
        """Returns set of simplices that are facets of the given simplices."""
        if isinstance(simplices, Simplex): simplices = set([simplices]) # turn simplices into a set if it is not
        if not self.validate_simplices_exist(simplices): raise ValueError("A given simplex does not exist in the Simplicial Complex.")

        facets = set()
        for simplex in simplices:
            facets = facets | simplex.facets()
        return facets
        
    def cofaces(self, simplices: Union[Simplex, Set[Simplex]]) -> Set[Simplex]:
        """Returns set of simplices that the given simplices are facets of."""
        if isinstance(simplices, Simplex): simplices = set([simplices]) # turn simplices into a set if it is not
        if not self.validate_simplices_exist(simplices): raise ValueError("A given simplex does not exist in the Simplicial Complex.")

        cofaces = set()
        for simplex in simplices:
            for max_simplex in self.max_simplices:
                if simplex.dim >= max_simplex.dim:
                    continue
                elif simplex.dim == max_simplex.dim - 1:
                    if simplex < max_simplex:
                        cofaces.add(max_simplex)
                    continue
                elif simplex.dim < max_simplex.dim - 1:
                    facets_of_max_simplex = max_simplex.ndfacets(simplex.dim+1)
                    for facet in facets_of_max_simplex:
                        if simplex < facet:
                            cofaces.add(facet)
        return cofaces

    def star(self, simplices: Union[Simplex, Set[Simplex]]):
        """Returns set of all simplices that are 'above' given simplices."""
        if isinstance(simplices, Simplex): simplices = set([simplices]) # turn simplices into a set if it is not
        if not self.validate_simplices_exist(simplices): raise ValueError("A given simplex does not exist in the Simplicial Complex.")

        star_set = set()
        for simplex in simplices:
            for max_simplex in self.max_simplices:
                if not simplex <= max_simplex:
                    continue
                d = max_simplex.dim
                ndfacets_of_max_simplex = max_simplex.ndfacets(d)
                while d >= simplex.dim:
                    d-=1
                    ndfacets_of_max_simplex = ndfacets_of_max_simplex | max_simplex.ndfacets(d)
                for ndfacet in ndfacets_of_max_simplex:
                    if simplex <= ndfacet:
                        star_set.add(ndfacet)
        return star_set

    def closure(self, simplices: Union[Simplex, Set[Simplex]]) -> Set[Simplex]:
        """Returns set of all simplices that are 'below' given simplices."""
        if isinstance(simplices, Simplex): simplices = set([simplices]) # turn simplices into a set if it is not
        if not self.validate_simplices_exist(simplices): raise ValueError("A given simplex does not exist in the Simplicial Complex.")

        closure_set = set()
        iterate_set = simplices.copy()
        while len(iterate_set) > 0:
            simplex = iterate_set.pop()
            closure_set.add(simplex)
            if simplex.dim > 0:
                iterate_set = iterate_set | simplex.facets()
        return closure_set

    def link(self, simplices: Union[Simplex, Set[Simplex]]) -> Set[Simplex]:
        """Returns link function (generalization of the neighborhood) of the given simplices."""
        if isinstance(simplices, Simplex): simplices = set([simplices]) # turn simplices into a set if it is not
        if not self.validate_simplices_exist(simplices): raise ValueError("A given simplex does not exist in the Simplicial Complex.")

        star = self.star(simplices)
        return self.closure(star) - star



def build_simplicial_complex(mat, columns = None):
    # Create edges of adjacency matrix
    if columns is None: columns = range(0, mat.shape[0])
    if not len(columns) == mat.shape[0]: raise ValueError

    simplices = set()

    for i in range(0, mat.shape[0]):
        simplices.add(Simplex([columns[i]]))

    for i in range(0, mat.shape[0]):
        for j in range(0, mat.shape[1]):
            if i < j and mat[i,j] == 1:
                simplices.add(Simplex([columns[i],columns[j]]))

    return SimplicialComplex(simplices)


def order_simplex_set(simplex_set) -> List[Simplex]:
    """Deterministically orders simplex set into list."""
    return sorted(list(simplex_set), key= lambda s: (s.dim, s.ordered()))


if __name__ == "__main__":
    mat = np.array([[0,1,1,0,0],[1,0,1,0,0],[1,1,0,1,0],[0,0,1,0,1],[0,0,0,1,0]])
    print(mat)
    sc = build_simplicial_complex(mat, columns=list('abcde'))
    # print(sc.simplex_indices)

    simplex_set = set([Simplex('abc'), Simplex('ab'), Simplex('ba'), Simplex('bc'), Simplex('ac')])
    print(order_simplex_set(simplex_set))
