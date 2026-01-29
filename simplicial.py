from typing import Union, List, Self, Type, Set, FrozenSet, Tuple, Iterable, Type
import numpy as np
import itertools
        

class Vertices(frozenset):
    def __repr__(self):
        rep = "V{" + f"{",".join([str(v) for v in self.ordered()])}" + "}"
        return rep
    @property
    def dim(self): return len(self) - 1
    def ordered(self) -> Tuple:
        """Orders vertices into a list."""
        return sorted(tuple(self))
    def closure(self):
        """Return Simplex object of Vertices"""
        return Simplex(self)

class Simplex(Vertices):
    def __repr__(self):
        rep = "S{" + f"{",".join([str(v) for v in self.ordered()])}" + "}"
        return rep
    def __contains__(self, o):
        if isinstance(o, (set,frozenset)):
            return o <= self.expand()
        else: return NotImplemented
    def __sub__(self, o):
        if isinstance(o, Simplex):
            return Vertices(self.expand() - o.expand())
        elif isinstance(o, Vertices):
            return Vertices(self.expand() - o)
        else: return NotImplemented

    def facets(self) -> SimplexSet:
        return self.ndfacets(self.dim-1)
    def ndfacets(self, facet_dim: int) -> SimplexSet:
        if facet_dim > self.dim: raise ValueError(f"Dimension of facet should be less than dimension of simplex. Given {facet_dim}, Expected <= {self.dim}.")
        ndfacets = SimplexSet({Simplex(verts) for verts in list(itertools.combinations(self, facet_dim+1))})
        return ndfacets
    def expand(self, k: int=0) -> VerticesSet:
        "Expand to dimension k."
        if k > self.dim:
            raise ValueError(f"K should not be greater than the current dimension. Given k={k}, expected <= {self.dim}")
        else:
            expansion = VerticesSet()
            while k <= self.dim:
                facets = self.ndfacets(k)
                for facet in facets:
                    expansion.add(Vertices(facet))
                k+=1
        return expansion
    def closure(self):
        """Simplices are represented as the closure of vertices."""
        return self

class VerticesSet(set):
    def __init__(self, iterable: Iterable[Vertices]=None):
        corrected = []
        if iterable:
            for item in iterable:
                corrected.append(self._validate(item))
        super().__init__(corrected)
    def __repr__(self):
        rep = "VerticesSet{" + f"{",".join([str(v) for v in self.ordered()])}" + "}"
        return rep
    def __and__(self, value):
        return type(self)(super().__and__(value))
    def __or__(self, value):
        return type(self)(super().__or__(value))

    def _validate(self, value):
        if not isinstance(value, Vertices):
            try:
                return Vertices(value)
            except:
                raise TypeError(f"Expected Vertices, got {value.__class__}.")
        return value
        if isinstance(o, Vertices):
            for vertices in self:
                if vertices > o:
                    return True
            return False
        elif isinstance(o, VerticesSet):
            count = 0
            for o_vertices in o:
                for vertices in self:
                    if vertices > o_vertices:
                        count += 1
                        break
            if count == len(o):
                return True
            else:
                return False
    
    def add(self, element):
        element = self._validate(element)
        super().add(element)

    @property
    def dim(self): return max([s.dim for s in self])

    def kvertices(self, k: int) -> VerticesSet:
        """Return all vertices of dimension k in the vertices set."""
        return VerticesSet([vertices for vertices in self if vertices.dim == k])
    
    def closure(self) -> SimplexSet:
        """Return SimplexSet of VerticesSet (returns the closure)."""
        return SimplexSet([Simplex(vertices) for vertices in self])
    
    def ordered(self) -> List:
        """Deterministically orders simplex set into list."""
        return sorted(list(self), key= lambda s: (s.dim, s.ordered()))
    

class SimplexSet(VerticesSet):
    def __repr__(self):
        rep = "SimplexSet{" + f"{",".join([str(v) for v in self.ordered()])}" + "}"
        return rep

    def _validate(self, value):
        if not isinstance(value, Simplex):
            try:
                return Simplex(value)
            except:
                raise TypeError(f"Expected Simplex, got {value.__class__}.")
        return value
    def __contains__(self, o):
        if isinstance(o, (set, frozenset)):
            if o <= self.expand():
                return True
        else: return NotImplemented
        return False
    def __le__(self, o):
        if isinstance(o, (Simplex, SimplexSet)):
            if self <= o.expand():
                return True
        elif isinstance(o, (set, frozenset)):
            return super().__le__(o)
        else: return NotImplemented
    def __lt__(self, o):
        if isinstance(o, (Simplex, SimplexSet)):
            if self < o.expand():
                return True
        elif isinstance(o, (set, frozenset)):
            return super().__lt__(o)
        else: return NotImplemented
    def __ge__(self, o):
        if isinstance(o, (Simplex, SimplexSet)):
            if self > o.expand():
                return True
        elif isinstance(o, (set, frozenset)):
            return super().__ge__(o)
        else: return NotImplemented
    def __gt__(self, o):
        if isinstance(o, (Simplex, SimplexSet)):
            if self > o.expand():
                return True
        elif isinstance(o, (set, frozenset)):
            return super().__gt__(o)
        else: return NotImplemented
    def __sub__(self, o):
        if isinstance(o, SimplexSet):
            return VerticesSet(self.expand() - o.expand())
        elif isinstance(o, (set, frozenset)):
            return VerticesSet(self.expand() - o)
        else: return NotImplemented
    
    # Overwrites
    def add(self, element):
        element = self._validate(element)
        super().add(element)

    def expand(self, k: int=0) -> VerticesSet:
        """Expand the set of simplices to a Vertices Set down to a given dimension k."""
        expansion = VerticesSet()
        for simplex in self:
            simplex_expansion = simplex.expand(k)
            for se in simplex_expansion:
                expansion.add(se)
        return expansion
    
    def closure(self):
        """Simplices are already representations of vertices after closure."""
        return self

    def between(self, simplex1: Simplex, simplex2: Simplex) -> VerticesSet:
        """Returns all vertices between two simplices from different dimensions (inclusive)."""
        if not SimplexSet([simplex1, simplex2]) <= self: raise ValueError("Given simplices must be a subset of a Simplex in SimplexSet")
        dim1, dim2 = simplex1.dim, simplex2.dim
        
        if simplex1 == simplex2:
            return VerticesSet([simplex1])
        elif simplex1 < simplex2:
            return VerticesSet([vertices for vertices in simplex2.expand(dim1) if simplex1 <= vertices])
        elif simplex2 < simplex1:
            return VerticesSet([vertices for vertices in simplex1.expand(dim2) if simplex2 <= vertices])
        else:
            return VerticesSet()

    def ksimplices(self, k: int) -> SimplexSet:
        """Return all simplices of dimension k in the simplex set."""
        ksimplices = SimplexSet()
        for simplex in self:
            if simplex.dim < k:
                continue
            elif simplex.dim == k:
                ksimplices.add(simplex)
            else:
                kfacets = kfacets | simplex.ndfacets(k)
        return ksimplices

    def facets(self, simplices: Union[Simplex, Iterable[Simplex]]) -> SimplexSet:
        if not isinstance(simplices, Iterable): simplices = [simplices]
        if not isinstance(simplices, SimplexSet): simplices = SimplexSet(simplices)
        if not simplices <= self: raise ValueError("Given simplices must be a subset of a Simplex in SimplexSet")

        facets = SimplexSet()
        for simplex in simplices:
            facets = facets | simplex.facets()
        return facets

    def cofaces(self, simplices: Union[Simplex, Iterable[Simplex]]) -> SimplexSet:
        if not isinstance(simplices, Iterable): simplices = [simplices]
        if not isinstance(simplices, SimplexSet): simplices = SimplexSet(simplices)
        if not simplices <= self: raise ValueError("Given simplices must be a subset of a Simplex in SimplexSet")

        cofaces = SimplexSet()
        for simplex in simplices:
            star = self.star(simplex)
            simplex_cofaces = SimplexSet([Simplex(vertices) for vertices in star if vertices.dim == simplex.dim+1])
            cofaces = cofaces | simplex_cofaces
        return cofaces

    def star(self, simplices: Union[Simplex, Iterable[Simplex]]) -> VerticesSet:
        if not isinstance(simplices, Iterable): simplices = [simplices]
        if not isinstance(simplices, SimplexSet): simplices = SimplexSet(simplices)
        if not simplices <= self: raise ValueError("Given simplices must be a subset of a Simplex in SimplexSet")

        star = VerticesSet()
        for given_simplex in simplices:
            for self_simplex in self:
                if given_simplex <= self_simplex:
                    between = self.between(given_simplex, self_simplex)
                    for vertices in between:
                        star.add(vertices)
        return star

    def link(self, simplices: Union[Simplex, Iterable[Simplex]]) -> SimplexSet:
        if not isinstance(simplices, Iterable): simplices = [simplices]
        if not isinstance(simplices, SimplexSet): simplices = SimplexSet(simplices)
        if not simplices <= self: raise ValueError("Given simplices must be a subset of a Simplex in SimplexSet")

        star = self.star(simplices)
        return SimplexSet(star.closure() - star)


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

    return SimplexSet(simplices)



if __name__ == "__main__":
    simplex_set = SimplexSet([Simplex('abc'),Simplex('cd'),Simplex('de')])
    link = simplex_set.link([Simplex('c'), Simplex('e')])
    cofaces = simplex_set.cofaces([Simplex('b'),Simplex('ac')])
