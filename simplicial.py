from typing import Union, List, Self, Type, Set, FrozenSet, Tuple, Iterable, Type
from functools import cached_property
import numpy as np
import itertools
        

class Point(frozenset):
    def __new__(cls, iterable: Iterable=None, distance: float=None):
        instance = super().__new__(cls, iterable)
        instance.distance  = distance
        return instance
    def __repr__(self):
        if self.distance: rep = f"P({self.distance:.2f})" + "{" + f"{",".join([str(v) for v in self.ordered()])}" + "}"
        else: rep = "P{"+ f"{",".join([str(v) for v in self.ordered()])}" + "}"
        return rep
    @property
    def dim(self): return len(self) - 1
    def ordered(self) -> Tuple:
        """Orders point into a list."""
        return sorted(tuple(self))
    def closure(self):
        """Return Simplex object of Point"""
        return Simplex(self)
    def path(self, to: Point) -> PointSet:
        """Returns PointSet of Points between (across dimensions) this Point and the given Point. INCLUSIVE."""
        if not isinstance(to, Point): raise TypeError(f"'to' must be of type Point, got {type(to)}")
        
        if self == to:
            return PointSet([self])
        elif self < to:
            return PointSet([point for point in to.closure().expand(self.dim) if self <= point])
        elif to < self:
            return PointSet([point for point in self.closure().expand(to.dim) if to <= point])
        else:
            return PointSet()

class Simplex(Point):
    def __repr__(self):
        rep = "S{"+ f"{",".join([str(v) for v in self.ordered()])}" + "}"
        return rep
    def __contains__(self, o):
        if isinstance(o, (set,frozenset)):
            return o <= self.expand()
        else: return NotImplemented
    def __sub__(self, o):
        if isinstance(o, Simplex):
            return Point(self.expand() - o.expand())
        elif isinstance(o, Point):
            return Point(self.expand() - o)
        else: return NotImplemented

    def facets(self) -> SimplexSet:
        return self.ndfacets(self.dim-1)
    def ndfacets(self, facet_dim: int) -> SimplexSet:
        if facet_dim > self.dim: raise ValueError(f"Dimension of facet should be less than dimension of simplex. Given {facet_dim}, Expected <= {self.dim}.")
        ndfacets = SimplexSet({Simplex(verts) for verts in list(itertools.combinations(self, facet_dim+1))})
        return ndfacets
    def expand(self, k: int=0) -> PointSet:
        "Expand to dimension k."
        if k > self.dim:
            raise ValueError(f"K should not be greater than the current dimension. Given k={k}, expected <= {self.dim}")
        else:
            expansion = PointSet()
            while k <= self.dim:
                facets = self.ndfacets(k)
                expansion |= facets
                k+=1
        return expansion
    def closure(self): # Overwrites
        """Simplices are represented as the closure of point."""
        return self


class PointSet(set):
    def __init__(self, iterable: Iterable[Point]=None):
        corrected = []
        if iterable:
            for item in iterable:
                corrected.append(self._validate(item))
        super().__init__(corrected)
    def __repr__(self):
        rep = "PointSet{" + f"{",".join([str(v) for v in self.ordered()])}" + "}"
        return rep
    def __and__(self, value):
        if type(self) == type(value):
            return type(self)(super().__and__(value))
        return PointSet(super().__and__(value))
    def __or__(self, value):
        if type(self) == type(value):
            return type(self)(super().__or__(value))
        return PointSet(super().__or__(value))

    def _validate(self, value):
        if not isinstance(value, Point):
            try:
                return Point(value)
            except:
                raise TypeError(f"Expected Point, got {value.__class__}.")
        return value
    
    def add(self, element):
        element = self._validate(element)
        super().add(element)

    @property
    def dim(self): return max([s.dim for s in self])

    def kpoints(self, k: int) -> PointSet:
        """Return all point of dimension k in the point set."""
        return PointSet([point for point in self if point.dim == k])
    
    def closure(self) -> SimplexSet:
        """Return SimplexSet of PointSet (returns the closure)."""
        return SimplexSet([Simplex(point) for point in self])
    
    def ordered(self) -> List:
        """Deterministically orders simplex set into list."""
        return sorted(list(self), key= lambda s: (s.dim, s.ordered()))
    

class SimplexSet(PointSet):
    @cached_property
    def max_simplices(self):
        return find_max_simplices(self)
    
    def __repr__(self):
        rep = "SimplexSet{" + f"{",".join([str(v) for v in self.ordered()])}" + "}"
        return rep
    def __contains__(self, o):
        if isinstance(o, (set, frozenset)):
            if o <= self.expand():
                return True
            else:
                return False
        else: return NotImplemented
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
            return PointSet(self.expand() - o.expand())
        elif isinstance(o, (set, frozenset)):
            return PointSet(self.expand() - o)
        else: return NotImplemented
    
    # Overwrites
    def add(self, element):
        element = self._validate(element)
        if not element <= self and 'max_simplices' in self.__dict__:
            del self.max_simplices
        super().add(element)

    def _validate(self, value):
        if not isinstance(value, Simplex):
            try:
                return Simplex(value)
            except:
                raise TypeError(f"Expected Simplex, got {value.__class__}.")
        return value

    def expand(self, k: int=0) -> PointSet:
        """Expand the set of simplices to a Point Set down to a given dimension k."""
        expansion = PointSet()
        for simplex in self:
            simplex_expansion = simplex.expand(k)
            expansion |= simplex_expansion
        return expansion
    
    def closure(self): # Overwrites
        """Simplices are already representations of point after closure."""
        return self

    def ksimplices(self, k: int) -> SimplexSet:
        """Return all simplices of dimension k in the simplex set."""
        ksimplices = SimplexSet()
        for simplex in self:
            if simplex.dim < k:
                continue
            elif simplex.dim == k:
                ksimplices.add(simplex)
            else:
                ksimplices |= simplex.ndfacets(k)
        return ksimplices

    def facets(self) -> SimplexSet:
        facets = SimplexSet()
        for simplex in self:
            facets |= simplex.facets()
        return facets

    def cofaces(self) -> SimplexSet:
        """Return the cofaces of the set. Cofaces are faces of which these simplices are facets."""
        cofaces = SimplexSet()
        for simplex in self:
            star = self.star(simplex)
            simplex_cofaces = SimplexSet([Simplex(point) for point in star if point.dim == simplex.dim+1])
            cofaces |= simplex_cofaces
        return cofaces

    def star(self) -> PointSet:
        max_simplices = find_max_simplices(self)
        star = PointSet()
        for simplex in self:
            for max_simplex in max_simplices:
                between = simplex.path(max_simplex)
                star |= between
        return star

    def link(self) -> SimplexSet:
        star = self.star()
        return SimplexSet(star.closure() - star)


class SimplicialComplex(SimplexSet):
    # Initialize in a way to ensure that only maximal simplices are held in the set.

    def __repr__(self):
        rep = "SimplicialComplex{" + f"{",".join([str(v) for v in self.ordered()])}" + "}"
        return rep


def build_simplicial_complex(mat, columns = None):
    # Create edges of adjacency matrix
    if columns is None: columns = range(0, mat.shape[0])
    if not len(columns) == mat.shape[0]: raise ValueError

    simplices = set()

    for i in range(0, mat.shape[0]):
        simplices.add(Simplex([columns[i]]))

    for i in range(0, mat.shape[0]):
        for j in range(0, mat.shape[1]):
            if i < j and mat[i,j] != 0:
                simplices.add(Simplex([columns[i],columns[j]], distance=float(mat[i,j])))

    return SimplicialComplex(simplices)

def find_max_simplices(simplex_set: SimplexSet) -> SimplexSet:
    verts = simplex_set.ksimplices(0)
    edges = simplex_set.ksimplices(1)

    max_simplices = verts | edges
    max_simplices -= edges.facets()
    d = 1
    faces_d = edges
    while len(faces_d) > d+1:
        faces_dp1 = set()
        candidate_vertices = [*simplex_set.ksimplices(d-1)]
        for combo in itertools.combinations(candidate_vertices, d+2):
            ps = set()
            ps.update(*[set(elem) for elem in combo])
            ps = Simplex(ps)
            needed_d_simplices = ps.facets()
            if needed_d_simplices <= faces_d:
                faces_dp1.add(ps)
                max_simplices.add(ps)
                max_simplices -= needed_d_simplices
        d+=1
        faces_d = faces_dp1

    return max_simplices
        



if __name__ == "__main__":
    mat = np.array([[0,1,1,0,0],[1,0,1,0,0],[1,1,0,1,0],[0,0,1,0,1],[0,0,0,1,0]])
    print(mat)
    sc = build_simplicial_complex(mat, columns=list('abcde'))
    print(sc.max_simplices)

    # simplex_set = SimplicialComplex([Simplex('abc'),Simplex('cd'),Simplex('de')])
    # link = simplex_set.link([Simplex('c'), Simplex('e')])
    # cofaces = simplex_set.cofaces([Simplex('b'),Simplex('ac')])
