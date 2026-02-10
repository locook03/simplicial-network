from typing import Union, List, Set, FrozenSet, Tuple, Iterable, Dict, Optional, Any
from functools import cached_property
import itertools

import numpy as np


class Point(frozenset):
    """
    A vertex-set with an optional 'distance' attribute. Dimension is |verts|-1.

    Note:
      - Point is a frozenset so it is hashable and can be used as dict keys.
      - 'distance' is carried as a convenience (e.g., for Rips complexes).
    """
    def __new__(cls, iterable: Iterable = None, distance: float = None):
        instance = super().__new__(cls, iterable if iterable is not None else ())
        if distance is not None:
            instance.distance = distance
        else:
            instance.distance = float('inf')
        return instance

    def __repr__(self):
        if self.distance != float('inf'):
            rep = f"P({self.distance:.2f})" + "{" + f"{','.join([str(v) for v in self.ordered()])}" + "}"
        else:
            rep = "P{" + f"{','.join([str(v) for v in self.ordered()])}" + "}"
        return rep

    @property
    def dim(self) -> int:
        return len(self) - 1

    def ordered(self) -> Tuple:
        """Deterministically orders vertices."""
        return tuple(sorted(tuple(self)))

    def closure(self) -> "Simplex":
        """Return Simplex object of Point."""
        return Simplex(self)

    def path(self, to: "Point") -> "PointSet":
        """
        Returns PointSet of Points between (across dimensions) this Point and the given Point. INCLUSIVE.
        """
        if not isinstance(to, Point):
            raise TypeError(f"'to' must be of type Point, got {type(to)}")

        if self == to:
            return PointSet([self])
        elif self < to:
            return PointSet([point for point in to.closure().expand(self.dim) if self <= point])
        elif to < self:
            return PointSet([point for point in self.closure().expand(to.dim) if to <= point])
        else:
            return PointSet()


class Simplex(Point):
    """
    A simplex is represented by its vertex set (a frozenset) with an optional distance attribute.
    """
    def __repr__(self):
        if self.distance != float('inf'):
            rep = f"S({self.distance:.2f})" + "{" + f"{','.join([str(v) for v in self.ordered()])}" + "}"
        else:
            rep = "S{" + f"{','.join([str(v) for v in self.ordered()])}" + "}"
        return rep

    def __contains__(self, o):
        if isinstance(o, (set, frozenset)):
            return o <= self.expand()
        return NotImplemented

    def __sub__(self, o):
        if isinstance(o, Simplex):
            return Point(self.expand() - o.expand())
        elif isinstance(o, Point):
            return Point(self.expand() - o)
        return NotImplemented

    def facets(self) -> "SimplicialComplex":
        return self.ndfacets(self.dim - 1)

    def ndfacets(self, facet_dim: int) -> "SimplicialComplex":
        if facet_dim > self.dim:
            raise ValueError(
                f"Dimension of facet should be <= dimension of simplex. Given {facet_dim}, expected <= {self.dim}."
            )
        ndfacets = SimplicialComplex({Simplex(verts) for verts in itertools.combinations(self, facet_dim + 1)})
        return ndfacets

    def expand(self, k: int = 0) -> "PointSet":
        """Expand simplex to include all faces down to dimension k."""
        if k > self.dim:
            raise ValueError(f"k should not be greater than the current dimension. Given k={k}, expected <= {self.dim}")
        expansion = PointSet()
        cur = k
        while cur <= self.dim:
            expansion |= self.ndfacets(cur)
            cur += 1
        return expansion

    def closure(self) -> "SimplicialComplex":  # overwrites Point.closure
        """Simplices are represented as the closure of point."""
        return self


class PointSet(set):
    def __init__(self, iterable: Iterable[Point] = None):
        corrected = []
        if iterable:
            for item in iterable:
                corrected.append(self._validate(item))
        super().__init__(corrected)

    def __repr__(self):
        rep = "PointSet{" + f"{','.join([str(v) for v in self.ordered()])}" + "}"
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
            except Exception:
                raise TypeError(f"Expected Point, got {value.__class__}.")
        return value

    def add(self, element):
        element = self._validate(element)
        super().add(element)

    @property
    def dim(self) -> int:
        return max([s.dim for s in self]) if len(self) else -1

    def kpoints(self, k: int) -> "PointSet":
        """Return all points of dimension k in the point set."""
        return PointSet([point for point in self if point.dim == k])

    def closure(self) -> "SimplicialComplex":
        """Return SimplicialComplex of PointSet (returns the closure)."""
        return SimplicialComplex([Simplex(point) for point in self])

    def ordered(self) -> List:
        """Deterministically orders set into list."""
        return sorted(list(self), key=lambda s: (getattr(s, 'distance', float('inf')), -s.dim, s.ordered()))


class SimplicialComplex(PointSet):
    """
    Family of Simplexes. Family of sets that is closed under taking subsets.
    """
    def __repr__(self):
        rep = "SimplicialComplex{" + f"{','.join([str(v) for v in self.ordered()])}" + "}"
        return rep

    def __contains__(self, o):
        if isinstance(o, (set, frozenset)):
            return o <= self.expand()
        return NotImplemented

    def __sub__(self, o):
        if isinstance(o, SimplicialComplex):
            return PointSet(self.expand() - o.expand())
        elif isinstance(o, (set, frozenset)):
            return PointSet(self.expand() - o)
        return NotImplemented

    def _validate(self, value):
        if not isinstance(value, Simplex):
            try:
                return Simplex(value)
            except Exception:
                raise TypeError(f"Expected Simplex, got {value.__class__}.")
        return value

    def expand(self, k: int = 0) -> PointSet:
        """Expand the set of simplices to a PointSet down to a given dimension k."""
        expansion = PointSet()
        for simplex in self:
            expansion |= simplex.expand(k)
        return expansion

    def closure(self) -> "SimplicialComplex":  # overwrites
        """Simplices are already representations of points after closure."""
        return self

    def ksimplices(self, k: int) -> "SimplicialComplex":
        """Return all simplices of dimension k in the set."""
        ks = SimplicialComplex()
        for simplex in self:
            if simplex.dim < k:
                continue
            elif simplex.dim == k:
                ks.add(simplex)
            else:
                ks |= simplex.ndfacets(k)
        return ks

    def facets(self) -> "SimplicialComplex":
        """Return the facets of the set."""
        facets = SimplicialComplex()
        for simplex in self:
            facets |= simplex.facets()
        return facets

    def cofaces(self) -> "SimplicialComplex":
        """Return the cofaces of the set (simplices for which these are facets)."""
        cofaces = SimplicialComplex()
        for simplex in self:
            star = self.star()  # star of entire set
            simplex_cofaces = SimplicialComplex([Simplex(point) for point in star if point.dim == simplex.dim + 1])
            cofaces |= simplex_cofaces
        return cofaces

    def star(self) -> PointSet:
        """Return a PointSet of all points that exist 'above' the set."""
        supersets = find_supersets(self)
        star = PointSet()
        for simplex in self:
            for max_simplex in supersets:
                star |= simplex.path(max_simplex)
        return star

    def link(self) -> "SimplicialComplex":
        """Link = closure(star) - star."""
        star = self.star()
        return SimplicialComplex(star.closure() - star)


class Chain(SimplicialComplex):
    """
    A chain over F2: a set of simplices (symmetric-difference addition).
    Enforces all simplices have the same dimension.
    """
    def __repr__(self):
        rep = "Chain{" + f"{','.join([str(v) for v in self.ordered()])}" + "}"
        return rep

    def _validate(self, value):
        value = super()._validate(value)
        if "dim" in self.__dict__ and self.dim != value.dim:
            raise ValueError(f"Simplices in Chain must share dimension. Expected {self.dim}, got {value.dim}.")
        return value

    def boundary(self) -> "Chain":
        """Boundary over F2 (XOR of facets)."""
        boundary = Chain()
        for simplex in self:
            for facet in simplex.facets():
                if facet in boundary:
                    boundary.remove(facet)
                else:
                    boundary.add(facet)
        return boundary


def find_supersets(simplex_set: SimplicialComplex, max_only: bool = True) -> SimplicialComplex:
    """
    Given a set containing simplices (not necessarily maximal), return only maximal simplices.

    This function grows higher-dimensional simplices when all faces exist (clique/flag style),
    and prunes faces that are contained in higher-dimensional simplices.
    """
    verts = simplex_set.ksimplices(0)
    edges = simplex_set.ksimplices(1)

    supersets = verts | edges
    supersets -= edges.facets()

    d = 1
    faces_d = edges
    while len(faces_d) > d + 1:
        faces_dp1 = set()
        for combo in itertools.combinations(faces_d, d + 2):
            ps = set()
            ps.update(*[set(elem) for elem in combo])
            ps = Simplex(ps, distance=max([s.distance for s in combo]))
            needed_d_simplices = ps.facets()
            if needed_d_simplices <= faces_d:
                faces_dp1.add(ps)
                supersets.add(ps)
                if max_only:
                    supersets -= needed_d_simplices
        d += 1
        faces_d = faces_dp1

    return supersets