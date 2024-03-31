from dataclasses import dataclass, field
import uuid
import jax
import jax.numpy as jnp
from typing import List, Optional, Dict, Set
from collections import deque

# same point norm distance
precision = 1e-1 


@dataclass
class NewtonPar:
    tol: float = 1e-5


@dataclass
class ContinuationPar:
    p_min: float
    p_max: float
    dsmin: float = 1e-3
    dsmax: float = 1e-2
    ds: float    = 1e-3
    n_inversion: int = 5
    new: int = 1
    max_steps: int = 1000
    newton_options: NewtonPar = NewtonPar()


@dataclass
class Point:
    z: jax.Array | None = None
    itnewton: int | None = None
    itlinear: int | None = None
    ds: float | None = None
    n_unstable: int | None = None
    n_imag: int | None = None
    stable: bool | None = None
    step: int | None = None
    evals: jax.Array | None = None
    evecs: jax.Array | None = None
    tp: str | None = None
    id: Optional[str] = field(default_factory = lambda: str(uuid.uuid1()))

    def __hash__(self):
        return hash(self.id)
    
    def similar(self, other):
        return jnp.linalg.norm(self.z - other.z, ord=2)<precision


class Branch:
    current_count: int = 0

    def __init__(self, points: List[Point] = []):
        self.id = Branch.current_count
        Branch.current_count += 1

        self.points: List[Point] = deque(points)
        self.specialpoint_id: List[int] = []
        self.alg = ''
        
    def add(self, p: Point, forward: bool = True):
        if forward:
            self.points.append(p)
        else:
            self.points.appendleft(p)
        

class Diagram:
    branches: List[Branch] = []
    bps: Dict[Point, Set[int]] = dict()

    def get_bp(p: Point) -> Optional[Point]:
        for pb in Diagram.bps.keys():
            if p.similar(pb):
                return pb
        return None

    def add_bp(p: Point, branches: List[Branch]):
        result = Diagram.get_bp(p)

        branches_set = set(branch.id for branch in branches)
        if result is None:
            Diagram.bp[p] = branches_set
        else:
            Diagram.bp[p].update(branches_set)

