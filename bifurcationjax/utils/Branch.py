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
    max_steps: int = 100
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
    def __init__(self, 
                 branches = [], 
                 bps = dict()):
        self.branches = branches
        self.bps = bps
        

    def get_bp(self, p: Point) -> Optional[Point]:
        for pb in self.bps.keys():
            if p.similar(pb):
                return pb
        return None
    
    def merge(self, branch2):
        self.branches += branch2.branches

    def add_bp(self, p: Point, branches: List[Branch]):
        result = self.get_bp(p)

        branches_set = set(branch.id for branch in branches)
        if result is None:
            self.bp[p] = branches_set
        else:
            self.bp[p].update(branches_set)

