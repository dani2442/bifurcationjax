from dataclasses import dataclass
import jax
from typing import List, Optional, Dict, Set


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
    z: Optional[jax.Array] = None
    itnewton: Optional[int] = None
    itlinear: Optional[int] = None
    ds: Optional[float] = None
    n_unstable: Optional[int] = None
    n_imag: Optional[int] = None
    stable: Optional[bool] = None
    step: Optional[int] = None
    evals: Optional[jax.Array] = None
    evecs: Optional[jax.Array] = None
    tp: Optional[str] = None


class Branch:
    current_count: int = 0

    def __init__(self, points: List[Point] = []):
        self.id = Branch.current_count
        Branch.current_count += 1

        self.points: List[Point] = points
        self.specialpoint_id: List[int] = []
        self.alg = ''
        
    def add(self, p: Point):
        self.points.append(p)
        

class Diagram:
    branches: List[Branch] = []
    bp: Dict[Point, Set[int]]

    def get_bp(p: Point) -> Optional[Point]:
        return False

    def add_bp(p: Point, branches: List[Branch]):
        result = Diagram.get_bp(p)

        branches_set = set(branch.id for branch in branches)
        if result is None:
            Diagram.bp[p] = branches_set
        else:
            Diagram.bp[p].update(branches_set)

