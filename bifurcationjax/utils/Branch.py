from dataclasses import dataclass
import jax
from typing import List, Optional


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
    id_current = 0

    def __init__(self, points: List[Point] = []):
        self.id = Branch.id_current
        Branch.id_current += 1

        self.points: List[Point] = points
        self.specialpoint_id: List[int] = []
        self.alg = ''
        
    def add(self, p: Point):
        self.points.append(p)
        

class Branches:
    branches: List[Branch] = []
