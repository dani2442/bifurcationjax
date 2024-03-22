from typing import Callable, Optional
import jax

class BifurcationProblem:
    def __init__(self, f: Callable, x0: jax.Array, p0: float, dx0: Optional[jax.Array] = None, dp0: Optional[float] = None):
        self.f = f
        self.x0 = x0
        self.p0 = p0
        self.dx0 = dx0
        self.dp0 = dp0