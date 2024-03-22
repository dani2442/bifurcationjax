from typing import Callable, Optional
import jax

class BifurcationProblem:
    def __init__(self, F: Callable, x0: jax.Array, mu0: float, dx0: Optional[jax.Array] = None, dp0: Optional[float] = None):
        self.F = F
        self.x0 = x0
        self.mu0 = mu0
        self.dx0 = dx0
        self.dp0 = dp0