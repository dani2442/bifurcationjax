from typing import Callable
import jax
import jax.numpy as jnp

from bifurcationjax.BifurcationProblem import BifurcationProblem
from bifurcationjax.continuation.correction import Correction
from bifurcationjax.continuation.prediction import Prediction


def continuation(prob: BifurcationProblem, correction: Correction, prediction: Prediction, p_min: float, p_max: float, dsmax: float = 1., max_steps: int = 1000):
    J = jax.jit(jax.jacobian(prob.f))
    z0 = jnp.append(prob.x0, prob.p0)
    zs = [z0]
    dz0 = jnp.append(prob.dx0, prob.dp0)

    ps = [prob.p0]
    xs = [prob.x0]
    stability = []
    for _ in range(max_steps):
        zpred, v = prediction(z, dsmax)
        if p_min>zpred[-1] or p_max<zpred[-1]:
            return zs, False
        z, success = correction(zpred, prob.f, v, dsmax)
        zs += [z]
        # Stop iteration if we exceed given parameter margins
        if not success: break
        # Detect stability of found fixed point
        eigenvalues = jnp.linalg.eigvals(J(zs[-1][:-1], zs[-1][-1]))  
        isstable = jnp.max(eigenvalues.real)<0
        stability += [isstable]

    xs = [z[:-1] for z in zs]
    ps = [z[-1] for z in zs]
    xs.pop(0)
    ps.pop(0)
    return xs, ps, stability