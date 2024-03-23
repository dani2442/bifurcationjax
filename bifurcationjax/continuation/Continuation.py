import jax
import jax.numpy as jnp

from bifurcationjax.BifurcationProblem import BifurcationProblem
from bifurcationjax.continuation import Corrector
from bifurcationjax.continuation import Predictor


def continuation(prob: BifurcationProblem, prediction: Predictor, correction: Corrector, p_min: float, p_max: float, dsmax: float = 1e-2, max_steps: int = 1000):
    J = jax.jit(jax.jacobian(prob.f))
    z = jnp.append(prob.x0, prob.p0)
    zs = []

    stability = []
    for _ in range(max_steps):
        zpred, v = prediction(z, dsmax, prob.f)
        if p_min>zpred[-1] or p_max<zpred[-1]:
            success = False
        else:
            z, success = correction(zpred, prob.f, v, dsmax)

        if not success: break
        zs += [z]

        eigenvalues = jnp.linalg.eigvals(J(z[:-1], z[-1]))  
        isstable = jnp.max(eigenvalues.real)<0
        stability += [isstable]

    xs = [z[:-1] for z in zs]
    ps = [z[-1] for z in zs]

    return xs, ps, stability