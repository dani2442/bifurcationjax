"""
Example shown in https://bifurcationkit.github.io/BifurcationKitDocs.jl/dev/educational/
"""

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

from bifurcationjax.continuation.Corrector import CorrectorParams
from bifurcationjax.continuation.Predictor import PredictorParams
from bifurcationjax.continuation.Continuation import continuation
from bifurcationjax.BifurcationProblem import BifurcationProblem
from bifurcationjax.utils.Branch import ContinuationPar
from bifurcationjax.utils.plot import plot_bifurcation_diagram


@jax.jit
def maasch_rule(u, p):
    x, y, z = u[...,0], u[...,1], u[...,2]
    q, r, s, = 1.2, 0.8, 0.8
    dx = -x - y
    dy = -p*z + r*y + s*z*z - z*z*y
    dz = -q*(x + z)
    return jnp.stack([dx, dy, dz], axis=-1)


def plot_fn(p):
    return p.z[0]

p0 = 0.0
x0 = jnp.array([-1.4, -1.4, -1.4])

prob = BifurcationProblem(maasch_rule, x0, p0,)
par = ContinuationPar(p_min=-0.1, p_max=2., dsmax=0.1, max_steps=500)
prediction_params = PredictorParams(method='tangent', k=0)
correction_params = CorrectorParams(method='PALC', epsilon=1e-3)
branches = continuation(prob, prediction_params, correction_params, par, max_depth=1, k_start=0)


plot_bifurcation_diagram(branches, plot_fn=plot_fn, path_save='images/example4.png')
plt.show()