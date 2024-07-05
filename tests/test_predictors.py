"""Test Correctors."""

import jax.numpy as jnp
import jax

import matplotlib.pyplot as plt

from bifurcationjax.continuation.Corrector import CorrectorParams
from bifurcationjax.continuation.Predictor import PredictorParams
from bifurcationjax.continuation.Continuation import continuation
from bifurcationjax.BifurcationProblem import BifurcationProblem
from bifurcationjax.utils.Branch import ContinuationPar
from bifurcationjax.utils.plot import plot_bifurcation_diagram


"""
Solving mu + x - x^3/3 = 0
"""

@jax.jit
def F(x, mu):
    return mu + x - jnp.pow(x, 3)/3

def plot_fn(p):
    return p.z[0]

p0 = 0.
x0 = jnp.array([-2.])

dz0 = jnp.array([0.,1.])

prob = BifurcationProblem(F, x0, p0,)
par = ContinuationPar(p_min=-1., p_max=1., dsmax=0.1, max_steps=500)

fig, axs = plt.subplots(1, 2, figsize=(15, 6))
for method, ax in zip(['tangent', 'secant'], axs.reshape(-1)):
    correction_params = CorrectorParams(method='PALC', epsilon=1e-3)
    prediction_params = PredictorParams(method=method, k=0, dz0=dz0)
    diagram = continuation(prob, prediction_params, correction_params, par, max_depth=1)
    plot_bifurcation_diagram(diagram, plot_fn=plot_fn, ax=ax, title=method)
plt.show()