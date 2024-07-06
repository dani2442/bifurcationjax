"""Test branch switching. Example 2 shown in https://bifurcationkit.github.io/BifurcationKitDocs.jl/stable/gettingstarted/"""

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
Solving x*(mu-x)
"""

@jax.jit
def F(x, mu):
    return x*(mu - x)

def plot_fn(p):
    return p.z[0]

p0 = -0.2
x0 = jnp.array([0.])

prob = BifurcationProblem(F, x0, p0,)

fig, axs = plt.subplots(1, 2, figsize=(15, 6))
for method, ax in zip(['normal_orthogonal_direction', 'crandall_rabinowitz'], axs):
    correction_params = CorrectorParams(method='PALC', epsilon=1e-3)
    prediction_params = PredictorParams(method='tangent', k=1)
    par = ContinuationPar(p_min=-1., p_max=1., dsmax=0.01, max_steps=500, branch_switch=method)
    diagram = continuation(prob, prediction_params, correction_params, par, max_depth=2)
    plot_bifurcation_diagram(diagram, plot_fn=plot_fn, ax=ax, title=method)
plt.show()