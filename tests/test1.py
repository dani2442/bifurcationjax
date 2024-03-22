import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

from bifurcationjax.continuation.Corrector import NaturalCorrector, PALC
from bifurcationjax.continuation.Predictor import SecantPredictor, TangentPredictor, NaturalPredictor
from bifurcationjax.continuation.Continuation import continuation
from bifurcationjax.BifurcationProblem import BifurcationProblem

@jax.jit
def maasch_rule(u, p):
    x, y, z = u[...,0], u[...,1], u[...,2]
    q, r, s, = 1.2, 0.8, 0.8
    dx = -x - y
    dy = -p*z + r*y + s*z*z - z*z*y
    dz = -q*(x + z)
    return jnp.stack([dx, dy, dz], axis=-1)


p_min = -0.1
p_max = 2
delta = 0.9
p0 = 0.0
x0 = jnp.array([-1.4, -1.4, -1.4])
dp0 = 0.02
dx0 = jnp.array([0.01, 0.01, 0.01])
dz0 = jnp.append(dx0, dp0)

### Test 1
prob = BifurcationProblem(maasch_rule, x0, p0, dx0, dp0)
correction = NaturalCorrector()
prediction = SecantPredictor(dz0)
xs, ps, stability = continuation(prob, correction, prediction, p_min, p_max)

colors = ["blue" if s else "red" for s in stability]
plt.scatter(ps, [x[0] for x in xs], c=colors)
plt.show()

### Test 2
correction = PALC()
prediction = SecantPredictor(dz0)
xs, ps, stability = continuation(prob, correction, prediction, p_min, p_max)

colors = ["blue" if s else "red" for s in stability]
plt.scatter(ps, [x[0] for x in xs], c=colors)
plt.show()

### Test 3
correction = PALC()
prediction = TangentPredictor(dz0)
xs, ps, stability = continuation(prob, correction, prediction, p_min, p_max)

colors = ["blue" if s else "red" for s in stability]
plt.scatter(ps, [x[0] for x in xs], c=colors)
plt.show()


### Test 4
correction = PALC()
prediction = NaturalPredictor(dz0)
xs, ps, stability = continuation(prob, correction, prediction, p_min, p_max)

colors = ["blue" if s else "red" for s in stability]
plt.scatter(ps, [x[0] for x in xs], c=colors)
plt.show()