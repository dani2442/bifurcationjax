import jax.numpy as jnp
import jax
from functools import partial

@partial(jax.jit, static_argnums=(1,2,3))
def mixed_jacobian(z, k, f, J):
    x, p = z[:-1], z[-1]
    j = J(x, p)

    pder = jax.jacobian(f, argnums=1)(x, p)
    Jmixed = jnp.column_stack([j, pder])

    last_row = jnp.eye(len(z), M=1, k=-k)
    Jfinal = jnp.concat([Jmixed, last_row.T])
    return Jfinal

@partial(jax.jit, static_argnums=(2,3,4))
def newton_step(z, zpred, i, f, J, delta):
    Jfinal = mixed_jacobian(z, i, f, J)
    x = z[:-1]
    p = z[-1]
    g = f(x, p)
    gz = jnp.append(g, z[i] - zpred[i])
    z = z - delta*jnp.linalg.inv(Jfinal) @ gz
    return z

def corrector(zpred, f, J, delta=0.9, max_steps=200, epsilon=1e-6, k=0):
    c = 0
    z0 = zpred
    z1 = newton_step(z0, zpred, k, f, J, delta)
    while jnp.linalg.norm(z1 - z0, ord=2)>epsilon:
        z0 = z1
        z1 = newton_step(z0, zpred, k, f, J, delta)
        c+=1
        if c>max_steps:
            print("Newton did not converge")
            return z1, False
    return z1, True

def predictor(zs, dz0):
    if len(zs) == 1:
        return zs[-1]
    elif len(zs) == 2:
        return zs[-1] + dz0
    else:
        return 2*zs[-1] - zs[-2]
    

def _continuation(zs, f, J, dz0, pmin, pmax):
    zpred = predictor(zs, dz0)
    if pmin>zpred[-1] or pmax<zpred[-1]:
        return zs, False
    z, success = corrector(zpred, f, J)
    zs += [z]
    return zs, success

def continuation(f, x0, p0, pmin, pmax, dp0, dx0, N=1000):
    J = jax.jit(jax.jacobian(f))
    z0 = jnp.append(x0, p0)
    zs = [z0]
    dz0 = jnp.append(dx0, dp0)

    ps = [p0]
    xs = [x0]
    stability = []
    for i in range(N):
        zs, success = _continuation(zs, f, J, dz0, pmin, pmax)
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


if __name__ == '__main__':
    #@jax.jit
    def maasch_rule(u, p):
        x, y, z = u[...,0], u[...,1], u[...,2]
        q, r, s, = 1.2, 0.8, 0.8
        dx = -x - y
        dy = -p*z + r*y + s*z*z - z*z*y
        dz = -q*(x + z)
        return jnp.stack([dx, dy, dz], axis=-1)

    def maasch_jacob(u, p):
        x, y, z = u[...,0], u[...,1], u[...,2]
        q, r, s = 1.2, 0.8, 0.8
        return jnp.array([[-1,-1,0],
                [0,r - jnp.square(z),-p + 2*z*s - 2*z*y],
                [-q,0,-q]])
    
    pmin = -0.1
    pmax = 2
    delta = 0.9
    p0 = 0.0
    x0 = jnp.array([-1.4, -1.4, -1.4])
    dp0 = 0.02
    dx0 = jnp.array([0.01, 0.01, 0.01])

    xs, ps, stability = continuation(maasch_rule, maasch_jacob, x0, p0, pmin, pmax, dp0, dx0)