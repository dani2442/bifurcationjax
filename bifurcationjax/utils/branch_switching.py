import jax
import jax.numpy as jnp

from bifurcationjax.continuation.Corrector import MethodIICorrector
from bifurcationjax.utils.Krylov_Newton import NewtonMethod


def normal_orthogonal_direction_method(f, z0, z1):
    v = z1 - z0
    #phi = eigenvectors[jnp.argmin(jnp.abs(eigenvalues))]
    z = (z1+z0)/2
    x, p = z[:-1], z[-1]
    j = jax.jacobian(f, argnums=0)(x, p)
    pder = jax.jacobian(f, argnums=1)(x, p)
    Jmixed = jnp.column_stack([j, pder])
    last_row = v.reshape(-1,1)
    Jfinal = jnp.concat([Jmixed, last_row.T])

    evalues1, evectors1 = jnp.linalg.eig(Jfinal)
    
    phi_2 = evectors1[jnp.argmin(jnp.abs(evalues1))].real
    return phi_2 / jnp.linalg.norm(phi_2, ord=2)


def normal_coeff(f, z0, z1):
    @jax.jit
    def G(z):
        return f(z[:-1],z[-1])

    v = z1 - z0
    #phi = eigenvectors[jnp.argmin(jnp.abs(eigenvalues))]
    z = (z1+z0)/2
    x, p = z[:-1], z[-1]
    j = jax.jacobian(f, argnums=0)(x, p)
    pder = jax.jacobian(f, argnums=1)(x, p)
    Jmixed = jnp.column_stack([j, pder])
    last_row = v.reshape(-1,1)
    Jfinal = jnp.concat([Jmixed, last_row.T])
    evalues1, evectors1 = jnp.linalg.eig(Jfinal)
    evalues2, evectors2 = jnp.linalg.eig(Jfinal.T)

    phi_1 = v
    phi_2 = evectors1[jnp.argmin(jnp.abs(evalues1))]
    psi = evectors2[jnp.argmin(jnp.abs(evalues2))][:-1]

    H = jax.hessian(G)(z)
    c_12 = psi.T @ ((H @ phi_1 ) @ phi_2 )
    c_22 = psi.T @ ((H @ phi_2 ) @ phi_2 )

    beta_2 = 1
    alpha_2 = -c_22 / (2*c_12)

    v_new = alpha_2 * phi_1 + beta_2 * phi_2
    v_new /= jnp.linalg.norm(v_new, ord=2)

    return v_new

def method_I(f, z0, z1):    
    v = z1 - z0
    z = (z1+z0)/2
    x, p = z[:-1], z[-1]

    f_u = jax.jacobian(f, argnums=0)(x, p)

    evalues1, evectors1 = jnp.linalg.eig(f_u)
    phi = evectors1[jnp.argmin(jnp.abs(evalues1))].real

    evalues2, evectors2 = jnp.linalg.eig(f_u.T)
    varphi = evectors2[jnp.argmin(jnp.abs(evalues2))].real

    f_p = jax.jacobian(f, argnums=1)(x, p)

    g_u = jnp.concat([f_u[:-1], phi])
    g_p = jnp.concat([f_p[:-1], 0.])

    phi_0 = jnp.linalg.inv(g_u) @ -g_p

    G_uu = jax.hessian(f, argnums=[0,0])(x, p)
    G_up = jax.hessian(f, argnums=[0,1])(x, p)
    G_pp = jax.hessian(f, argnums=[1,1])(x, p)

    a_11 = varphi.T @ G_uu @ phi @ phi
    a_12 = varphi.T @ (G_uu @ phi_0 + G_up) @ phi
    a_22 = varphi.T @ (G_uu @ phi_0 @ phi_0 + 2*G_up @ phi_0 + G_pp)
    

def method_II(f, z0, z1):
    v = z1 - z0
    z = (z1+z0)/2
    x, p = z[:-1], z[-1]

    f_u = jax.jacobian(f, argnums=0)(x, p)

    evalues1, evectors1 = jnp.linalg.eig(f_u)
    phi = evectors1[jnp.argmin(jnp.abs(evalues1))].real

    evalues2, evectors2 = jnp.linalg.eig(f_u.T)
    varphi = evectors2[jnp.argmin(jnp.abs(evalues2))].real

    f_p = jax.jacobian(f, argnums=1)(x, p)

    g_u = jnp.concat([f_u[:-1], phi])
    g_p = jnp.concat([f_p[:-1], 0.])

    phi_0 = jnp.linalg.inv(g_u) @ -g_p

    corrector = MethodIICorrector(v, phi, phi_0, 0.1, jnp.array([0., 0.]))
    z_pred, success = corrector(jnp.array([0., 0.]), None, f, None)

    return z_pred - z

    
def method_crandall_rabinowitz(f, z0, z1):
    """Method IV."""
    v = z1 - z0
    z = (z1+z0)/2
    x_old, p_old = z[:-1], z[-1]

    f_u = jax.jacobian(f, argnums=0)(x_old, p_old)

    evalues1, evectors1 = jnp.linalg.eig(f_u)
    phi = evectors1[jnp.argmin(jnp.abs(evalues1))].real

    evalues2, evectors2 = jnp.linalg.eig(f_u.T)
    varphi = evectors2[jnp.argmin(jnp.abs(evalues2))].real

    eps = 0.01
    @jax.jit
    def F(z):
        x, p = z[:-1], z[-1]
        x_out = 1/eps*f(x_old + eps*(phi + x), p_old + p)
        p_out = jnp.inner(varphi, x)
        return jnp.append(x_out, p_out)

    vp_pred = NewtonMethod(F, jnp.array([0., 0.]))
    v = jnp.append(eps*(phi + vp_pred[:-1]), vp_pred[-1])
    return v/jnp.linalg.norm(v, ord=2)
