import jax
import jax.numpy as jnp


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
    return phi_2 /jnp.linalg.norm(phi_2, ord=2)


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
