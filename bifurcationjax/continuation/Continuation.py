import jax
import jax.numpy as jnp
from typing import Tuple

from bifurcationjax.BifurcationProblem import BifurcationProblem
from bifurcationjax.continuation.Corrector import Corrector, NaturalCorrector, PALC
from bifurcationjax.continuation.Predictor import Predictor, TangentPredictor
from bifurcationjax.utils.branch_switching import normal_orthogonal_direction_method
from bifurcationjax.utils.Branch import Point, Branch, Branches
from bifurcationjax.utils import get_bifurcation_type, is_stable


""" def continuation(prob: BifurcationProblem, prediction: Predictor, correction: Corrector, p_min: float, p_max: float, dsmax: float = 1e-2, max_steps: int = 1000):
    J = jax.jit(jax.jacobian(prob.f))
    z = jnp.append(prob.x0, prob.p0)
    dz = prob.dz0

    z, success = NaturalCorrector(delta=0.1, k=1)(z, None, prob.f, dz, dsmax)
    zs = [z]
    tps = [None]

    eigenvalues, eigenvectors = jnp.linalg.eig(J(z[:-1], z[-1])) 
    eigenvalues_list = [eigenvalues]

    n_unstable_prev, n_imag_prev, isstable = is_stable(eigenvalues)
    stability = [isstable]

    for _ in range(max_steps):
        zpred, v = prediction(z, dsmax, prob.f)
        if p_min>zpred[-1] or p_max<zpred[-1]:
            success = False
        else:
            z, success = correction(zpred, z, prob.f, v, dsmax)

        if not success: 
            print("hello")
            break
        zs += [z]

        eigenvalues = jnp.linalg.eigvals(J(z[:-1], z[-1])) 
        eigenvalues_list += [eigenvalues] 

        n_unstable, n_imag, isstable = is_stable(eigenvalues)
        known, tp = get_bifurcation_type(n_unstable, n_unstable_prev, n_imag, n_imag_prev)
        
        if tp == 'bp':
            dz_new = normal_orthogonal_direction_method(prob.f, zs[-1], zs[-2])
            z2_0 = (zs[-1] + zs[-2])/2
            z2_pred = z2_0 + dsmax*dz_new
            corrector2 = PALC(delta=0.1)
            z2, success2 = corrector2(z2_pred, z2_0, prob.f, dz_new, dsmax)
            if not success2:
                print("hello 2")
            zs += [z2]
            dz2 = z2 - z2_0
            dz2 /= jnp.linalg.norm(dz2, ord=2)
            predictor2 = TangentPredictor(dz0 = dz2)

            for i in range(1000):
                zpred2, v2 = predictor2(z2, dsmax, prob.f)
                z2, success2 = corrector2(zpred2, z2, prob.f, v2, dsmax)
                if not success2:
                    print("hello 2 (loop)")
                zs += [z2]

        n_unstable_prev, n_imag_prev = n_unstable, n_imag
        tps += [tp]
        stability += [isstable]

    xs = [z[:-1] for z in zs]
    ps = [z[-1] for z in zs]

    return xs, ps, stability, eigenvalues_list, tps """


def continuation(prob: BifurcationProblem, prediction: Predictor, correction: Corrector, p_min: float, p_max: float, dsmax: float = 1e-2, max_steps: int = 1000):
    p0 = Point(step=0)
    J = jax.jit(jax.jacobian(prob.f))
    z = jnp.append(prob.x0, prob.p0)

    p0.z, success = NaturalCorrector(delta=0.1, k=1)(z, None, prob.f, None, dsmax)
    
    p0.evals, p0.evecs = jnp.linalg.eig(J(p0.z[:-1], p0.z[-1])) 
    p0.n_unstable, p0.n_imag, p0.stable = is_stable(p0.evals)

    zpred, v = prediction(p0.z, dsmax, prob.f)
    z, success = correction(zpred, p0.z, prob.f, v, dsmax)
    
    if not success:
        pass
    
    p1 = Point(z=z)
    p1.evals, p1.evecs = jnp.linalg.eig(J(p1.z[:-1], p1.z[-1])) 

    p1.n_unstable, p1.n_imag, p1.stable = is_stable(p1.evals)

    branch = Branch([p0, p1])
    _continuation(branch, prob, prediction, correction, p_min, p_max, dsmax, max_steps)

    return Branches.branches

def _continuation(branch: Branch, prob: BifurcationProblem, prediction: Predictor, correction: Corrector, p_min: float, p_max: float, dsmax: float = 1e-2, max_steps: int = 1000):
    p_initial, p1 = branch.points[-2:]
    J = jax.jit(jax.jacobian(prob.f))

    dz = p1.z - p_initial.z
    dz /= jnp.linalg.norm(dz, ord=2)

    predictor = TangentPredictor(dz0 = dz, k=1)
    corrector = PALC(delta=0.1)
    
    p0 = p1
    for i in range(max_steps):
        zpred, v = predictor(p0.z, dsmax, prob.f)
        z, success = corrector(zpred, p0.z, prob.f, v, dsmax)
        
        if p_min>zpred[-1] or p_max<zpred[-1]:
            break
        
        if not success:
            pass
        
        p1 = Point(step=i, z=z)

        p1.evals, p1.evecs = jnp.linalg.eig(J(p1.z[:-1], p1.z[-1])) 
        p1.n_unstable, p1.n_imag, p1.stable = is_stable(p1.evals)
        known, p1.tp = get_bifurcation_type(p0, p1)

        if p1.tp == 'bp':
            v_b = normal_orthogonal_direction_method(prob.f, p0.z, p1.z)
            z1_b = p1.z + dsmax*v_b
            corrector_b = PALC(delta=0.1)
            z2_b, success_b = corrector_b(z1_b, p1.z, prob.f, v_b, dsmax)
            if success_b:
                p0_b = p1
                
                p1_b = Point(z=z2_b)
                p1_b.evals, p1_b.evecs = jnp.linalg.eig(J(p1_b.z[:-1], p1_b.z[-1])) 
                p1_b.n_unstable, p1_b.n_imag, p1_b.stable = is_stable(p1_b.evals)

                branch_new = Branch([p0_b, p1_b])
                _continuation(branch_new, prob, prediction, correction, p_min, p_max, dsmax, max_steps)
        
        branch.add(p1)
        p0 = p1
        
    Branches.branches.append(branch)