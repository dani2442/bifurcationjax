import matplotlib.pyplot as plt
import jax.numpy as jnp
import matplotlib as mpl

from bifurcationjax.utils.Branch import Diagram

def plot_bifurcation_diagram(diagram: Diagram, axis: int = 0, plot_fn = lambda p: p.z[0], path_save: str | None = None):
    dict_color = {'bp':0, 'hopf':1, 'nd':2}
    fig, ax = plt.subplots(figsize=(8,6))

    cmap = mpl.colormaps['Blues']

    b_len = len(diagram.branches)
    for i, branch in enumerate(diagram.branches):
        xs = [plot_fn(p) for p in branch.points]
        ps = [p.z[-1] for p in branch.points]
        ax.plot(ps, xs, color=cmap((b_len - i)/b_len), label=f"Branch $n={i+1}$")
        #ax.scatter(ps, xs)

    for bp in diagram.bps.keys():
        ax.scatter(bp.z[-1], plot_fn(bp), label=bp.tp, color='red', zorder=20)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    plt.legend(by_label.values(), by_label.keys())
    plt.grid()
    
    if path_save:
        plt.savefig(path_save)
    plt.show()