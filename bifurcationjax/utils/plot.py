import matplotlib.pyplot as plt
import jax.numpy as jnp
import matplotlib as mpl

from bifurcationjax.utils.Branch import Diagram

def plot_bifurcation_diagram(
        diagram: Diagram, 
        axis: int = 0, 
        plot_fn = lambda p: p.z[0], 
        path_save: str | None = None, 
        plot_dots: bool = False, 
        ax: mpl.axes.Axes = None,
        title: str = ''
    ):
    
    dict_color = {'bp':0, 'hopf':1, 'nd':2}

    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6))

    cmap = mpl.colormaps['Blues']

    b_len = len(diagram.branches)
    for i, branch in enumerate(diagram.branches):
        xs = [plot_fn(p) for p in branch.points]
        ps = [p.z[-1] for p in branch.points]
        ax.plot(ps, xs, color=cmap((b_len - i)/b_len), label=f"Branch $n={i+1}$")
        
        if plot_dots:
            ax.scatter(ps, xs, s=10)

    for bp in diagram.bps.keys():
        ax.scatter(bp.z[-1], plot_fn(bp), label=bp.tp, color='red', zorder=20)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    ax.set_xlabel('$\mu$')
    ax.set_ylabel('$x$')

    ax.legend(by_label.values(), by_label.keys())
    ax.grid()
    ax.set_title(title)
    
    if path_save:
        plt.savefig(path_save, dpi=300)