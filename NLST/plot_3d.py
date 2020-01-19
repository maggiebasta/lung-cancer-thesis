import matplotlib as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plotly.tools import FigureFactory as FF
from plotly.offline import init_notebook_mode, iplot
from plotly.graph_objs import *
from skimage import measure

init_notebook_mode(connected=True) 

"""
Plots 3D volumes
source code:
https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/
"""


def make_mesh(image, threshold=.5, step_size=1):
    p = image.transpose(2, 1, 0)

    verts, faces, norm, val = measure.marching_cubes_lewiner(
        p, threshold, step_size=step_size, allow_degenerate=True
    )
    return verts, faces


def plotly_3d(verts, faces):
    x, y, z = zip(*verts)

    # Make the colormap single color since axes are positional not intensity.
    colormap = ['rgb(236, 236, 212)', 'rgb(236, 236, 212)']
    fig = FF.create_trisurf(
        x=x,
        y=y,
        z=z,
        plot_edges=False,
        colormap=colormap,
        simplices=faces,
        backgroundcolor='rgb(64, 64, 64)',
        title="Interactive Visualization"
    )
    iplot(fig)


def plt_3d(verts, faces):
    x, y, z = zip(*verts)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=1)
    face_color = [1, 1, 0.9]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, max(x))
    ax.set_ylim(0, max(y))
    ax.set_zlim(0, max(z))
    ax.set_facecolor((0.7, 0.7, 0.7))
    plt.show()
