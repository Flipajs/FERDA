__author__ = 'flipajs'


# for some reason on MAC machines it is necessary to import it, else there is problem with loading MSER DLLs

import sys

import numpy as np
from PyQt4 import QtGui
from skimage.transform import rescale

from core.project.project import Project
from scripts.region_graph3 import NodeGraphVisualizer, visualize_nodes
from utils.misc import is_flipajs_pc
from utils.video_manager import get_auto_video_manager


def call_visualizer(t_start, t_end, project, solver, min_chunk_len, update_callback=None, node_size=30, node_margin=0.1, show_in_visualizer_callback=None, reset_cache=True, show_vertically=False):

    if t_start == t_end == -1:
        sub_g = solver.g
    else:
        g_nodes = []
        for vertex in project.gm.get_all_relevant_vertices():
            r = project.gm.region(vertex)
            if t_start <= r.frame_ < t_end:
                g_nodes.append((r, vertex))

    vid = get_auto_video_manager(project)
    regions = {}

    nodes = []
    chunks = set()
    for n, vertex in g_nodes:
        ch, _ = project.gm.is_chunk(vertex)
        if ch:
            if ch.length() >= min_chunk_len:
                nodes.append(n)
                chunks.add(ch)

    optimized = optimize_frame_access_vertices(nodes)

    i = 0
    num_parts = 100
    part_ = len(optimized) / num_parts + 1

    # TODO: optimize for same frames...
    cache = {}

    for n, seq, _ in optimized:
        if n.frame_ in regions:
            regions[n.frame_].append(n)
        else:
            regions[n.frame_] = [n]

        if reset_cache or 'img' not in solver.g.node[n]:
            im = vid.get_frame(n.frame_)

            sf = project.other_parameters.img_subsample_factor
            if sf > 1.0:
                if n.frame_ not in cache:
                    im = np.asarray(rescale(im, 1/sf) * 255, dtype=np.uint8)
                    cache[n.frame_] = im
                else:
                    im = cache[n.frame_]

            # TODO: optimize... and add opacity parameter
            ch, _ = project.gm.is_chunk(n)
            c = tuple(ch.color) + (0.9, )

            solver.g.node[n]['img'] = visualize_nodes(im, n, margin=node_margin, color=c)

        i += 1

        if update_callback is not None and i % part_ == 0:
            update_callback(i / float(len(optimized)))

    ngv = NodeGraphVisualizer(solver, project.gm.g, regions, list(chunks), node_size=node_size, show_in_visualize_callback=show_in_visualizer_callback, show_vertically=show_vertically)
    ngv.visualize()

    return ngv


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)

    if is_flipajs_pc():
        project = Project()
        # project.load('/Users/flipajs/Documents/wd/eight/eight.fproj')
        project.load('/Users/flipajs/Documents/wd/GT/C210/C210.fproj')
        # project.load('/Users/flipajs/Documents/wd/colonies_crop1/colonies.fproj')
    else:
        # EDIT HERE....

        project = Project()
        project.load('/home/simon/Documents/res/c3_1h30/c3_1h30.fproj')

    # from utils.video_manager import get_auto_video_manager
    # vid = get_auto_video_manager(project)
    # img = vid.next_frame()
    #
    # from pylab import ogrid, gca, show
    # from mpl_toolkits.mplot3d import Axes3D
    # from matplotlib.cbook import get_sample_data
    # from matplotlib._png import read_png
    #
    # x, y = ogrid[0:img.shape[0], 0:img.shape[1]]
    # ax = gca(projection='3d')
    # ax.plot_surface(x, y, 0, rstride=5, cstride=5, facecolors=np.asarray(img, dtype=np.float)/255.)
    #
    # show()

    # ex = call_visualizer(-1, -1, project)
    ex = call_visualizer(0, 700, project, project.solver, 10)
    ex.show()
    ex.move(-500, -500)
    ex.showMaximized()
    ex.setFocus()

    app.exec_()
    app.deleteLater()
    sys.exit()
