__author__ = 'flipajs'


# for some reason on MAC machines it is necessary to import it, else there is problem with loading MSER DLLs
import networkx as nx

import sys
from PyQt4 import QtGui
from scripts.region_graph3 import NodeGraphVisualizer, visualize_nodes
from core.settings import Settings as S_
from utils.video_manager import get_auto_video_manager, optimize_frame_access
import numpy as np
from skimage.transform import rescale
from core.project import Project
from utils.misc import is_flipajs_pc


def call_visualizer(t_start, t_end, project):
    solver = project.saved_progress['solver']
    if t_start == t_end == -1:
            sub_g = solver.g
    else:
        nodes = []
        for n in solver.g.nodes():
            if t_start <= n.frame_ < t_end:
                nodes.append(n)

        sub_g = solver.g.subgraph(nodes)

    vid = get_auto_video_manager(project.video_paths)
    regions = {}

    nodes = []
    for n in sub_g.nodes():
        is_ch, t_reversed, ch = solver.is_chunk(n)
        if is_ch:
            if ch.length() >= 0:
                nodes.append(n)

    optimized = optimize_frame_access(nodes)

    i = 0
    num_parts = 50
    part_ = len(optimized) / num_parts + 1
    for n, seq, _ in optimized:
        if n.frame_ in regions:
            regions[n.frame_].append(n)
        else:
            regions[n.frame_] = [n]

        if 'img' not in solver.g.node[n]:
            im = vid.get_frame(n.frame_, sequence_access=seq)

            if S_.mser.img_subsample_factor > 1.0:
                im = np.asarray(rescale(im, 1/S_.mser.img_subsample_factor) * 255, dtype=np.uint8)

            solver.g.node[n]['img'] = visualize_nodes(im, n)
            sub_g.node[n]['img'] = solver.g.node[n]['img']

        i += 1

        if i % part_ == 0:
            print "PROGRESS ", i, " / ", len(optimized)

    ngv = NodeGraphVisualizer(sub_g, regions)
    ngv.visualize()

    return ngv



if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)

    if is_flipajs_pc():
        project = Project()
        project.load('/Users/flipajs/Documents/wd/eight_test/test.fproj')
    else:
        # EDIT HERE....

        project = Project()
        project.load('/home/simon/Documents/res/c3_1h30/c3_1h30.fproj')

    ex = call_visualizer(-1, -1, project)
    # ex = call_visualizer(0, 50, project)
    ex.showMaximized()
    ex.setFocus()

    app.exec_()
    app.deleteLater()
    sys.exit()
