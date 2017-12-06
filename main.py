import sys
import cPickle as pickle
from PyQt4 import QtGui

from gui import main_window
from core.settings import Settings as S_
from utils.misc import is_flipajs_pc, is_matejs_pc
import time
from core.project.project import Project
import timeit
import numpy as np


app = QtGui.QApplication(sys.argv)
ex = main_window.MainWindow()
ex.setFocus()

t_ = time.time()

project = Project()

S_.general.print_log = False

# This is development speed up process (kind of fast start). Runs only on developers machines...
# if is_flipajs_pc() and False:
wd = None
if is_flipajs_pc():
    # wd = '/Users/flipajs/Documents/wd/FERDA/Cam1_rf'
    # wd = '/Users/flipajs/Documents/wd/FERDA/Cam1_playground'
    # wd = '/Users/flipajs/Documents/wd/FERDA/test6'
    # wd = '/Users/flipajs/Documents/wd/FERDA/zebrafish_playground'
    # wd = '/Users/flipajs/Documents/wd/FERDA/zebrafish_new'
    # wd = '/Users/flipajs/Documents/wd/FERDA/Camera3'
    # wd = '/Users/flipajs/Documents/wd/FERDA/Cam1_rfs2'
    wd = '/Users/flipajs/Documents/wd/FERDA/Cam1'
    # wd = '/Volumes/Seagate Expansion Drive/HH1_PRE_upper_thr_'
    # wd = '/Volumes/Seagate Expansion Drive/HH1_POST'
    # wd = '/Users/flipajs/Documents/wd/FERDA/Barbara_flu_bug/test6'
    # wd = '/Users/flipajs/Documents/wd/FERDA/rep1-cam2'
    # wd = '/Users/flipajs/Documents/wd/FERDA/rep1-cam3'

    # wd = '/Users/flipajs/Documents/wd/FERDA/Sowbug3'
    # wd = '/Users/flipajs/Documents/wd/FERDA/Sowbug3_new'

    # wd = '/Users/flipajs/Documents/wd/FERDA/test'

if is_matejs_pc():
    # wd = '/home/matej/prace/ferda/10-15/'
    # wd = '/home/matej/prace/ferda/10-15 (copy)/'
    pass

if wd is not None:
    project.load(wd)

    # TODO !! add it to assembly process
    # project.solver.one2one(check_tclass=True)

    # cases = []
    # for v in project.gm.g.vertices():
    #     if v.in_degree() == 2:
    #         t = project.gm.get_chunk(v)
    #
    #         if t.is_multi() and t.end_vertex(project.gm).out_degree() == 2:
    #             new_one = True
    #             for u in v.in_neighbours():
    #                 if not project.gm.get_chunk(u).is_single():
    #                     new_one = False
    #                     break
    #
    #             if new_one:
    #                 cases.append(t)
    #
    # print "#CASES: {}".format(len(cases))
    from core.region.fitting import Fitting
    from tqdm import tqdm
    from scipy.ndimage.morphology import binary_erosion, binary_dilation

    # erosion = True
    #
    # ii = 0
    # for t in cases:
    #     ii += 1
    #     if ii == 1:
    #         break
    #
    #     if t.length() > 5:
    #         continue
    #
    #     print "#tID: {}, start: {}, end: {}".format(t.id(), t.start_frame(project.gm), t.end_frame(project.gm))
    #
    #     v1 = t.start_vertex(project.gm)
    #     animals_r = []
    #     for u in v1.in_neighbours():
    #         reg = project.gm.region(u)
    #
    #         if erosion:
    #             pts = reg.pts()
    #             roi = reg.roi()
    #
    #             bim = np.zeros((roi.height(), roi.width()), dtype=np.bool)
    #             bim[pts[:, 0] - roi.y(), pts[:, 1] - roi.x()] = True
    #
    #             bim2 = binary_erosion(bim, iterations=3)
    #             bim2 = binary_dilation(bim2, iterations=2)
    #             new_pts = np.argwhere(bim2) + roi.top_left_corner()
    #             reg.pts_ = new_pts
    #             reg.roi_ = None
    #
    #         animals_r.append(reg)
    #
    #     for i in range(len(t)):
    #         # todo: invalidate original regions
    #
    #         # todo for each region in tracklet...
    #         reg = project.gm.region(t[i])
    #
    #         if erosion:
    #             pts = reg.pts()
    #             roi = reg.roi()
    #
    #             bim = np.zeros((roi.height(), roi.width()), dtype=np.bool)
    #             bim[pts[:, 0] - roi.y(), pts[:, 1] - roi.x()] = True
    #
    #             bim2 = binary_erosion(bim, iterations=3)
    #             bim2 = binary_dilation(bim2, iterations=2)
    #             new_pts = np.argwhere(bim2) + roi.top_left_corner()
    #             reg.pts_ = new_pts
    #             reg.roi_ = None
    #
    #         f = Fitting(reg, animals_r, num_of_iterations=10)
    #         results, stats = f.fit()
    #         print "\t", stats
    #
    #         new_animals = []
    #         for r in results:
    #             project.rm.add(r)
    #
    #             v = project.gm.add_vertex(r)
    #             new_t, _ = project.chm.new_chunk([int(v)], project.gm)
    #             new_t.color = QtGui.QColor.fromRgb(255, 0, 0)
    #
    #             new_animals.append(r)
    #
    #         animals = new_animals

        # from scripts.regions_stats import decide_one2one
        # decide_one2one(project)

        # project.chm.add_single_vertices_chunks(project)
        # project.save()

        # project.load_semistate(wd, 'edge_cost_updated', update_t_nodes=True)
        # project.load_semistate(wd, 'first_tracklets')
        # project.load_semistate(wd, 'lp_id_SEG_IDCR_0')
        # project.load_semistate(wd, 'lp_HIL_INIT3_0')

        # project.gm.update_nodes_in_t_refs()

    try:
        # old projects WORKAROUND:
        for t in project.chm.chunk_gen():
            if not hasattr(t, 'N'):
                t.N = set()
                t.P = set()
    except AttributeError:
        pass

    ex.widget_control('load_project', project)
    # colorize_project(project)
    # ex.move(-500, -500)
    ex.showMaximized()

print "FERDA is READY, loaded in {:.3}s".format(time.time()-t_)

app.exec_()
app.deleteLater()
sys.exit()
