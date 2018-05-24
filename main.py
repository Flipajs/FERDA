import argparse
import sys
import time
from PyQt4 import QtGui

from core.project.project import Project
from gui import main_window
from core.config import config

parser = argparse.ArgumentParser(description='FERDA laboratory animal tracking system.')
parser.add_argument('project', nargs='?', help='project directory or file')
args = parser.parse_args()

app = QtGui.QApplication(sys.argv)
ex = main_window.MainWindow()
ex.setFocus()

t_ = time.time()
config['general']['print_log'] = False
if args.project is not None:
    project = Project()
    project.load(args.project)
    # for t in project.chm.chunk_gen():
    #     print "Cardinality: {}, t_id: {}".format(t.get_cardinality(project.gm), t.id())
    # from tqdm import tqdm
    # thetas = []
    # for t in tqdm(project.chm.chunk_gen(), total=len(project.chm)):
    #     thetas.extend([r.theta_ for r in t.r_gen(project.gm, project.rm)])
    #
    # import matplotlib.pyplot as plt
    #
    # plt.hist(thetas)
    # plt.show()
    # TODO !! add it to assembly process
    # project.solver.one2one(check_tclass=True)
    # cases = []
    # for v in project.gm.g.vertices():
    #     if v.in_degree() == 2:
    #         t = project.gm.get_chunk(v)
    #
    #         if t.is_multi() and t.end_vertex(project.gm).out_degree() == 2:
    #             new_one = True
    #             for u in v.in_neighbors():
    #                 if not project.gm.get_chunk(u).is_single():
    #                     new_one = False
    #                     break
    #
    #             if new_one:
    #                 cases.append(t)
    #
    # print "#CASES: {}".format(len(cases))
    #from core.region.fitting import Fitting
    #from tqdm import tqdm
    #from scipy.ndimage.morphology import binary_erosion, binary_dilation
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
    #     for u in v1.in_neighbors():
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
    #
    # from scripts.regions_stats import decide_one2one
    # decide_one2one(project)
    #
    # project.chm.add_single_vertices_chunks(project)
    # project.save()
    #
    # project.load_semistate(wd, 'edge_cost_updated', update_t_nodes=True)
    # project.load_semistate(wd, 'first_tracklets')
    # project.load_semistate(wd, 'lp_id_SEG_IDCR_0')
    # project.load_semistate(wd, 'lp_HIL_INIT3_0')
    #
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
    ex.showMaximized()

print "FERDA is READY, loaded in {:.3}s".format(time.time()-t_)

app.exec_()
app.deleteLater()
sys.exit()
