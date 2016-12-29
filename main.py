import sys
import cPickle as pickle
from PyQt4 import QtGui

from gui import main_window
from core.settings import Settings as S_
from utils.misc import is_flipajs_pc
import time

app = QtGui.QApplication(sys.argv)
ex = main_window.MainWindow()
# ex.showMaximized()
ex.setFocus()

t_ = time.time()

from core.project.project import Project
project = Project()

S_.general.print_log = False

# This is development speed up process (kind of fast start). Runs only on developers machines...
if is_flipajs_pc():
    sn_id = 875
    cam_ = 1

    wd = '/Users/flipajs/Documents/wd/FERDA/Cam1_playground'
    wd = '/Users/flipajs/Documents/wd/FERDA/zebrafish_playground'
    # wd = '/Users/flipajs/Documents/wd/FERDA/Camera3'
    # wd = '/Users/flipajs/Documents/wd/FERDA/Sowbug3'

    # wd = '/Users/flipajs/Documents/wd/'
    # snapshot = {'chm': wd+name+'/.auto_save/'+str(sn_id)+'__chunk_amanager.pkl',
    #             'gm': wd+name+'/.auto_save/'+str(sn_id)+'__graph_manager.pkl'}

    # project.load(wd+name+'/cam'+str(ca
    # m_)+'.fproj')
    # project.load_semistate(wd, 'eps_edge_filter')
    # project.load_semistate(wd, 'eps_without_noise')
    # project.load_semistate(wd, 'eps_edge_filter')

    project.load_semistate(wd, 'id_classified')


    # from core.graph.chunk_manager import ChunkManager
    #
    # project.chm = ChunkManager()
    # # with open(wd+'/temp/isolation_score.pkl', 'rb') as f:
    # # with open(wd+'/temp/strongly_better_filter.pkl', 'rb') as f:
    # # with open(wd+'/temp/isolation_score.pkl', 'rb') as f:
    # # with open(wd+'/temp/isolation_score.pkl', 'rb') as f:
    # # with open('/Users/flipajs/Documents/wd/FERDA/Cam1_playground/temp/isolation_score.pkl', 'rb') as f:
    # with open(wd+'/temp/isolation_score.pkl', 'rb') as f:
    # # with open(wd+'/temp/strongly_better_filter.pkl', 'rb') as f:
    # # with open('/Users/flipajs/Documents/wd/FERDA/Sowbug3/temp/isolation_score.pkl', 'rb') as f:
    #     up = pickle.Unpickler(f)
    #     project.gm.g = up.load()
    #     up.load()
    #     chm = up.load()
    #     project.chm = chm
    #
    # from core.region.region_manager import RegionManager
    # project.rm = RegionManager(wd+'/temp', db_name='part0_rm.sqlite3')
    # project.gm.rm = project.rm
    #
    # project.gm.update_nodes_in_t_refs()
    #
    # # # Z detection
    # # for v in project.gm.active_v_gen():
    # #     if project.gm.z_case_detection(v):
    # #         print "z in frame: {}".format(project.gm.region(v).frame_)
    #
    #
    # # test_vs = []
    # # for v in project.gm.active_v_gen():
    # #     if int(v) == 25419:
    # #         print "a"
    # #     es, sc = project.gm.get_2_best_out_edges_appearance_motion_mix(v)
    # #     if int(v) == 25419:
    # #         print es, sc
    # #
    # #     if es[1] is not None:
    # #         ratio = sc[0] / sc[1]
    # #         test_vs.append((v, ratio))
    # #
    # #
    # # test_vs = sorted(test_vs, key=lambda x: -x[1])
    # #
    # # for v, ratio in test_vs:
    # #     print project.gm.region(v).frame(), v, ratio
    #
    #
    # # edges = []
    # # for e in project.gm.g.edges():
    # #     if not project.gm.edge_is_chunk(e):
    # #         edges.append(e)
    # #
    # # edges = sorted(edges, key=lambda x: -project.gm.g.ep['score'][x]*project.gm.g.ep['movement_score'][x])
    # #
    # # i = 0
    # # for e in edges:
    # #     if i == 500:
    # #         break
    # #
    # #     r = project.gm.region(e.source())
    # #
    # #     print r.frame(), e.source(), e.target(), project.gm.g.ep['score'][e]*project.gm.g.ep['movement_score'][e], project.gm.g.ep['score'][e], project.gm.g.ep['movement_score'][e]
    # #
    # #     i += 1
    #
    #
    # project.chm.add_single_vertices_chunks(project, frames=range(5000))
    # project.gm.update_nodes_in_t_refs()
    # from utils.gt.gt import GT
    # gt = GT()
    # gt.load(project.GT_file)
    # # gt.check_none_occurence()
    # match = gt.match_on_data(project, max_d=3, frames=range(5000))
    # with open('/Users/flipajs/Desktop/temp/match.pkl', 'wb') as f:
    #     pickle.dump(match, f)


    # match = {}
    # for frame in range(gt.min_frame(), gt.max_frame()):
    #     match[frame] = [None for _ in range(len(project.animals))]
    #
    #     for



    # project.chm.reset_itree(project.gm)

    # project.load_snapshot(snapshot)

    try:
        # WORKAROUND:
        for t in project.chm.chunk_gen():
            if not hasattr(t, 'N'):
                t.N = set()
                t.P = set()
    except AttributeError:
        pass

    from utils.color_manager import colorize_project
    colorize_project(project)

    # from core.id_detection.feature_manager import FeatureManager
    # fm = FeatureManager(project.working_directory, db_name='fm_idtracker_i_d50.sqlite3')


    ex.widget_control('load_project', project)



print "FERDA is READY, loaded in {:.3}s".format(time.time()-t_)
ex.move(-500, -500)
ex.showMaximized()
app.exec_()
app.deleteLater()
sys.exit()
