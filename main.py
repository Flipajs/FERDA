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

    wd = '/Users/flipajs/Documents/wd/FERDA/C210min'
    wd = '/Users/flipajs/Documents/wd/FERDA/Cam1_'
    wd = '/Users/flipajs/Documents/wd/FERDA/Cam1_playground'
    # wd = '/Users/flipajs/Documents/wd/zebrafish_playground'
    # wd = '/Users/flipajs/Documents/wd/FERDA/Camera3'
    # wd = '/Users/flipajs/Documents/wd/FERDA/Sowbug3'

    # wd = '/Users/flipajs/Documents/wd/'
    # snapshot = {'chm': wd+name+'/.auto_save/'+str(sn_id)+'__chunk_amanager.pkl',
    #             'gm': wd+name+'/.auto_save/'+str(sn_id)+'__graph_manager.pkl'}

    # project.load(wd+name+'/cam'+str(ca
    # m_)+'.fproj')
    project.load(wd)
    from core.graph.chunk_manager import ChunkManager

    project.chm = ChunkManager()
    # with open(wd+'/temp/isolation_score.pkl', 'rb') as f:
    with open(wd+'/temp/strongly_better_filter.pkl', 'rb') as f:
    # with open(wd+'/temp/isolation_score.pkl', 'rb') as f:
    # with open('/Users/flipajs/Documents/wd/FERDA/Cam1_playground/temp/isolation_score.pkl', 'rb') as f:
        up = pickle.Unpickler(f)
        project.gm.g = up.load()
        up.load()
        chm = up.load()
        project.chm = chm

    from core.region.region_manager import RegionManager
    project.rm = RegionManager(wd+'/temp', db_name='part0_rm.sqlite3')
    project.gm.rm = project.rm

    project.gm.update_nodes_in_t_refs()

    # # Z detection
    # for v in project.gm.active_v_gen():
    #     if project.gm.z_case_detection(v):
    #         print "z in frame: {}".format(project.gm.region(v).frame_)



    project.chm.add_single_vertices_chunks(project, frames=range(4500))
    from utils.gt.gt import GT
    gt = GT()
    gt.load(project.GT_file)
    # gt.check_none_occurence()
    match = gt.match_on_data(project, max_d=3)
    with open('/Users/flipajs/Desktop/temp/match.pkl', 'wb') as f:
        pickle.dump(match, f)


    # match = {}
    # for frame in range(gt.min_frame(), gt.max_frame()):
    #     match[frame] = [None for _ in range(len(project.animals))]
    #
    #     for



    # project.chm.reset_itree(project.gm)

    # project.load_snapshot(snapshot)

    try:
        # WORKAROUND:
        for t in project.chm.chunk_list():
            if not hasattr(t, 'N'):
                t.N = set()
                t.P = set()
    except AttributeError:
        pass

    from utils.color_manager import colorize_project
    colorize_project(project)

    ex.widget_control('load_project', project)



print "FERDA is READY, loaded in {:.3}s".format(time.time()-t_)
ex.move(-500, -500)
ex.showMaximized()
app.exec_()
app.deleteLater()
sys.exit()
