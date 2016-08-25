import sys
import cPickle as pickle
from PyQt4 import QtGui

from gui import main_window
from core.settings import Settings as S_
from utils.misc import is_flipajs_pc

app = QtGui.QApplication(sys.argv)
ex = main_window.MainWindow()
# ex.showMaximized()
ex.setFocus()

from core.project.project import Project
project = Project()

S_.general.print_log = False

# This is development speed up process (kind of fast start). Runs only on developers machines...
if is_flipajs_pc():
    sn_id = 2
    cam_ = 1
    name = 'Cam'+str(cam_)+' copy'
    wd = '/Users/flipajs/Documents/wd/gt/'
    # wd = '/Users/flipajs/Documents/wd/'
    snapshot = {'chm': wd+name+'/.auto_save/'+str(sn_id)+'__chunk_amanager.pkl',
                'gm': wd+name+'/.auto_save/'+str(sn_id)+'__graph_manager.pkl'}

    project.load(wd+name+'/cam'+str(cam_)+'.fproj')

    try:
        # WORKAROUND:
        for t in project.chm.chunk_list():
            t.N = set()
            t.P = set()

        # with open(project.working_directory+'/temp/chunk_available_ids.pkl', 'rb') as f_:
        #     data = pickle.load(f_)
        #
        #     #outdated
        #     Ps = data['ids_present_in_tracklet']
        #     Ns = data['ids_not_present_in_tracklet']
        #     probabilities = data['probabilities']
        #
        # for ch_id in project.gm.chunk_list():
        #     animal_id = -1
        #     if ch_id in Ps:
        #         probs = None
        #         if ch_id in probabilities:
        #             probs = probabilities[ch_id]
        #
        #         animal_id = {'P': Ps[ch_id], 'N': Ns[ch_id], 'probabilities': probs}
        #
        #     project.chm[ch_id].animal_id_ = animal_id
    except IOError:
        pass

    ex.widget_control('load_project', project)


app.exec_()
app.deleteLater()
sys.exit()
