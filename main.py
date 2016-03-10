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
    name = 'Cam1_orig'
    wd = '/Users/flipajs/Documents/wd/GT/'
    # wd = '/Users/flipajs/Documents/wd/'
    snapshot = {'chm': wd+name+'/.auto_save/'+str(sn_id)+'__chunk_amanager.pkl',
                'gm': wd+name+'/.auto_save/'+str(sn_id)+'__graph_manager.pkl'}

    project.load(wd+name+'/cam1.fproj')

    with open(project.working_directory+'/temp/animal_id_mapping.pkl', 'rb') as f_:
        animal_id_mapping = pickle.load(f_)

    for ch_id in project.gm.chunk_list():
        animal_id = -1
        if ch_id in animal_id_mapping:
            animal_id = animal_id_mapping[ch_id]

        project.chm[ch_id].animal_id_ = animal_id


    ex.widget_control('load_project', project)


app.exec_()
app.deleteLater()
sys.exit()
