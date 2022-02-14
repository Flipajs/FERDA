import argparse
import sys
import time
from PyQt6 import QtGui, QtWidgets

from core.project.project import Project
from core.config import config
from gui.main_tab_widget import MainTabWidget

parser = argparse.ArgumentParser(description='FERDA laboratory animal tracking system')
parser.add_argument('project', nargs='?', help='project directory or file')
args = parser.parse_args()

app = QtWidgets.QApplication(sys.argv)
main_window = MainTabWidget()
main_window.show()

t_ = time.time()
config['general']['print_log'] = False
if args.project is not None:
    project = Project()
    project.load(args.project, regions_optional=True, graph_optional=True, tracklets_optional=True)

    try:
        # old projects WORKAROUND:
        for t in project.chm.chunk_gen():
            if not hasattr(t, 'N'):
                t.N = set()
                t.P = set()
    except AttributeError:
        pass
    main_window.update_project(project)


print("FERDA is READY, loaded in {:.3}s".format(time.time()-t_))

sys.exit(app.exec_())
