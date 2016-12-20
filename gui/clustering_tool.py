from PyQt4 import QtGui, QtCore
import sys
from core.region.clustering import clustering, display_cluster_representants, most_distant
import cPickle as pickle


class ClusteringTool(QtGui.QWidget):
    def __init__(self, p):
        super(ClusteringTool, self).__init__()

        self.p = p


        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)

        self.b = QtGui.QPushButton('test')
        self.vbox.addWidget(self.b)

        self.update()
        self.show()

    def show_results(self, first_run=True):
        try:
            with open(p.working_directory + '/temp/clustering.pkl') as f:
                up = pickle.Unpickler(f)
                data = up.load()
                vertices = up.load()
                labels = up.load()
        except:
            if first_run:
                clustering(self.p)
                self.show_results(first_run=False)
            else:
                raise Exception("loading failed...")

        most_distant(self.p)

        for label in set(labels):
            print label
            display_cluster_representants(self.p, N=5)


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    from core.project.project import Project

    p = Project()
    p.load('/Users/flipajs/Documents/wd/FERDA/Cam1_playground')

    ex = ClusteringTool(p)
    ex.raise_()
    ex.activateWindow()

    ex.show_results()

    app.exec_()
    app.deleteLater()
    sys.exit()



