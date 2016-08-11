from PyQt4 import QtGui

import numpy as np
import sys

from core.project.project import Project
from scripts.transformation_probability.grand_truth_widget import GrandTruthWidget

results = None
widget = None

def get_results():
    results = widget.get_results()
    print results

def descriptor(r1, r2):
    # centroid distance
    centr_dist = np.linalg.norm(r2.centroid() - r1.centroid())

    # margin difference
    margin_diff = r2.margin() - r1.margin()

    # intensity
    max_intensity = r2.max_intensity_ - r1.max_intensity_
    min_intensity = r2.min_intensity_ - r1.min_intensity_

    # area difference
    area_diff = r2.area() - r1.area()

    # axis difference
    axis_diff = r2.major_axis_ - r1.major_axis_

    return centr_dist, margin_diff, max_intensity, min_intensity, area_diff, axis_diff

if __name__ == "__main__":
    project = Project()
    project.load("/home/simon/FERDA/projects/Cam1_/cam1.fproj")

    print descriptor(project.rm[1], project.rm[2])

    app = QtGui.QApplication(sys.argv)

    widget = GrandTruthWidget(project)
    widget.set_data([(project.rm[1], project.rm[2]), (project.rm[2], project.rm[3])])
    widget.show()
    app.aboutToQuit.connect(get_results)
    sys.exit(app.exec_())


