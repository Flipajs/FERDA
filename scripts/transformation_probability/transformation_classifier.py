from PyQt4 import QtGui
import cPickle as pickle
import numpy as np
import sys

from os.path import exists

from core.project.project import Project
import ground_truth_widget

PRIME = 2 ** 8 + 1
FNAME = "region_probability_results.p"
results = {}

if exists(FNAME):
    results = pickle.load(open(FNAME, 'rb'))

widget = None


def get_results():
    global results
    results.update(widget.get_results())
    pickle.dump(results, open(FNAME, 'wb'))


def improve_ground_truth(widget, project):
    data = [(project.rm[1], project.rm[2]), (project.rm[2], project.rm[3])]
    regions = filter(lambda x: hash_region_tuple(x) not in results, data)
    # if region
    widget.set_data(regions)


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


def hash_region_tuple(region_tuple):
    return (PRIME + region_tuple[0].id()) * PRIME + region_tuple[1].id()

if __name__ == "__main__":
    project = Project()
    project.load("/home/simon/FERDA/projects/Cam1_/cam1.fproj")

    app = QtGui.QApplication(sys.argv)
    widget = ground_truth_widget.GroundTruthWidget(project)
    widget.show()

    improve_ground_truth(widget, project)

    app.aboutToQuit.connect(get_results)
    app.exec_()

    get_results()


