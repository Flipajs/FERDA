from PyQt5 import QtCore, QtGui, QtWidgets
# regenerate the ui_tracking_widget code with `pyuic4 tracking_widget.ui -o ui_tracking_widget.py -x`
from gui.generated.ui_tracking_widget import Ui_tracking_widget
import core.segmentation
import core.graph_assembly
from core.id_detection.complete_set_matching import do_complete_set_matching

# TODO:
# - PRIORITY: close project before processing, some tasks fail on duplicate access to sqlite db
# - don't recompute parts that are already computed
# - do per frame progress bar updates


class TrackingWidget(QtWidgets.QWidget):
    def __init__(self, project):
        QtWidgets.QWidget.__init__(self)

        self.project = project

        # Set up the user interface from Designer.
        self.ui = Ui_tracking_widget()
        self.ui.setupUi(self)
        self.ui.start_tracking_button.clicked.connect(self.start_tracking)

        self.segmentation_thread = SegmentationThread(self.project.working_directory)
        # self.segmentation_thread.update.connect(self.progress_bar_update)
        self.segmentation_thread.finished.connect(self.segmentation_thread_finished)
        # if core.graph_assembly.is_assemply_completed(project):
        #     self.ui.pbar_segmentation.setValue(self.ui.pbar_segmentation.maximum())

        self.graph_thread = GraphConstructionThread(self.project)
        # self.segmentation_thread.update.connect(self.progress_bar_update)
        self.graph_thread.finished.connect(self.graph_thread_finished)
        # if core.graph_assembly.is_assemply_completed(project):
        #     self.ui.pbar_graph.setValue(self.ui.pbar_graph.maximum())

        self.regions_classification_thread = RegionsClassificationThread(self.project)
        # self.segmentation_thread.update.connect(self.progress_bar_update)
        self.regions_classification_thread.finished.connect(self.regions_classification_thread_finished)
        # if core.region.clustering.is_project_cardinality_classified(project):
        #     self.ui.pbar_regions_classification.setValue(self.ui.pbar_regions_classification.maximum())

        self.reid_thread = ReIdentificationThread(self.project)
        # self.segmentation_thread.update.connect(self.progress_bar_update)
        self.reid_thread.finished.connect(self.reid_thread_finished)

        # self.setDisabled(True)

    def start_tracking(self):
        self.segmentation_thread.start()

    def segmentation_thread_finished(self):
        self.ui.pbar_segmentation.setValue(self.ui.pbar_segmentation.maximum())
        self.graph_thread.start()

    def graph_thread_finished(self):
        self.ui.pbar_graph.setValue(self.ui.pbar_graph.maximum())
        self.regions_classification_thread.start()

    def regions_classification_thread_finished(self):
        self.ui.pbar_regions_classification.setValue(self.ui.pbar_regions_classification.maximum())
        self.reid_thread.start()

    def reid_thread_finished(self):
        self.ui.pbar_reid.setValue(self.ui.pbar_reid.maximum())


class SegmentationThread(QtCore.QThread):
    # update = QtCore.pyqtSignal(int)

    def __init__(self, project_dir):
        super(SegmentationThread, self).__init__()
        self.project_dir = project_dir

    def run(self):
        core.segmentation.segmentation(self.project_dir)

    def __del__(self):
        self.wait()


class GraphConstructionThread(QtCore.QThread):
    # update = QtCore.pyqtSignal(int)

    def __init__(self, project):
        super(GraphConstructionThread, self).__init__()
        self.project = project

    def run(self):
        graph_solver = core.graph.solver.Solver(self.project)
        core.graph_assembly.graph_assembly(self.project, graph_solver)
        self.project.save()

    def __del__(self):
        self.wait()


class RegionsClassificationThread(QtCore.QThread):
    # update = QtCore.pyqtSignal(int)

    def __init__(self, project):
        super(RegionsClassificationThread, self).__init__()
        self.project = project

    def run(self):
        self.project.region_cardinality_classifier.classify_project(self.project)
        self.project.save()

    def __del__(self):
        self.wait()


class ReIdentificationThread(QtCore.QThread):
    # update = QtCore.pyqtSignal(int)

    def __init__(self, project):
        super(ReIdentificationThread, self).__init__()
        self.project = project

    def run(self):
        do_complete_set_matching(self.project)
        self.project.save()

    def __del__(self):
        self.wait()


