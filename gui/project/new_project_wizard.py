from PyQt6 import QtGui, QtWidgets

from core.project.project import Project
from gui.init.create_project_wizard_page import CreateProjectPage
from gui.init.crop_video_page import CropVideoPage
from gui.init.circle_arena_editor_widget import CircleArenaEditorWidget
from gui.init.setup_msers_wizard_page import SetupMSERsWizardPage
from gui.region_classifier_tool import RegionClassifierTool


class NewProjectWizard(QtWidgets.QWizard):
    def __init__(self,  *args, **kwargs):
        super(NewProjectWizard, self).__init__(*args, **kwargs)

        self.setWindowTitle('New Project Wizard')
        self.create_project_page = CreateProjectPage()
        self.addPage(self.create_project_page)

        self.crop_video_page = CropVideoPage()
        self.addPage(self.crop_video_page)

        self.arena_editor_page = CircleArenaEditorWidget()
        self.addPage(self.arena_editor_page)

        self.setup_msers_page = SetupMSERsWizardPage()
        self.addPage(self.setup_msers_page)

        self.cardinality_classification_page = RegionClassifierTool()
        self.addPage(self.cardinality_classification_page)

        self.project = Project()
        # self.project_ready = QtCore.pyqtSignal('Project', name='project_ready')
        # self.accepted.connect(self.on_accepted)

    # def on_accepted(self):
    #     self.project_ready.emit(self.project)


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    wizard = NewProjectWizard()
    wizard.show()
    sys.exit(app.exec_())
