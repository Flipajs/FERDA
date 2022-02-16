import time
import unittest
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtTest import QTest
from core.project.project import Project
from utils.video_manager import get_auto_video_manager
from gui.init.setup_msers_wizard_page import SetupMSERsWizardPage


class TestSetupMSERsWizardPage(unittest.TestCase):

    def setUp(self):
        project = Project()
        project.working_directory = 'out/test_project'
        project.video_path = 'test/Sowbug3_cut.mp4'
        project.name = 'untitled'
        project.description = ''
        project.date_created = time.time()
        project.date_last_modification = time.time()

        self.app = QtWidgets.QApplication([])
        self.ui = SetupMSERsWizardPage()
        self.ui.project = project
        self.ui.set_video(get_auto_video_manager(project))

    def tearDown(self):
        self.app.deleteLater()

    # def test_play(self):
    #     self.assertEqual(self.ui.video.frame_number(), 0)
    #     QTest.mouseClick(self.ui.playPause, QtCore.Qt.MouseButton.LeftButton)
    #     QTest.qWait(50)
    #     self.assertGreater(self.ui.video.frame_number(), 0)
    #
    # def test_mark_start(self):
    #     QTest.mouseClick(self.ui.mark_start, QtCore.Qt.MouseButton.LeftButton)

    # def test_mark_end(self):
    #     QTest.mouseClick(self.ui.playPause, QtCore.Qt.MouseButton.LeftButton)
    #     QTest.qWait(50)
    #     QTest.mouseClick(self.ui.playPause, QtCore.Qt.MouseButton.LeftButton)
    #     QTest.mouseClick(self.ui.mark_stop, QtCore.Qt.MouseButton.LeftButton)

    def test_show(self):
        self.ui.show()
        self.app.exec()
