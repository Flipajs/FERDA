from PyQt4 import QtGui

import sys

from gui.segmentation import segmentation
from gui.segmentation.segmentation import SegmentationPicker
from utils.video_manager import get_auto_video_manager


class GTWidget:

    def __init__(self, project, cluster_chunks_id):
        # self.video_manager = get_auto_video_manager(project)
        # for ch_id in cluster_chunks_id:
        #     chunk = project.chm[ch_id]
        #     for region in chunk:
        #         print region
        # app = QtGui.QApplication(sys.argv)
        # ex = SegmentationPicker('/home/dita/img_67.png')
        # ex.show()
        # ex.move(-500, -500)
        # ex.showMaximized()
        # ex.setFocus()
        #
        # app.exec_()
        # app.deleteLater()
        # sys.exit()



