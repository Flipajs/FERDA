import os, sys
from PyQt4 import QtGui, QtCore


def video_exists(paths):
    if isinstance(paths, list):
        for path in paths:
            if os.path.isfile(paths[0]):
                return True
        return False
    return os.path.isfile(paths)

def check_video_path(paths, parent):
    path_changed = False
    if not video_exists(paths):
        reply = QtGui.QMessageBox.question(parent, "Video file not found", "A video file for this project wasn't found. Please select the new position of the video in your filesystem.", "Cancel", "Choose video")
        print "Reply: %s" % reply
        if reply == 1:
            paths = []
            paths.append(str(QtGui.QFileDialog.getOpenFileName(parent, 'Select new video location', '.')))
            path_changed = True
        else:
            # TODO: Perhaps hide the error?
            parent.window().close()

    return paths, path_changed


