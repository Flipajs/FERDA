__author__ = 'filip@naiser.cz'

from utils import video_manager
import sys
import numpy as np
import utils
from viewer.background_corrector import background_corrector_core
from PyQt4 import QtGui


def get_bg(video_file=None, step=100, max_frame=1000):
    if not video_file:
        print "No video_file"
        return None

    bg = None
    i = 0

    vid = video_manager.VideoManager(video_file)

    # we can try to use frame seek
    try:
        while True:
            img = vid.seek_frame(i)

            if img is None:
                break

            if i > max_frame:
                break

            if bg is not None:
                print 'processing... ' + str(int(i / (max_frame / 100.))) + ' / 100'
                bg = np.maximum(bg, img)
            else:
                bg = img

            i += step

    except Exception as e:
        utils.misc.print_exception(e)

        i = -1
        while True:
            i += 1
            img = video_manager.move2_next()

            if img is None:
                break

            if i > max_frame:
                break

            if i % step != 0:
                continue

            if bg is not None:
                print 'processing... ' + str(int(i / (max_frame / 100.))) + ' / 100'
                bg = np.maximum(bg, img)
            else:
                bg = img

    app = QtGui.QApplication(sys.argv)
    dialog = background_corrector_core.BackgroundCorrector(bg, None)
    app.exec_()


if __name__ == "__main__":
    # change here for your usage, after the computation of BG estimation, there will be dialog with bg correction
    # tool displayed.

    path = '/home/flipajs/Downloads/c_bigLense_colormarks3_corrected.avi'
    step = 50
    max_frame = 1000
    get_bg(path, step, max_frame)