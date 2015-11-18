from PyQt4 import QtGui, QtCore
from matplotlib import colors
from matplotlib import cm as cmx

import sys
import math
import random

class ColorManager():
    def __init__(self, length, limit, mode="rand", cmap='Accent'):
        """
        :param length: the length of the video (frames)
        :param limit: the max number of colors to be used
        :param mode: "cmap" or "rand", chooses the colors randomly or from a cmap
        :param cmap: the cmap to be used in "cmap" mode
        """

        # TODO: http://llllll.li/randomColor/ has a distinguishable color generator on his todo list, check it later
        # list to store all the tracks
        self.tracks = []

        #lenght of the video
        self.length = length

        # max count of colors
        self.limit = limit

        if mode == "cmap":
            self.mode = "cmap"
            color_norm = colors.Normalize(vmin=0, vmax=limit)
            self.scalar_map = cmx.ScalarMappable(norm=color_norm, cmap=cmap)
        else:
            self.mode = "rand"

        random.seed()
        self.id = 0

    def get_next_id(self):
        # return current id and raise it by one
        self.id += 1
        return self.id - 1

    def new_track(self, start, stop):
        """
        Adds a new track to the color manager
        :param start: the first frame of the track
        :param stop: the last frame
        :return: tuple: (color, id)
        """

        # create a new track
        track = Track(start, stop, self.get_next_id(), QtGui.QColor().fromRgb(0, 0, 0))

        # find a suitable color for it
        if self.mode == "cmap":
            color = self.find_color_cmap(track)
        else:
            color = self.find_color(track)
        track.set_color(color)

        # add it in the tracks list
        self.tracks.append(track)

        return color, self.id

    def delete_(self, id):
        """
        Delete the track from CM. To merge two tracks, use ColorManager.merge().
        :param id:
        :return: None
        """
        self.tracks.remove(self.find_by_id(id))

    def merge(self, id1, id2):
        """
        Merges two tracks into the first one. All free space between the tracks is added to the first track, too. The
        new track will have the color and id of the first track.
        :param id1: id of the first track
        :param id2: id of the second one
        :return: None
        """

        t1 = self.find_by_id(id1)
        t2 = self.find_by_id(id2)

        if t1.start > t2.start:
            t1.start = t2.start
        if t1.stop < t2.stop:
            t1.stop = t2.stop

        self.tracks.remove(t2)

    def find_by_id(self, id):
        for track in self.tracks:
            if track.id == id:
                return track

    def collide(self, track1, track2):
        # returns the length of the intersection of track1 and track2 (how long they "exist" together)
        if (track1.start <= track2.start and track1.stop <= track2.start)\
                or (track2.start <= track1.start and track2.stop <= track1.start):
            return 0
        if track1.start <= track2.start and track1.stop <= track2.stop:
            return abs(track1.stop - track2.start)
        if track2.start <= track1.start and track2.stop <= track1.stop:
            return abs(track2.stop - track1.start)
        if track1.start <= track2.start and track1.stop >= track2.stop:
            return abs(track2.len)
        if track2.start <= track1.start and track2.stop >= track1.stop:
            return abs(track1.len)
        print "Ooops! [%s - %s] and [%s - %s]" % (track1.start, track1.stop, track2.start, track2.stop)

    def find_color(self, track):
        i = 0
        # 0.7 seems to be the ideal value (distinguishable, yet not too hard to achieve)
        limit = 0.7
        while True:
            ok = True
            # try to pick a color
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            # do not use colors, that are too close to black
            if (r + g + b) < 70:
                continue
            else:
                c1 = QtGui.QColor().fromRgb(r, g, b)
                for t in self.tracks:
                    # it must be different from any other track color
                    c2 = t.get_color()
                    distance = self.get_yuv_distance(c1, c2)
                    # if two of the colors are all the same, move on
                    if distance == 0:
                        ok = False
                        break
                    # the more frames the tracks share, the more different (distant) they must be
                    value = self.collide(t, track) / self.get_yuv_distance(c1, c2)
                    if value > limit:
                        ok = False
                if ok:
                    print i
                    return QtGui.QColor().fromRgb(r, g, b)

                if i > 500:
                    # if no color was found in 500 laps, return the current color
                    print "No color found"
                    # return QtGui.QColor().fromRgb(0, 0, 0)
                    return QtGui.QColor().fromRgb(r, g, b)
                i += 1
                # try to make the choosing easier by enlarging the limit each time a wrong color is picked
                limit += 0.02

    def find_color_cmap(self, track):
        while(True):
            ok = True
            # try to pick a random color position
            i = random.randint(0, self.limit)
            for t in self.tracks:
                # if a colliding track already uses it, try again
                if self.collide(track, t) > 0 and i == t.color_id:
                    i = (i + 1) % self.limit
                    ok = False
                    break
            if ok:
                tmp_color = self.scalar_map.to_rgba(i)
                track.set_color_id(i)
                return QtGui.QColor().fromRgbF(tmp_color[0], tmp_color[1], tmp_color[2])

    def get_yuv_distance(self, c1, c2):
        # returns the "distance" of two colors in the YUV color system, which corresponds best to human perception
        # formulas to convert RGB to YUV (from wikipedia)
        y1 =  0.299 * c1.red() + 0.587 * c1.green() + 0.114 * c1.blue()
        u1 = -0.147 * c1.red() - 0.289 * c1.green() + 0.436 * c1.blue()
        v1 =  0.615 * c1.red() - 0.515 * c1.green() - 0.100 * c1.blue()
        y2 =  0.299 * c2.red() + 0.587 * c2.green() + 0.114 * c2.blue()
        u2 = -0.147 * c2.red() - 0.289 * c2.green() + 0.436 * c2.blue()
        v2 =  0.615 * c2.red() - 0.515 * c2.green() - 0.100 * c2.blue()
        # distance of 2 points in 3D area (YUV cube)
        # the U and V distances are enlarged, because they are harder to see for humans
        distance = math.sqrt((y1-y2)**2 + 3*(u1-u2)**2 + 3*(v1-v2)**2)
        return distance

class TempGui(QtGui.QWidget):
    def __init__(self):
        super(TempGui, self).__init__()

        self.const = 30
        self.cm = ColorManager(1300, self.const)

        self.setLayout(QtGui.QVBoxLayout())
        widget = QtGui.QWidget()
        widget.setLayout(QtGui.QHBoxLayout())
        self.layout().setAlignment(QtCore.Qt.Alignment(QtCore.Qt.AlignBottom))
        self.text_field = QtGui.QLineEdit()
        self.button = QtGui.QPushButton("Confirm")
        self.button.clicked.connect(self.command)
        widget.layout().addWidget(self.text_field)
        widget.layout().addWidget(self.button)
        self.layout().addWidget(widget)

        self.setAutoFillBackground(True)

        p = self.palette()
        p.setColor(self.backgroundRole(), QtCore.Qt.black)
        self.setPalette(p)

        tracks = []
        for i in range (0, self.const):
            start = random.randint(0, 1300)
            stop = random.randint(0, 1300)
            while stop >= start:
                start = random.randint(0, 1300)
                stop = random.randint(0, 1300)
            color = QtGui.QColor(0, 0, 0)
            track = Track(start, stop, 0, color)
            tracks.append(track)

        tracks.sort(key=lambda x: x.len)

        for track in tracks:
            self.cm.new_track(track.start, track.stop)

        #for i in range(0, self.const):
        #   self.tracks[i].set_color(self.colors[i])

    def command(self):
        command = self.text_field.text().split(" ")
        self.cm.merge(int(command[0]), int(command[1]))
        self.update()

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        pen = QtGui.QPen()
        pen.setWidth(10)
        for i in range(0, len(self.cm.tracks)):
            track = self.cm.tracks[i]
            pen.setColor(track.get_color())
            qp.setPen(pen)
            qp.drawLine(track.start, 12*i + 50, track.stop, 12*i + 50)
        qp.end()

class Track():
    def __init__(self, start, stop, id, color):
        self.color = color
        if start > stop:
            self.stop = start
            self.start = stop
        else:
            self.start = start
            self.stop = stop
        self.id = id
        self.len = stop - start
    def get_color(self):
        return self.color
    def set_color(self, color):
        self.color = color
    def get_len(self):
        return self.stop - self.start
    def set_color_id(self, color_id):
        self.color_id = color_id

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)

    ex = TempGui()
    ex.show()
    ex.move(-500, -500)
    ex.showMaximized()
    ex.setFocus()

    app.exec_()
    app.deleteLater()
    sys.exit()