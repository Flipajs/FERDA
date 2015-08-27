from PyQt4 import QtGui, QtCore

import sys
import math
import random

class ColorManager():
    def __init__(self, lenght, limit):
        self.tracks = []
        #lenght of the video
        self.lenght = lenght
        # max count of colors
        self.limit = limit

        random.seed()
        self.id = 0

    def get_cmap_colors(self, count):
        colors = []
        for i in range (0, count):
            c = self.scalar_map.to_rgba(i)
            result = QtGui.QColor().fromRgbF(c[0], c[1], c[2])
            colors.append(result)
        return colors

    def get_next_id(self):
        # return current id and raise it by one
        self.id += 1
        return self.id - 1

    def new_track(self, start, stop):
        # create a new track
        track = Track(start, stop, self.get_next_id(), QtGui.QColor().fromRgb(0, 0, 0))
        # add it in the tracks list
        self.tracks.append(track)
        # find a suitable color for it
        color = self.find_color(track)
        track.set_color(color)

    def delete(self, id):
        self.tracks.remove(self.find_by_id(id))

    def merge(self, id1, id2):
        # merges two tracks into the first one. All free space between the tracks is added to the first track, too.
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
        limit = 0.7
        while True:
            ok = True
            # try to pick a color
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            c1 = QtGui.QColor().fromRgb(r, g, b)
            for t in self.tracks:
                # it must be different from any other track color
                c2 = t.get_color()
                # the more frames the tracks share, the more different (distant) they must be
                value = self.collide(t, track) / self.get_yuv_distance(c1, c2)
                if value > limit:
                    ok = False
            if ok:
                print i
                return QtGui.QColor().fromRgb(r, g, b)

            if i > 500:
                # return black if no color was found in 500 laps
                print "No color found"
                return QtGui.QColor().fromRgb(0, 0, 0)
            i += 1
            # try to make the choosing easier by enlarging the limit each time a wrong color is picked
            limit += 0.02

    def get_yuv_distance(self, c1, c2):
        # returns the "distance" of two colors in the YUV color system, which corresponds best with human perception
        y1 =  0.299 * c1.red() + 0.587 * c1.green() + 0.114 * c1.blue()
        u1 = -0.147 * c1.red() - 0.289 * c1.green() + 0.436 * c1.blue()
        v1 =  0.615 * c1.red() - 0.515 * c1.green() - 0.100 * c1.blue()
        y2 =  0.299 * c2.red() + 0.587 * c2.green() + 0.114 * c2.blue()
        u2 = -0.147 * c2.red() - 0.289 * c2.green() + 0.436 * c2.blue()
        v2 =  0.615 * c2.red() - 0.515 * c2.green() - 0.100 * c2.blue()
        # distance of 2 points in 3D area (YUV cube)
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

        # tracks.sort(key=lambda x: x.len)

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