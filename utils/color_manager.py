from PyQt4 import QtGui, QtCore
import numpy as np

import matplotlib.cm as cmx
import matplotlib.colors as colors
import colorsys
import sys
import math
import random


class ColorManager():
    def __init__(self, lenght, limit, cmap='Accent'):
        self.tracks = []
        self.lenght = lenght
        self.limit = limit
        random.seed()
        self.U_OFF = .436
        self.V_OFF = .615
        color_norm = colors.Normalize(vmin=0, vmax=self.limit-1)
        self.scalar_map = cmx.ScalarMappable(norm=color_norm, cmap=cmap)
        self.index = 0
        self.step = 500/limit
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
        track = Track(start, stop, self.get_next_id(), QtGui.QColor().fromRgb(0, 0, 0))
        """
        c = self.scalar_map.to_rgba(self.index)
        self.index = (self.index + self.step) % 55
        color = QtGui.QColor().fromRgbF(c[0], c[1], c[2])
        track.set_color(color)
        self.tracks.append(track)
        return color
        """
        color = self.find_color(track)
        self.tracks.append(track)
        track.set_color(color)
        return color

    def delete(self, id):
        self.tracks.remove(self.find_by_id(id))

    def merge(self, id1, id2):
        # merges two tracks into the first one. All free space between the tracks is added to the first track, too.
        t1 = self.find_by_id(id1)
        t2 = self.find_by_id(id2)

        if(t1.start > t2.start):
            t1.start = t2.start
        if(t1.stop < t2.stop):
            t1.stop = t2.stop

        self.tracks.remove(t2)

    def find_by_id(self, id):
        for track in self.tracks:
            if track.id == id:
                return track

    def collide(self, track1, track2):
        margin = 20
        if (track1.start < track2.start and track1.stop < track2.start)\
                or (track2.start < track1.start and track2.stop < track1.start):
            return False
        print "match, [%s-%s] and [%s-%s]" % (track1.start, track1.stop, track2.start, track2.stop)
        return True

    def find_color(self, track):
        i = 0
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
                # check if both colors exist at the same time
                if self.collide(t, track) and self.get_yuv_distance(c1, c2) < 400:
                    ok = False
            if ok:
                return QtGui.QColor().fromRgb(r, g, b)
            if i > 15:
                print "No color found"
                return QtGui.QColor().fromRgb(0, 0, 0)
            i += 1

    def get_yuv_distance(self, c1, c2):
        y1 =  0.299 * c1.red() + 0.587 * c1.green() + 0.114 * c1.blue()
        u1 = -0.147 * c1.red() - 0.289 * c1.green() + 0.436 * c1.blue()
        v1 =  0.615 * c1.red() - 0.515 * c1.green() - 0.100 * c1.blue()
        y2 =  0.299 * c2.red() + 0.587 * c2.green() + 0.114 * c2.blue()
        u2 = -0.147 * c2.red() - 0.289 * c2.green() + 0.436 * c2.blue()
        v2 =  0.615 * c2.red() - 0.515 * c2.green() - 0.100 * c2.blue()

        distance = math.sqrt(40*(y1-y2)**2 + (u1-u2)**2 + (v1-v2)**2)
        print distance
        return distance


    def compare_priorities(self, track):
        greater = 0
        smaller = 0
        for t in self.tracks:
            if t.len > track.len:
                greater += 1
            else:
                smaller += 1
        return greater, smaller

    def delete(self, color, start, stop):
        return False

    def get_colors(self, num_colors):
        # code from http://stackoverflow.com/questions/470690/how-to-automatically-generate-n-distinct-colors
        colors = []
        rnd = np.random
        for i in np.arange(0., 360., 360. / num_colors):
            hue = i/360.
            lightness = (50 + rnd.rand() * 10)/100.
            saturation = (90 + rnd.rand() * 10)/100.
            c = QtGui.QColor().fromHslF(hue, saturation, lightness)
            colors.append(c)
        return colors

    def spiral_method(self, N):
        # code from http://web.archive.org/web/20120421191837/http://www.cgafaq.info/wiki/Evenly_distributed_points_on_sphere
        dlong = math.pi*(3-math.sqrt(5)) # /* ~2.39996323 */
        dz = 2.0/N
        lon = 0
        z = 1 - dz/2
        node = []
        for k in range(0, N-1):
            r = math.sqrt(1-z*z)
            point = Point3D(math.cos(lon)*r, math.sin(lon)*r, z)
            node.append(point.to_color())
            z = z - dz
            lon = lon + dlong
            print point.to_string()

        return node

    def yuv_method(self, N, min=0.8, max=0.3):
        # working,but slow method (takes more than 10s to choose 50 colors)
        # code from http://stackoverflow.com/questions/470690/how-to-automatically-generate-n-distinct-colors

        yuv_colors = []

        for i in range (0, N):
            yuv_colors.append(self.get_rnd_yuv(min, max))

        for c in range(0, N*100):
            # print c
            worst = 8888
            worstID = 0
            for i in range (1, len(yuv_colors)):
                for j in range (0, i):
                    dist = self.sqrdist(yuv_colors[i], yuv_colors[j])
                    if dist < worst:
                        worst = dist
                        worstID = i
            best = self.compare_rand_yuv(worst, min, max, yuv_colors)
            if best is None:
                break
            else:
                yuv_colors[worstID] = best

        rgb_colors = []
        for i in range(0, len(yuv_colors)):
            rgb = self.yuv2rgb(yuv_colors[i].x, yuv_colors[i].y, yuv_colors[i].z)
            rgb_colors.append(QtGui.QColor().fromRgbF(rgb.x, rgb.y, rgb.z))
        return rgb_colors

    def compare_rand_yuv(self, bestDistSqrd, min, max, candidates):
        # print "compare_rand_yuv"
        for attempt in range(1, 100*len(candidates)):
            candidate = self.get_rnd_yuv(min, max)
            good = True
            for i in range (1, len(candidates)):
                if (self.sqrdist(candidate,candidates[i]) < bestDistSqrd):
                    good = False
            if good:
                return candidate
        return None

    def yuv2rgb(self, y, u, v):
        r = 1 * y + 0 * u + 1.13983 * v
        g = 1 * y + -.39465 * u + -.58060 * v
        b = 1 * y + 2.03211 * u + 0 * v
        return Point3D(r, g, b)

    def sqrdist(self, point_a, point_b):
        sum = 0;
        diff = point_a.x - point_b.x
        sum += diff * diff
        diff = point_a.y - point_b.y
        sum += diff * diff
        diff = point_a.z - point_b.z
        sum += diff * diff
        return sum

    def get_rnd_yuv(self, min, max):
        while True:
            y = random.random()
            u = random.random() * 2 * self.U_OFF - self.U_OFF
            v = random.random() * 2 * self.V_OFF - self.V_OFF
            rgb = self.yuv2rgb(y, u, v)
            if(0 <= rgb.x and rgb.x <= 1 and
                0 <= rgb.y and rgb.y <= 1 and
                0 <= rgb.z and rgb.z <= 1 and
                (rgb.x > min or rgb.y > min or rgb.z > min) and
                (rgb.x < max or rgb.y < max or rgb.z < max)):
                return Point3D(y, u, v)

    # TODO: formula to count color distances: http://www.emanueleferonato.com/2009/09/08/color-difference-algorithm-part-2/
    # TODO: Golden ratio at http://devmag.org.za/2012/07/29/how-to-choose-colours-procedurally-algorithms/

class Point3D():
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def to_string(self):
        return "(%s, %s, %s)" %(self.x, self.y, self.z)

    def to_color(self):
        r = 2.5623 * self.x + (-1.1661) * self.y + (-0.3962) * self.z
        g = (-1.0215) * self.x + 1.9778 * self.y + 0.0437 * self.z
        b = 0.0752 * self.x + (-0.2562) * self.y + 1.1810 * self.z
        c = QtGui.QColor().fromRgbF(r, g, b)
        return c

class TempGui(QtGui.QWidget):
    def __init__(self):
        super(TempGui, self).__init__()

        self.const = 50
        self.cm = ColorManager(1300, self.const)
        # self.colors = cm.get_colors(self.const)
        # self.colors = cm.spiral_method(self.const)
        # self.colors = cm.yuv_method(self.const)
        # print self.colors
        # self.colors = cm.get_cmap_colors(self.const)
        # self.tracks = []
        for i in range (0, self.const):
            start = random.randint(0, 1300)
            stop = random.randint(0, 1300)
            while stop >= start:
                start = random.randint(0, 1300)
                stop = random.randint(0, 1300)
            self.cm.new_track(start, stop)
            #color = cm.new_track(start, stop)
            #track = Track(start, stop, 0, color)
            #self.tracks.append(track)

        #self.tracks.sort(key=lambda x: x.len)

        #for i in range(0, self.const):
        #   self.tracks[i].set_color(self.colors[i])

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