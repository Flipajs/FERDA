__author__ = 'simon'


import cv2
import random
import matplotlib.colors as colors
from utils.drawing.points import draw_points_crop
from skimage.transform import resize
from gui.img_controls.my_scene import MyScene
from gui.img_controls.my_view_zoomable import MyViewZoomable
from PyQt4 import QtGui, QtCore
from gui.img_controls.utils import cvimg2qtpixmap
from core.region.region import Region
from configs.eight import *
from gui.custom_line_selectable import Custom_Line_Selectable
from gui.pixmap_selectable import Pixmap_Selectable

# Number of max boxes below graph, max is six
BOX_NUM = 6
if BOX_NUM > 6:
    BOX_NUM = 6
# Selectable opacity of background color in %
OPACITY = 60
# fixed height of buttons
HEIGHT = 70
# size of font used in labels (px)
FONT_SIZE = 13
# Margins of the workspace
M = 3
#Multiplayer of width of graph
GRAPH_WIDTH = 1

SIMILARITY = 'sim'
STRONG = 's'
CONFIRMED = 'c'
MERGED = 'm'
SPLIT = 'split'


class NodeGraphVisualizer(QtGui.QWidget):

        def __init__(self, g, regions):
            super(NodeGraphVisualizer, self).__init__()
            self.G = g
            self.regions = regions
            self.edges_obj = {}
            self.nodes_obj = {}
            self.pixmaps = {}
            self.selected_edge = [[None, None], None]
            self.show_frames_number = True
            self.used_rows = {}
            self.column_count = 1
            self.frames = []
            self.positions = {}
            self.node_displayed = {}
            self.node_size = NODE_SIZE
            self.y_step = self.node_size + 2
            self.x_step = self.node_size + 200
            self.availability = np.zeros(len(regions))

            self.view = MyViewZoomable(self)
            self.setLayout(QtGui.QVBoxLayout())
            self.edge_info_layout = QtGui.QHBoxLayout()
            self.node_info_layout = QtGui.QHBoxLayout()
            self.edge_info_layout.setSpacing(M)
            self.node_info_layout.setSpacing(M)
            self.view.setMouseTracking(True)
            self.scene = MyScene()
            self.view.setScene(self.scene)
            self.scene.clicked.connect(self.scene_clicked)
            self.layout().addLayout(self.edge_info_layout)
            self.layout().addWidget(self.view)
            self.layout().addLayout(self.node_info_layout)
            self.layout().setContentsMargins(M, M, M, M)

            self.info_label_upper = QtGui.QLabel()
            self.stylesheet_info_label = "font: bold %spx" % FONT_SIZE
            self.info_label_upper.setStyleSheet(self.stylesheet_info_label)
            self.info_label_upper.setText("Frame:\nID:\nCentroid:\nArea:")
            self.info_label_upper.setFixedWidth(HEIGHT)
            self.info_label_upper.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)
            self.edge_info_layout.addWidget(self.info_label_upper)
            self.left_label = QtGui.QLabel()
            self.edge_info_layout.addWidget(self.left_label)
            self.chunk_label = QtGui.QLabel()
            self.edge_info_layout.addWidget(self.chunk_label)
            self.right_label = QtGui.QLabel()
            self.edge_info_layout.addWidget(self.right_label)
            stylesheet = "font: %spx; border-style: solid; border-radius: 25px; border-width: 1.5px" % FONT_SIZE
            self.right_label.setStyleSheet(stylesheet)
            self.left_label.setStyleSheet(stylesheet)
            self.chunk_label.setStyleSheet(stylesheet)
            self.left_label.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
            self.right_label.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
            self.chunk_label.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
            self.hide_button = QtGui.QPushButton("Hide (h)")
            self.hide_button.setStyleSheet("background-color: grey; border-style:outset; border-radius: 25px; \
                        border-width: 2px; border-color: beige; font: bold 14px; min-width:10em; padding 6px")
            self.hide_button.setFixedHeight(HEIGHT)
            self.hide_button.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_H))
            self.hide_button.clicked.connect(self.hide_button_function)
            self.edge_info_layout.addWidget(self.hide_button)
            self.aux_space_upper = QtGui.QLabel()
            self.aux_space_upper.setFixedHeight(HEIGHT)
            self.aux_space_upper.setFixedWidth(0)
            self.upper_widgets = [self.info_label_upper, self.left_label, self.right_label,
                                  self.chunk_label, self.hide_button, self.aux_space_upper]
            self.widgets_hide(self.upper_widgets)
            for label in self.upper_widgets:
                label.setFixedHeight(HEIGHT)

            self.info_label_lower = QtGui.QLabel()
            self.info_label_lower.setStyleSheet(self.stylesheet_info_label)
            self.info_label_lower.setText("Frame:\nID:\nCentroid:\nArea:")
            self.info_label_lower.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)
            self.info_label_lower.setFixedWidth(HEIGHT)
            self.node_info_layout.addWidget(self.info_label_lower)
            self.aux_space_lower = QtGui.QLabel()
            self.aux_space_lower.setFixedHeight(HEIGHT)
            self.aux_space_lower.setFixedWidth(0)
            self.edge_info_layout.addWidget(self.aux_space_upper)
            self.node_info_layout.addWidget(self.aux_space_lower)
            self.auxes = [self.aux_space_upper, self.aux_space_lower]
            self.widgets_hide(self.auxes)
            self.boxes = []
            self.box_aux_count = 0
            for i in range(BOX_NUM):
                label = QtGui.QLabel()
                label.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
                label.setFixedHeight(HEIGHT)
                label.hide()
                self.boxes.append([label, None, None])
                self.node_info_layout.addWidget(label)
            self.clear_all_button = QtGui.QPushButton("Clear All (c)")
            self.clear_all_button.setStyleSheet("background-color: grey; border-style:outset; border-width: 2px;\
                                border-color: beige; font: bold 14px;min-width:10em; border-radius:25px; padding 6px")
            self.clear_all_button.setFixedHeight(HEIGHT)
            self.clear_all_button.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_C))
            self.clear_all_button.clicked.connect(self.clear_all_button_function)
            self.clear_all_button.setFixedHeight(HEIGHT)
            self.node_info_layout.addWidget(self.clear_all_button)
            self.lower_widgets = [self.clear_all_button, self.info_label_lower, self.aux_space_lower]
            self.widgets_hide(self.lower_widgets)

        def scene_clicked(self, click_pos):
            item = self.scene.itemAt(click_pos)
            self.selected_edge[1] = click_pos

            if not item:
                self.widgets_hide(self.upper_widgets)
                self.widgets_hide(self.lower_widgets)
                for box in self.boxes:
                    box[0].hide()
                    # if box[2] != None:
                    #     box[2].setClipped(None)

            if item and isinstance(item, Custom_Line_Selectable):
                e_ = self.edges_obj[item]
                self.selected_edge[0] = e_
                e = self.G[e_[0]][e_[1]]
                try:
                    chunk = e["chunk_ref"]
                except KeyError:
                    chunk = None
                self.edge_labels_update(e_, chunk)

            if item and isinstance(item, Pixmap_Selectable):
                parent_pixmap = item.parent_pixmap
                n_ = self.nodes_obj[parent_pixmap]
                self.node_label_update(n_)

        def node_label_update(self, node):
            if self.info_label_lower.isHidden():
                self.widgets_show(self.lower_widgets)
            self.widgets_show(self.auxes)

            if self.box_aux_count < BOX_NUM:
                label = self.boxes[self.box_aux_count][0]
                label.setText(str(node.frame_) + "\n" + str(node.id_) + "\n" +
                              str(Region.centroid(node)) + "\n" + str(Region.area(node)))

                if self.box_aux_count % 2 == 0:
                    self.boxes[self.box_aux_count][1] = color = random_hex_color_str()
                else:
                    self.boxes[self.box_aux_count][1] = color = \
                        inverted_hex_color_str(self.boxes[self.box_aux_count - 1][1])

                color_alpha = hex2rgb_opacity_tuple(color)
                stylesheet = "font: %spx; color: black; background-color: rgba%s; border-style: dashed; \
                            border-width: 1.5px; border-radius: 25px" % (str(FONT_SIZE), str(color_alpha))
                label.setStyleSheet(stylesheet)
                self.boxes[self.box_aux_count][0] = label
                label.show()

                self.store_border_color(color, node)
                self.update_dashed_borders()
                self.box_aux_count += 1

            else:
                self.box_aux_count = 0
                for box in self.boxes:
                    box[0].hide()
                    box[2].setClipped(None)
                self.scene.update()
                self.node_label_update(node)

        def store_border_color(self, color, node):
            color_rgb = hex2rgb_tuple(color)
            q_color = QtGui.QColor(color_rgb[0], color_rgb[1], color_rgb[2])
            pixmap = self.pixmaps[node]
            pixmap.setClipped(q_color)
            self.boxes[self.box_aux_count][2] = pixmap

        def update_dashed_borders(self):
            if self.box_aux_count > 0:
                label_changed = self.boxes[self.box_aux_count - 1][0]
                color2 = self.boxes[self.box_aux_count - 1][1]
                color2_alpha = hex2rgb_opacity_tuple(color2)
                stylesheet = "font: %spx; color: black; background-color: rgba%s; border-style: solid; border-width: 1.5px; border-radius: 25px" \
                             % (str(FONT_SIZE), str(color2_alpha))
                label_changed.setStyleSheet(stylesheet)

        def edge_labels_update(self, edge, chunk):
            self.left_label.setText(str(edge[0].frame_) + "\n" + str(edge[0].id_) + "\n" +
                                    str(Region.centroid(edge[0])) + "\n" + str(Region.area(edge[0])))
            if chunk is None:
                text = "Not a chunk"
            else:
                text = "Chunk info:" + "\nSorted: " + str(chunk.is_sorted) + "\nReduced nodes: " + str(len(chunk.reduced))
            self.chunk_label.setText(text)
            self.right_label.setText(str(edge[1].frame_) + "\n" + str(edge[1].id_) + "\n" +
                                     str(Region.centroid(edge[1])) + "\n" + str(Region.area(edge[1])))
            self.widgets_show(self.upper_widgets)
            self.widgets_show(self.auxes)

        def widgets_hide(self, labels):
            for label in labels:
                if type(label) is list:
                    for l in label:
                        l.hide()
                else:
                    label.hide()

        def widgets_show(self, labels):
            for label in labels:
                if type(label) is list:
                    for l in label:
                        l.show()
                else:
                    label.show()

        def hide_button_function(self):
            self.widgets_hide(self.upper_widgets)
            if self.clear_all_button.isHidden():
                self.widgets_hide(self.auxes)

        def clear_all_button_function(self):
            for box in self.boxes:
                box[0].hide()
                if box[2] is not None:
                    box[2].setClipped(None)
            self.widgets_hide(self.lower_widgets)
            if self.hide_button.isHidden():
                self.widgets_hide(self.auxes)

        def show_node_with_edges(self, n, prev_pos=0):
            if n in self.node_displayed or n not in self.G.node:
                return

            self.node_displayed[n] = True

            t = n.frame_
            t_num = self.frames.index(t) * GRAPH_WIDTH

            if n in self.positions:
                pos = self.positions[n]
            else:
                pos = self.get_nearest_free_slot(t, prev_pos)
                self.positions[n] = pos

            vis = self.G.node[n]['img']
            if vis.shape[0] > self.node_size or vis.shape[1] > self.node_size:
                vis = np.asarray(resize(vis, (self.node_size, self.node_size)) * 255, dtype=np.uint8)
            else:
                z = np.zeros((self.node_size, self.node_size, 3), dtype=np.uint8)
                z[0:vis.shape[0], 0:vis.shape[1]] = vis
                vis = z

            it = self.scene.addPixmap(cvimg2qtpixmap(vis))
            it.setPos(self.x_step * t_num, self.y_step * pos)
            self.nodes_obj[it] = n
            it_ = Pixmap_Selectable(it, self.node_size)
            self.pixmaps[n] = it_

        def draw_edge_selectable(self, n1, n2):
            t1 = n1.frame_
            t2 = n2.frame_
            t1_framenum = self.frames.index(t1) * GRAPH_WIDTH
            t2_framenums = self.frames.index(t2) * GRAPH_WIDTH

            from_x = self.x_step * t1_framenum + self.node_size
            to_x = self.x_step * t2_framenums
            from_y = self.y_step * self.positions[n1] + self.node_size / 2
            to_y = self.y_step * self.positions[n2] + self.node_size / 2

            line_ = QtCore.QLineF(from_x, from_y, to_x, to_y)
            custom_line_ = Custom_Line_Selectable(line_)

            self.scene.addItem(custom_line_)
            self.edges_obj[custom_line_] = (n1, n2)

        def prepare_positions(self, frames):
            for f in frames:
                for n1 in self.regions[f]:
                    if n1 not in self.G.node:
                        continue
                    if n1 in self.positions:
                        continue

                    for _, n2, d in self.G.out_edges(n1, data=True):
                            if n2 in self.positions:
                                continue

                            t1 = n1.frame_
                            t2 = n2.frame_
                            t1_framenum = self.frames.index(t1) * GRAPH_WIDTH
                            t2_framenum = self.frames.index(t2) * GRAPH_WIDTH
                            p1 = self.get_nearest_free_slot(t1_framenum, 0)
                            p2 = self.get_nearest_free_slot(t2_framenum, p1)

                            self.positions[n1] = p1
                            self.positions[n2] = p2

                            for t in range(t1_framenum, t2_framenum):
                                if t in self.used_rows:
                                    self.used_rows[t][p1] = True
                                else:
                                    self.used_rows[t] = {p1: True}

        def get_nearest_free_slot(self, t, pos):
            if t in self.used_rows:
                step = 0
                while True:
                    test_pos = pos - step
                    if test_pos > -1 and test_pos not in self.used_rows[t]:
                        self.used_rows[t][test_pos] = True
                        return test_pos
                    if pos + step not in self.used_rows[t]:
                        self.used_rows[t][pos + step] = True
                        return pos + step
                    step += 1
            else:
                self.used_rows[t] = {pos: True}
                return pos

        def visualize(self):
            k = np.array(self.regions.keys())
            frames = np.sort(k)
            for f in frames:
                self.frames.append(f)
            self.prepare_positions(frames)

            nodes_queue = []

            visited = {}
            for f in frames:
                for r in self.regions[f]:
                    if r in visited or r not in self.G.node:
                        continue

                    temp_queue = [r]

                    while True:
                        if not temp_queue:
                            break

                        n = temp_queue.pop()
                        if n in visited:
                            continue

                        visited[n] = True
                        nodes_queue.append(n)
                        for e_ in self.G.out_edges(n):
                            temp_queue.append(e_[1])

            for n in nodes_queue:
                self.show_node_with_edges(n)

            for e in self.G.edges():
                self.draw_edge_selectable(e[0], e[1])

            for f in frames:
                if self.show_frames_number:
                    f_num = self.frames.index(f) * GRAPH_WIDTH
                    t_ = QtGui.QGraphicsTextItem(str(f))

                    t_.setPos(self.x_step * f_num + self.node_size * 0.2, -20)
                    self.scene.addItem(t_)

def visualize_nodes(im, r):
    vis = draw_points_crop(im, r.pts(), square=True, color=(0, 255, 0, 0.35))
    cv2.putText(vis, str(r.id_), (1, 10), cv2.FONT_HERSHEY_PLAIN, 0.55, (255, 255, 255), 1, cv2.cv.CV_AA)

    return vis


def random_hex_color_str():
    # color = "#%06x" % random.randint(0, 0xFFFFFF)
    rand_num = random.randint(1, 3)
    l1 = "0123456789abcdef"
    color = "#"
    for i in range(1, 4):
        if i == rand_num:
            color += "ff"
        else:
            color += (l1[random.randint(0, 15)] + l1[random.randint(0, 15)])
    return color


def inverted_hex_color_str(color):
    string = str(color).lower()
    code = {}
    l1 = "#;0123456789abcdef"
    l2 = "#;fedcba9876543210"

    for i in range(len(l1)):
        code[l1[i]] = l2[i]

    inverted = ""

    for j in string:
        inverted += code[j]

    return inverted


def hex2rgb_tuple(color):
    rgb = colors.hex2color(color)
    return_list = [int(255 * x) for x in rgb]
    return tuple(return_list)


def hex2rgb_opacity_tuple(color):
    rgb = colors.hex2color(color)
    rgb_list = [int(255 * x) for x in rgb]
    rgb_list.append(OPACITY)
    string_color = "(%s, %s, %s, %s%%)" % (rgb_list[0], rgb_list[1],
                                           rgb_list[2], rgb_list[3])
    return string_color
