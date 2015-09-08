__author__ = 'flipajs'

from core.project.project import Project
from utils.img import prepare_for_visualisation, get_safe_selection
from utils.video_manager import get_auto_video_manager
from utils.drawing.points import draw_points
from utils.roi import ROI, get_roi
import numpy as np


class ImgManager:
    def __init__(self, project, max_num_of_instances=-1, max_size_mb=-1):
        self.project = project
        self.vid = get_auto_video_manager(project)
        self.whole_images_cache = {}

    def get_whole_img(self, frame):
        im = prepare_for_visualisation(self.vid.get_frame(frame), self.project)

        # TODO: if it is from cache, return .copy()
        return im

    def get_crop(self, frame, roi, margin=0, relative_margin=0, width=-1, height=-1, max_width=-1, max_height=-1,
                 min_width=-1, min_height=-1, visualise=False, regions=[], colors=[], default_color=(255, 255, 255, 0.8), constant_propotions=True, fill_color=(0, 0, 0)):
        """
        roi list of regions or (y, x, height, width) or class ROI
        margin - in pixels
        relative_margin - <0, inf>, relative to the ROI (region of interest / bounding box) computed from regions

        (min/max) width/height - result image will be scaled into these.
        If none is set, it will stay as it is.
        if width and height is set, it will be strictly scaled to this shape...

        regions - empty -> no visualisation
        colors - empty -> default colors, else it is a dict.... colors[region[0]] ...

        constant_proportions ... if True the max(width, heigth) will be choosen and the rest will be filled with fill_color
        fill_color - see previous line...


        :param regions:
        :param margin:
        :param relative_margin:
        :return:
        """

        # list of regions
        if isinstance(roi, list):
            pts = np.empty((0, 2), int)
            for r in roi:
                pts = np.append(pts, r.pts(), axis=0)

            roi = get_roi(pts)
        elif isinstance(roi, tuple):
            roi = ROI(roi[0], roi[1], roi[2], roi[3])

        im = self.get_whole_img(frame)

        # is there anything to visualise?
        if regions:
            for r in regions:
                c = default_color
                if r in colors:
                    c = colors[r]

                # # TODO: deal with opacity...
                # if len(c) > 4:
                #     c = c[0:3]

                draw_points(im, r.pts(), c)

        if relative_margin > 0:
            m_ = max(width, height)
            margin = m_ * margin

        y_ = roi.y() - margin
        x_ = roi.x() - margin
        height_ = roi.height() + 2 * margin
        width_ = roi.width() + 2 * margin

        crop = get_safe_selection(im, y_, x_, height_, width_, fill_color=fill_color)

        # resize TODO...
        # h_, w_, _ = crop.shape()

        return crop


if __name__ == "__main__":
    p = Project()
    p.load('/Users/flipajs/Documents/wd/eight_22/eight22.fproj')

    im_manager = ImgManager(p)

    import cv2
    solver = p.saved_progress['solver']
    nodes = solver.nodes_in_t[0]

    im = im_manager.get_crop(0, nodes[0:3], regions=nodes, relative_margin=0.1)
    cv2.imshow("im", im)
    cv2.waitKey(0)