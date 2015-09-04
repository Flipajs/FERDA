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
        self.whole_images_frames = []  # since dictionary changes its' order, list must be used to hold it
        self.crop_cache = {}
        self.crop_frames = []

    def get_whole_img(self, frame):
        print "Saved frames: %s" % self.whole_images_frames
        for f, img in self.whole_images_cache.items():
            if f == frame:
                # remove it from the frames list
                self.whole_images_frames.remove(f)
                # add it again so it doesn't get erased as unused
                self.whole_images_frames.append(f)
                print "Image %s was cached" % f
                return img.copy()
        # if the image isn't in the cache, load it and add it
        image = prepare_for_visualisation(self.vid.get_frame(frame), self.project)

        # only save last 5 images
        if len(self.whole_images_cache) > 4:
            self.whole_images_cache.pop(self.whole_images_frames.pop(0), None)
        self.whole_images_frames.append(frame)
        self.whole_images_cache[frame] = image
        print "Image %s wasn't cached" % frame
        return image

    def get_ccrop(self, frame, x0, y0, x1, y1):
        fm = Frame(frame, x0, y0, x1, y1)
        for f, crop in self.crop_cache.items():
            if f.equals(fm):
                # remove it from the frames list
                self.crop_frames.remove(f)
                # add it again so it doesn't get erased as unused
                self.crop_frames.append(f)
                print "Crop %s was cached" % f.frame
                return crop.copy()
        # if the image isn't in the cache, load it and add it
        image = prepare_for_visualisation(self.vid.get_frame(fm.frame), self.project)
        result = image[y0:y1,x0:x1]

        # only save last 5 images
        if len(self.crop_cache) > 4:
            self.crop_cache.pop(self.crop_frames.pop(0), None)
        self.crop_frames.append(fm)
        self.crop_cache[fm] = result
        print "Crop %s wasn't cached" % fm.frame
        return result

    def get_crop(self, frame, roi, margin=0, relative_margin=0, width=-1, height=-1, max_width=-1, max_height=-1,
                 min_width=-1, min_height=-1, regions=[], colors={}, default_color=(255, 255, 255, 0.8), constant_propotions=True, fill_color=(0, 0, 0)):
        """

        :param frame:
        :param roi: list of regions or (y, x, height, width) or class ROI
        :param margin: in pixels
        :param relative_margin: <0, inf>, relative to the ROI (region of interest / bounding box) computed from regions
        :param width:
        :param height:
        :param max_width: result image will be scaled into these. If none is set, it will stay as it is. If width and
        height is set, it will be strictly scaled to this shape...
        :param max_height:
        :param min_width:
        :param min_height:
        :param regions: empty -> no visualisation
        :param colors: empty -> default colors, else it is a dict.... colors[region[0]] ...
        :param default_color:
        :param constant_propotions: if True the max(width, heigth) will be choosen and the rest will be filled with fill_color
        :param fill_color: see previous line
        :return: cropped image
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
            margin = m_ * relative_margin

        y_ = roi.y() - margin
        x_ = roi.x() - margin
        height_ = roi.height() + 2 * margin
        width_ = roi.width() + 2 * margin

        crop = get_safe_selection(im, y_, x_, height_, width_, fill_color=fill_color)

        # resize TODO...
        # h_, w_, _ = crop.shape()



        if height <= 0:
            scalex = 1
        else:
            scalex = width / (roi.height() + margin + 0.0)

        if width <= 0:
            scaley = 1
        else:
            scaley = height / (roi.width() + margin + 0.0)

        print "roi width: %s, roi height: %s, margin: %s" % (roi.width(), roi.height(),
                                                             margin)
        print "xscale: %s, yscale: %s" % (scalex, scaley)

        dst = cv2.resize(crop, (0,0), fx=scalex, fy=scaley)

        return dst


class Frame:
    def __init__(self, frame, x0, y0, x1, y1):
        self.frame = frame
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    def equals(self, crop):
        if self.frame == crop.frame and self.x0 == crop.x0 and self.y0 == crop.y0 and self.x1 == crop.x1 and self.y1 == crop.y1:
            return True
        return False


if __name__ == "__main__":
    p = Project()
    #p.load('/Users/flipajs/Documents/wd/eight_22/eight22.fproj')
    p.load('/home/dita/PycharmProjects/eight/eight.fproj')

    im_manager = ImgManager(p)

    import cv2
    #solver = p.saved_progress['solver']
    #nodes = solver.nodes_in_t[0]

    import random
    rnd = random.randint(0, 10)
    rnd *= 100
    # im = im_manager.get_crop(0, nodes[0:3], regions=nodes, relative_margin=0.1)
    im = im_manager.get_whole_img(rnd)
    cv2.imshow("im", im)
    print "Press SPACE to show another image"
    key = cv2.waitKey(0)
    while key == 32:
        rnd = random.randint(0, 10)
        rnd *= 100

        import time
        t = time.time()
        r = ROI(400, 400, 400, 400)
        im = im_manager.get_crop(rnd, r, width=100, height=300)
        #im = im_manager.get_ccrop(rnd, 400, 400, 800, 800)
        #im = im_manager.get_whole_img(rnd)
        print "Time taken: %s" % (time.time() - t)
        cv2.imshow("im", im)
        key = cv2.waitKey(0)
    print "done"
