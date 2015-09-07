__author__ = 'flipajs'

from core.project.project import Project
from utils.img import prepare_for_visualisation, get_safe_selection
from utils.video_manager import get_auto_video_manager
from utils.drawing.points import draw_points
from utils.roi import ROI, get_roi
import numpy as np
import cv2
from cv2 import copyMakeBorder as make_border


class ImgManager:
    def __init__(self, project, max_num_of_instances=-1, max_size_mb=-1):
        self.project = project
        self.vid = get_auto_video_manager(project)
        self.crop_cache = {}
        self.crop_properties = []
        self.max_size_mb = max_size_mb

    def get_whole_img(self, frame):

        props = Properties(frame, False)
        for p, img in self.crop_cache.items():
            if p.__eq__(props):
                # remove it from the frames list
                self.crop_properties.remove(props)
                # add it again so it doesn't get erased as unused
                self.crop_properties.append(props)
                return img.copy()
        # if the image isn't in the cache, load it and add it
        image = prepare_for_visualisation(self.vid.get_frame(frame), self.project)

        self.check_cache_size()
        self.crop_properties.append(props)
        self.crop_cache[props] = image
        return image

    def new_get_crop(self, frame, roi, margin=0, relative_margin=0, width=-1, height=-1, border_color=[255, 255, 255],
                 max_width=-1, max_height=-1,  min_width=-1, min_height=-1, regions=[], colors={},
                 default_color=(255, 255, 255, 0.8), constant_propotions=True, fill_color=(0, 0, 0)):
        cache = ""
        for p in self.crop_properties:
            cache += (str(p.frame) + " ")
        print cache


        # list of regions
        if isinstance(roi, list):
            pts = np.empty((0, 2), int)
            for r in roi:
                pts = np.append(pts, r.pts(), axis=0)

            roi = get_roi(pts)

        elif isinstance(roi, tuple):
            roi = ROI(roi[0], roi[1], roi[2], roi[3])

        props = Properties(frame, True, roi, margin, relative_margin, width, height, 0, 0,
                           border_color, regions, colors, default_color, fill_color)

        for p in self.crop_properties:
            if props.__eq__(p):
                print "Already in cache!"
                # remove it from the frames list
                self.crop_properties.remove(p)
                # add it again so it doesn't get erased as unused
                self.crop_properties.append(p)

                return self.crop_cache[props]

        print "Not in cache"

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
        height_ = roi.width() + 2 * margin
        width_ = roi.height() + 2 * margin

        crop = get_safe_selection(im, y_, x_, height_, width_, fill_color=fill_color)

        scaled = self.scale_crop(crop, width, height, max_width, max_height, min_width, min_height, constant_propotions=constant_propotions)


        self.check_cache_size()
        self.crop_cache[props] = scaled
        self.crop_properties.append(props)
        return scaled

    def get_crop(self, frame, roi, margin=0, relative_margin=0, width=-1, height=-1, wrap_width=-1, wrap_height=-1, border_color=[255, 255, 255], max_width=-1, max_height=-1,
                 min_width=-1, min_height=-1, regions=[], colors={}, default_color=(255, 255, 255, 0.8), constant_propotions=True, fill_color=(0, 0, 0)):
        """

        :param frame:
        :param roi: region of interest
        :param margin: area around roi that has to be included in the image
        :param relative_margin: <0, inf>
        :param width: width of the new image (if both width and height are set, image may be deformed)
        :param height: height of the new image (if both width and height are set, image may be deformed)
        :param wrap_width: enlarge the image from width to wrap_width and fill the border with border_color
        :param wrap_height: enlarge the image from height to wrap_height and fill the border with border_color
        :param border_color: list[r, g, b]
        :param regions:
        :param colors:
        :param default_color:
        :param fill_color:
        :return:
        """
        """
        OLD
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
        cache = ""
        for p in self.crop_properties:
            cache += (str(p.frame) + " ")
        print cache


        # list of regions
        if isinstance(roi, list):
            pts = np.empty((0, 2), int)
            for r in roi:
                pts = np.append(pts, r.pts(), axis=0)

            roi = get_roi(pts)

        elif isinstance(roi, tuple):
            roi = ROI(roi[0], roi[1], roi[2], roi[3])

        props = Properties(frame, True, roi, margin, relative_margin, width, height, wrap_width, wrap_height,
                           border_color, regions, colors, default_color, fill_color)

        for p in self.crop_properties:
            if props.__eq__(p):
                print "Already in cache!"
                # remove it from the frames list
                self.crop_properties.remove(p)
                # add it again so it doesn't get erased as unused
                self.crop_properties.append(p)

                return self.crop_cache[props]

        print "Not in cache"

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
        height_ = roi.width() + 2 * margin
        width_ = roi.height() + 2 * margin

        crop = get_safe_selection(im, y_, x_, height_, width_, fill_color=fill_color)

        # no scaling
        if width <= 0 and height <= 0:
            scalex = 1
            scaley = 1
        else:
            # scale only width
            if width > 0 and height <= 0:
                scalex = width / (width_ + 0.0)
                scaley = scalex
            # scale only height
            elif height > 0 and width <= 0:
                scaley = height / (height_ + 0.0)
                scalex = scaley
            # scale both
            else:
                scaley = height / (height_ + 0.0)
                scalex = width / (width_ + 0.0)

        scaled = cv2.resize(crop, (0,0), fx=scalex, fy=scaley)
        height = scaled.shape[0]
        width = scaled.shape[1]

        if wrap_height > height:
            border_height = (wrap_height - height) / 2.0
            scaled = make_border(scaled, int(border_height), int(border_height), 0, 0, cv2.BORDER_CONSTANT,value=border_color)
        if wrap_width > width:
            border_width = (wrap_width - width) / 2.0
            scaled = make_border(scaled, 0, 0, int(border_width), int(border_width), cv2.BORDER_CONSTANT,value=border_color)

        self.check_cache_size()
        self.crop_cache[props] = scaled
        self.crop_properties.append(props)
        return scaled

    def scale_crop(self, crop, width=-1, height=-1, max_width=-1, max_height=-1, min_width=-1, min_height=-1,
                constant_propotions=True, fill_color=(0, 0, 0)):
        cr_height = crop.shape[0]
        cr_width = crop.shape[1]
        if width > 0 and height > 0:
            scaley = height / (cr_height + 0.0)
            scalex = width / (cr_width + 0.0)
            if constant_propotions and scaley != scalex:
                #return nothing if image proportions would change
                print "ERROR: Invalid arguments! Constant proportions can't be used when width and height are set!"
                return False
            return cv2.resize(crop, (0,0), fx=scalex, fy=scaley)

        # if max dimensions are set
        if max_height > 0 or max_width > 0:
            # set default scaling
            scalex = 1
            scaley = 1
            # get scale values needed
            if cr_height > max_height and max_height > 0:
                scaley = max_height / (cr_height + 0.0)
            if cr_width > max_width and max_width > 0:
                scalex = max_width / (cr_width + 0.0)
            print "scalex: %s, scaley: %s" % (scalex, scaley)

            if not constant_propotions:
                # scale exactly to [max_height, or max_width]
                return cv2.resize(crop, (0,0), fx=scalex, fy=scaley)
            # else choose the smaller scale (so image fits in both dimensions if set)
            elif scalex < scaley:
                return cv2.resize(crop, (0,0), fx=scalex, fy=scalex)
            else:
                return cv2.resize(crop, (0,0), fx=scaley, fy=scaley)

        if min_height > 0 or min_width > 0:
            scalex = 1
            scaley = 1
            if cr_height < min_height and min_height > 0:
                scaley = min_height / (cr_height + 0.0)
            if cr_width > min_width and min_width > 0:
                scalex = min_width / (cr_width + 0.0)

            if not constant_propotions:
                return cv2.resize(crop, (0,0), fx=scalex, fy=scaley)
            elif scalex > scaley:
                return cv2.resize(crop, (0,0), fx=scalex, fy=scalex)
            else:
                return cv2.resize(crop, (0,0), fx=scaley, fy=scaley)

        return crop




    def check_cache_size(self):
        size = 0
        for props, image in self.crop_cache.items():
            size += image.nbytes

        print "Size: %s" % size

        # TODO Convert bytes to mbytes
        if self.max_size_mb != -1 and size > self.max_size_mb:
            self.crop_cache.pop(self.crop_properties.pop(0), None)


class Frame:
    def __init__(self, frame, x0, y0, x1, y1):
        self.frame = frame
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    def equals(self, crop):
        return self.frame == crop.frame and self.x0 == crop.x0 and self.y0 == crop.y0 and self.x1 == crop.x1 and self.y1 == crop.y1

class Properties:
    def __init__(self, frame, is_crop, roi=ROI(), margin=0, relative_margin=0, width=-1, height=-1, wrap_width=-1,
                wrap_height=-1, border_color=[255, 255, 255], regions=[], colors={}, default_color=(255, 255, 255, 0.8),
                fill_color=(0, 0, 0)):
        self.frame = frame
        self.roi = roi
        self.is_crop = is_crop
        self.margin = margin
        self.relative_margin = relative_margin
        self.width = width
        self.height = height
        self.wrap_width = wrap_width
        self.wrap_height = wrap_height
        self.border_color = border_color
        self.regions = regions
        self.colors = colors
        self.default_color = default_color
        self.fill_color = fill_color

    def __hash__(self):
        # TODO: fix this! Two same hashes may be created with this
        if self.is_crop:
            return hash(self.frame + self.roi.x() + self.roi.y() + self.roi.width() + self.roi.height() + self.margin\
               + self.relative_margin + self.width + self.height + self.wrap_height + self.wrap_width)
        else:
            return hash(self.frame)
        #return hash((self.frame, self.roi, self.margin, self.relative_margin, self.width, self.height, self.wrap_width,
        #    self.wrap_height, self.border_color, self.regions, self.colors, self.default_color, self.fill_color))

    def __eq__(self, prop):
        if self.is_crop and prop.is_crop:
            return self.frame == prop.frame and self.margin == prop.margin and self.relative_margin == prop.relative_margin \
                and self.width == prop.width and self.height == prop.height and self.wrap_height == prop.wrap_height \
                and self.wrap_width == prop.wrap_width and self.border_color == prop.border_color \
                and self.regions == prop.regions and self.colors == prop.colors \
                and self.default_color == prop.default_color and self.fill_color == prop.fill_color \
                and self.roi.width() == prop.roi.width() and self.roi.height() == prop.roi.height() \
                and self.roi.x() == prop.roi.x() and self.roi.y() == prop.roi.y()
        elif not self.is_crop and not prop.is_crop:
            return self.frame == prop.frame
        else:
            return False


def get_image(im_manager):
    rnd = random.randint(0, 10)

    rnd *= 100
    r = ROI(200, 200, 400, 400)
    im = im_manager.new_get_crop(rnd, r, min_height=600, min_width=500, constant_propotions=True)
    """
    if rnd == 1:
        print "Getting whole image"
        rnd = random.randint(0, 10)
        rnd *= 100
        im = im_manager.get_whole_img(rnd)
        return im
    else:
        print "Getting crop"
        rnd = random.randint(0, 10)
        rnd *= 100
        r = ROI(200, 200, 400, 400)
        im = im_manager.get_crop(rnd, r, width= 300, height=300, wrap_width=400, wrap_height=400)
        return im"""
    return im


if __name__ == "__main__":
    p = Project()
    # p.load('/Users/flipajs/Documents/wd/eight_22/eight22.fproj')
    p.load('/home/dita/PycharmProjects/eight_22/eight22.fproj')

    im_manager = ImgManager(p)

    # solver = p.saved_progress['solver']
    # nodes = solver.nodes_in_t[0]

    import random
    image = get_image(im_manager)
    cv2.imshow("im", image)
    print "Press SPACE to show another image"
    key = cv2.waitKey(0)
    while key == 32:
        rnd = random.randint(0, 10)
        rnd *= 100

        import time
        t = time.time()
        image = get_image(im_manager)
        print "Time taken: %s" % (time.time() - t)
        cv2.imshow("im", image)
        key = cv2.waitKey(0)
    print "done"
