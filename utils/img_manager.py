from utils.img import prepare_for_visualisation, get_safe_selection
from utils.video_manager import get_auto_video_manager
from utils.drawing.points import draw_points
from utils.roi import ROI, get_roi
import numpy as np
import cv2
from core.region.region import Region


class ImgManager:
    def __init__(self, project, max_size_mb=-1, max_num_of_instances=-1):
        """
        This class can be used to load images from FERDA videos. It keeps used images in cache and is able to provide
        them quickly. It also offers methods that crop_ images if necessary.

        :param project: project to load images from
        :param max_size_mb: max volume of images kept in cache (unlimited by default)
        :param max_num_of_instances: max count of images kept in cache (unlimited by default, only used when max_size_mb is not set)
        """
        self.project = project
        self.vid = get_auto_video_manager(project)
        self.crop_cache = {}
        self.crop_properties = []
        self.max_size_mb = max_size_mb
        self.max_num_of_instances = max_num_of_instances

    def get_whole_img(self, frame):
        """
        Load a frame from project video/cache, add it to cache and
        :param frame: int <0, len(video)> frame to load
        :return: numpy uint8 array
        """

        # create properties to describe the image
        props = Properties(frame, False)

        # check if these properties are already saved
        for p, img in self.crop_cache.items():
            # if so:
            if p.__eq__(props):
                # re-append it to the end of self.crop_properties list (used recently)
                # remove it from the frames list
                self.crop_properties.remove(props)
                # add it again
                self.crop_properties.append(props)
                # return a copy
                return img.copy()

        # if the image isn't in the cache, load it and add it
        image = prepare_for_visualisation(self.vid.get_frame(frame), self.project)

        # check if cache isn't full (and maybe clean it)
        self.check_cache_size(image.nbytes)

        # add new image and it's properties to list and dictionary
        self.crop_properties.append(props)
        self.crop_cache[props] = image
        return image.copy()

    def get_crop(self, frame, roi, margin=0, relative_margin=0, width=-1, height=-1, fill_color=(0, 0, 0),
                 max_width=-1, max_height=-1,  min_width=-1, min_height=-1, regions=[], colors={},
                 default_color=(0, 0, 255, 0.8), constant_propotions=True):
        """
        Gets a crop_ around the given ROI. The crop_ can be modified with the following parameters.
        :param frame:                         int (<0, len(video)>)
        Frame number to make the crop_ from
        :param roi:                           list of ROI, or tuple with ROI parameters
        Regions of interest
        :param margin:                        int (<0, infinity>)           default=-1
        Absolute margin in px
        :param relative_margin:               float (<0, infinity>)         default=-1
        Margin relative to roi size. It overrides "regular" margin
        :param constant_propotions:           boolean                       default=True
        Keep proportions when scaling image
        :param width, height:                 int (<0, infinity>)           default=-1
        Scale image to width (px). Both variables have to be set. If constant_proportions is False, the image is
        deformed and stretched to fit the dimensions exactly.
        :param fill_color:                    tuple(0-255, 0-255, 0-255)    default=(0, 0, 0)
        If width and height are both set and constant_proportions is True, image is scaled to one of the dimensions and
        the rest is filled with fill_color. Fill color should be in (r, g, b) format.
        :param max_width, max_height:         int (<0, infinity>)           default=-1
        The image will be scaled to be at least the given size. If constant proportions are set, it will be expanded to
        fit both max_width and max_height, otherwise it will be deformed and stretched to exactly [max_width,max_height]
        :param min_width, min_height:         int (<0, infinity>)           default=-1
        The image will be scaled to be smaller than the given size. If constant proportions are set, it will be shrunk
        to be smaller than min_width, min_height. Otherwise it will be deformed and stretched to [min_width,min_height]
        :param regions:
        :param colors:
        :param default_color:
        :return: numpy uint8 array
        """

        # one region
        if isinstance(roi, Region):
            # append it to regions - so it will be visualised
            if regions:
                regions.append(roi)
            else:
                regions = [roi]

            roi = get_roi(roi.pts())

        # list of regions
        elif isinstance(roi, list):
            pts = np.empty((0, 2), int)
            for r in roi:
                pts = np.append(pts, r.roi().corner_pts(), axis=0)

            roi = get_roi(pts)

        elif isinstance(roi, tuple):
            roi = ROI(roi[0], roi[1], roi[2], roi[3])

        # create properties to describe the crop_
        props = Properties(frame, True, roi, margin, relative_margin, width, height, 0, 0,
                           regions, colors, default_color, fill_color)

        # check if this crop_ is already cached
        for p in self.crop_properties:
            if props.__eq__(p):
                self.crop_properties.remove(p)
                self.crop_properties.append(p)
                # return it if so
                return self.crop_cache[props].copy()

        # otherwise load the frame and modify it
        im = self.get_whole_img(frame)

        # is there anything to visualise?
        if regions:
            for r in regions:
                c = default_color
                if r in colors:
                    c = colors[r]
                    # TODO: deal with opacity...
                    # dita: It seems to be working fine even without it
                    # if len(c) > 4:
                    #     c = c[0:3]

                draw_points(im, r.pts(), c)


        # expand the ROI by margin
        if relative_margin > 0:
            m_ = max(width, height)
            margin = m_ * relative_margin

        y_ = roi.y() - margin
        x_ = roi.x() - margin
        height_ = roi.height() + 2 * margin
        width_ = roi.width() + 2 * margin

        # get image with the crop_
        crop = get_safe_selection(im, y_, x_, height_, width_, fill_color=fill_color)

        # scale the crop_
        scaled = self.scale_crop(crop, width, height, max_width, max_height, min_width, min_height, constant_propotions=constant_propotions)

        # check if cache isn't full (and maybe clean it)
        self.check_cache_size(scaled.nbytes)

        # add new image and it's properties to list and dictionary
        self.crop_cache[props] = scaled
        self.crop_properties.append(props)
        return scaled

    def scale_crop(self, crop, width=-1, height=-1, max_width=-1, max_height=-1, min_width=-1, min_height=-1,
                constant_propotions=True, fill_color=(0, 0, 0)):
        """
        Scales the crop_ according to the parameters.
        :param crop:                          numpy uint8 array
        Image or crop_ to work with
        :param constant_propotions:           boolean                       default=True
        Keep proportions when scaling image
        :param width, height:                 int (<0, infinity>)           default=-1
        Scale image to width (px). Both variables have to be set. If constant_proportions is False, the image is
        deformed and stretched to fit the dimensions exactly.
        :param fill_color:                    tuple(0-255, 0-255, 0-255)    default=(0, 0, 0)
        If width and height are both set and constant_proportions is True, image is scaled to one of the dimensions and
        the rest is filled with fill_color. Fill color should be in (r, g, b) format.
        :param max_width, max_height:         int (<0, infinity>)           default=-1
        The image will be scaled to be at least the given size. If constant proportions are set, it will be expanded to
        fit both max_width and max_height, otherwise it will be deformed and stretched to exactly [max_width,max_height]
        :param min_width, min_height:         int (<0, infinity>)           default=-1
        The image will be scaled to be smaller than the given size. If constant proportions are set, it will be shrunk
        to be smaller than min_width, min_height. Otherwise it will be deformed and stretched to [min_width,min_height]
        :return: scaled crop_ (numpy uint8 array)
        """

        # convert BGR <-> RGB
        b = fill_color[0]
        g = fill_color[1]
        r = fill_color[2]
        fill_color=(r, g, b)

        # make variables with shape to have easier access
        cr_height = crop.shape[0]
        cr_width = crop.shape[1]

        # if both width and height are set
        if width > 0 and height > 0:
            scaley = width / (cr_width + 0.0)
            scalex = height / (cr_height + 0.0)
            # if the image should not be deformed
            if constant_propotions and scaley != scalex:
                new_image = np.zeros((height, width, 3), dtype=np.uint8)
                new_image[:] = fill_color
                if scaley < scalex:
                    # create top and bottom borders
                    resized = cv2.resize(crop, (0,0), fx=scaley, fy=scaley)
                    border = int((height - resized.shape[1])/2.0)
                    new_image[border:border+resized.shape[0], :] = resized
                    return new_image
                else:
                    # create left and right borders
                    resized = cv2.resize(crop, (0,0), fx=scalex, fy=scalex)
                    border = int((width - resized.shape[0])/2.0)
                    new_image[:, border:border+resized.shape[1]] = resized
                    return new_image
            # return deformed image
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

        # no comments needed here, it works exactly the same way as max_height and max_width check
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

    def check_cache_size(self, file_size):
        """
        This method cleans cache so the file with file_size can fit in there.
        :param file_size: size of file in bytes
        :return: None
        """
        # check MB size if it is given
        if self.max_size_mb > 0:
            tmp_size = self.get_cache_size_bytes()
            # remove files from cache until there is enough space for "file_size"
            while(tmp_size + file_size > self.max_size_mb*1048576.0):
                self.crop_cache.pop(self.crop_properties.pop(0), None)
                tmp_size = self.get_cache_size_bytes()
            # print "Cache size: %.2f/%s MB, %s items" % (tmp_size/1048576.0, self.max_size_mb, len(self.crop_cache))

        # if MB size is not set, try to use cache length limit
        elif self.max_num_of_instances > 0:
            if len(self.crop_cache) > self.max_num_of_instances:
                self.crop_cache.pop(self.crop_properties.pop(0), None)
            # print "Cache size: %.2f MB, %s/%s items" % (self.get_cache_size_bytes()/1048576.0, len(self.crop_cache), self.max_num_of_instances)

        # if limits aren't set, do nothing
        else:
            # print "Cache size: %.2f MB, %s items" % (self.get_cache_size_bytes()/1048576.0, len(self.crop_cache))
            pass

    def get_cache_size_bytes(self):
        size = 0
        for props, image in self.crop_cache.items():
            size += image.nbytes
        return size


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
                wrap_height=-1, regions=[], colors={}, default_color=(255, 255, 255, 0.8),
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
        self.regions = regions
        self.colors = colors
        self.default_color = default_color
        self.fill_color = fill_color

    def __hash__(self):
        # TODO: fix this! Two same hashes may be created with this
        if self.is_crop:
            return hash(self.frame + self.roi.x()*10 + self.roi.y()*100 + self.roi.width()*1000
                        + self.roi.height()*10000 + self.margin*100000 + self.relative_margin*1000000
                        + self.width*10000000 + self.height*100000000 + self.wrap_height*1000000000
                        + self.wrap_width*10000000000)
        else:
            return hash(self.frame)

    def __eq__(self, prop):
        if self.is_crop and prop.is_crop:
            return self.frame == prop.frame and self.margin == prop.margin and self.relative_margin == prop.relative_margin \
                and self.width == prop.width and self.height == prop.height and self.wrap_height == prop.wrap_height \
                and self.wrap_width == prop.wrap_width \
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
    if rnd == 1:
        rnd = random.randint(0, 10)
        rnd *= 100
        im = im_manager.get_whole_img(rnd)
        return im
    else:
        rnd = random.randint(0, 10)
        rnd *= 100
        roi = []
        roi.append(ROI(200, 200, 400, 400))
        roi.append(ROI(500, 500, 400, 400))
        im = im_manager.get_crop(rnd, roi, width=250, height=300)
        return im
    return im


if __name__ == "__main__":
    from core.project.project import Project
    p = Project()
    # p.load('/Users/flipajs/Documents/wd/eight_22/eight22.fproj')
    p.load('/home/dita/PycharmProjects/eight_22/eight22.fproj')

    im_manager = ImgManager(p)

    import random
    image = get_image(im_manager)
    cv2.imshow("im", image)
    print "Press SPACE to show another image"
    key = cv2.waitKey(0)
    while key == 32:
        rnd = random.randint(0, 10)
        rnd *= 100

        # import time
        # t = time.time()
        image = get_image(im_manager)
        # print "Time taken: %s" % (time.time() - t)
        cv2.imshow("im", image)
        key = cv2.waitKey(0)
    print "done"
