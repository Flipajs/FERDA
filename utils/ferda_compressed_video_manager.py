import cv_compatibility

__author__ = 'filip@naiser.cz'

import cv2
import utils.misc


class FerdaCompressedVideoManager():
    """
    this class arranges easy operations with FERDA compressed video files.

    FERDA compressed video consists of 2 video files:
    > compressed one (we recommend:)
        avconv              (you can use ffmpeg if avconv is not available)
            -r 15           (forces input to be 15 FPS)
            -i input.avi
            -c:v libx264    (using h264 codec)
            -crf 25         (compression might be changed higher number means higher compression
                             rate thus lower image quality and lower video file size)
            -g 30           (sets maximum frame number between key frames, can be changed
                             but must be the same value as keyint_min to support frame seek without problems)

            -r 15           (forces output to be 15 FPS, can be changed but must be the same
                             as first one -r for input to support frame seek without problems)
            -keyint_min 30  (sets minimum frame number between key frames. We need the same value as
                             for -g, then key frames are forced to be each 30th frame).
            output_compressed.avi

        if there are some compability issues with avconv, there is another option. Unfold whole
        video into frames and then folding it together:

        avconv -i orig.avi -f image2 -q:v 1 '%07d.png'
        avconv -f image2 -i '%07d.png' -c:v libx264 -crf 25 -g 30 out.avi

    > lossless video file as an output of scripts/segmentation.py compressed by followin command:
       avconv
            -f image2
            -i "frames/%07d.png"
            -c:v libx264
            -crf 0                  (this ensure video lossless)
            -g 30
            -keyint_min 30
            segmented.avi

        as the output of scripts/segmentation.py file is image where everything except interesting regions
        is replaced with one color the compressed video file is ~ 3 times smaller then using only PNG images.


        !!!It is important to have same -r and -g and -keyint_min values for both video files, the compressed
        and lossless one.

        This approach allows reducing the size while maintaining original quality in interesting parts.
    """

    def __init__(self, compressed_file, lossless_file, r=255, g=255, b=255):
        """
        :type compressed_file: str,
        :type lossless_file: str,
        :type r: int,   (this allows to set background color to be replaced).
        :type g: int,
        :type b: int
        """

        self.compressed_file_ = compressed_file
        self.lossless_file_ = lossless_file

        self.capture_compressed_ = None
        self.capture_lossless_ = None

        self.compressed_img_ = None
        self.lossless_img_ = None
        self.combined_img_ = None

        self.position_ = -1

        self.r_ = r
        self.g_ = g
        self.b_ = b

        self.capture_init_()

    def capture_init_(self):
        self.capture_compressed_ = cv2.VideoCapture(self.compressed_file_)
        self.capture_lossless_ = cv2.VideoCapture(self.lossless_file_)

        if not self.capture_compressed_.isOpened():
            raise Exception("Cannot open compressed video! Path: " + self.compressed_file_)

        if not self.capture_lossless_.isOpened():
            raise Exception("Cannot open lossless video! Path: " + self.lossless_file_)

    def move2_next(self):
        """
        returns next next combined image if exists, else raises exception
        """

        f, self.lossless_img_ = self.capture_lossless_.read()
        if not f:
            raise Exception("No more frames (" + str(self.position_) + ") in file: " + self.lossless_file_)

        f, self.compressed_img_ = self.capture_compressed_.read()
        if not f:
            raise Exception("No more frames (" + str(self.position_) + ") in file: " + self.lossless_file_)

        l = self.lossless_img_
        mask = (l[:, :, 0] != self.r_) & (l[:, :, 1] != self.g_) & (l[:, :, 2] != self.b_)

        self.combined_img_ = self.compressed_img_.copy()
        self.combined_img_[:, :, :][mask] = self.lossless_img_[:, :, :][mask]

        self.position_ += 1
        return self.combined_img_

    def seek_frame(self, frame_number):
        """
        :type frame_number: int
        returns sought frame if exists, else raises exception
        """

        if frame_number < 0 or frame_number >= self.total_frame_count():
            raise Exception("Frame_number is invalid")

        self.capture_compressed_.set(cv_compatibility.cv_CAP_PROP_POS_FRAMES, frame_number)
        self.capture_lossless_.set(cv_compatibility.cv_CAP_PROP_POS_FRAMES, frame_number)

        # because in move2_next it will be increased by one
        self.position_ = frame_number - 1

        return self.move2_next()

    def reset(self):
        self.position_ = -1
        self.combined_img_ = None
        self.compressed_img_ = None
        self.lossless_img_ = None

        self.capture_init_()

    def frame_number(self):
        return self.position_

    def img(self):
        return self.combined_img_

    def compressed_img(self):
        return self.compressed_img_

    def lossless_img(self):
        return self.lossless_img_

    def total_frame_count(self):
        return self.capture_compressed_.get(cv_compatibility.cv_CAP_PROP_FRAME_COUNT)

    def fps(self):
        return self.capture_compressed_.get(cv_compatibility.cv_CAP_PROP_FPS)

    def get_manager_copy(self):
        """
        returns copy of FerdaCompressedVideoManager, might be useful in cases of asynchronous operations (mainly seeking) on video
        while you want to maintain right position in original one.
        """
        vid = FerdaCompressedVideoManager(self.compressed_file_, self.lossless_file_, self.r_, self.g_, self.b_)
        vid.seek_frame(self.frame_number())

        return vid


if __name__ == "__main__":
    compressed = "/home/flipajs/segmentation/camera1_test3_c25_f.avi"
    lossless = "/home/flipajs/segmentation/out.avi"
    try:
        vid = FerdaCompressedVideoManager(compressed, lossless)
        # vid2 = FerdaCompressedVideoManager(compressed, lossless)
    except Exception as e:
        utils.misc.print_exception(e)

    i = 0
    while True:
        try:
            img = vid.next_frame()

            if i % 23 == 0 or i % 7 == 0:
                vid2 = FerdaCompressedVideoManager(compressed, lossless)
                img_seek = vid2.seek_frame(i)
                if sum(sum(sum(img_seek - img))) > 0:
                    print "problem"

            img_compressed = vid.compressed_img()
            img_lossless = vid.lossless_img()

            cv2.imshow("img", img)
            cv2.imshow("compressed", img_compressed)
            cv2.imshow("lossless", img_lossless)
            cv2.waitKey(1)
            i += 1

        except Exception as e:
            utils.misc.print_exception(e)
            break