# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
a = "/Users/flipajs/Pictures/vlcsnap-2017-02-02-14h03m57s501.png"
b = "/Users/flipajs/Pictures/vlcsnap-2017-01-30-15h57m09s596.png"
c = "/Users/flipajs/Pictures/vlcsnap-2016-09-05-10h36m15s229.png"

ap.add_argument("-i", "--image", required=False, help="Path to the image", default=a)
args = vars(ap.parse_args())

# load the image and convert it to a floating point data type
image = img_as_float(io.imread(args["image"]))

# loop over the number of segments
for numSegments in [40000]:
    # apply SLIC and extract (approximately) the supplied number
    # of segments
    # segments = slic(image, n_segments=numSegments, sigma=0, compactness=0.1)
    segments = slic(image, n_segments=numSegments, sigma=0.1, compactness=10)

    # show the output of SLIC
    fig = plt.figure("Superpixels -- %d segments" % (numSegments))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(image, segments))
    plt.axis("off")

# show the plots
plt.show()
