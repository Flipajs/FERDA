__author__ = 'filip@naiser.cz'

"""
How to use this tool:

set output_folder and video_file to proper value

During initialization it is important, to select all ants on first frame even if there are not good
colormarks. You can go back to previous colormark assignment by pressinb 'b' key.
So if the situation is like this. 4 colormarks on first frame are "perfect" and 2 are blured or invisible
I select first good ant, pressing spacebar after that and continue to 2nd 3rd and 4th perfect colormark.
Then pressing spacebar and I select body color of first unperfect ant in first frame pressing spacebar
I select the second body color of unperfect ant. Then I press b and after thet n till I see frame with
good colormark...

This is important because I need to know the init position of each ant and if I have colormark from
random frame, it is likely that this position will be far away from starting position at first frame
and I will not find colormark there.

CONTROLS:
spacebar - next assignment
b - previous assignment
n - next random image

"""

import os
import pickle

import cv2
import numpy as np
from skimage.transform import rescale

from utils import video_manager
from core.region.mser import Mser
from core import region
from core.region import mser_operations
import utils.img
import matplotlib.mlab as mlab


im = None
scale_offset = (-1, -1)
clicked_pos = (-1, -1)
border = 100
radius = 1
ant_colors = None
init_regions = None
init_scores = None
init_positions = None
init_means = None
ant_id = 0
ant_number = 0
mser = None

collection_cols = 5
collection_rows = 2
collection_cell_size = 100

NEIGH_SQUARE_SIZE = 10
FAST_START = False
CROP_SIZE = 200
MSER_MAX_SIZE = 200
MSER_MIN_SIZE = 5
MSER_MIN_MARGIN = 5

# COLORMARK size statistics
COLORMARK_RADIUS = 3.7
COLORMARK_S = 2.2

# AVG colormark intensity in distance map
INTENSITY_MEAN = 20
INTENSITY_S = 10

# AVG min neihbour intensity in image
NEIGH_MEAN = 40
NEIHG_S = 15

# W_AREA = 0.5
# W_INT = 1.0
# W_NEIGH_INT = 1.5

# COLORMARK_AREA_SQROOT = (3.14*COLORMARK_RADIUS**2)**0.5

#255*3 + 1
#*3 is normalazing I to [0..1/3] as the GBR components are
#*2 is lowering the weight of intensity to half
I_NORM = 766 * 3 * 2

output_folder = '/Users/fnaiser/Documents/colormarktests'
video_file = '/Users/fnaiser/Documents/Camera 1_biglense2.avi'


def get_color_around(im, pos, radius):
    c = np.zeros((1, 3), dtype=np.double)
    num_px = 0
    for h in range(radius * 2 + 1):
        for w in range(radius * 2 + 1):
            d = ((w - radius) ** 2 + (h - radius) ** 2) ** 0.5
            if d <= radius:
                num_px += 1
                c += im[pos[1] - radius + w, pos[0] - radius + h, :]

    print num_px
    c /= num_px

    return [c[0, 0], c[0, 1], c[0, 2]]


def igr_transformation(im):
    """
    this function takes image in BRG format and returns image in IGR space
    [R+G+B, G / (R + G + B), R / (R + G + B)]
    """

    # channels from cv2 -> numpy array are in order BGR
    igr = np.asarray(im.copy(), dtype=np.double)
    # +1 to avoid dividing by zero in future
    igr[:, :, 0] = np.sum(igr, axis=2) + 1

    igr[:, :, 1] = igr[:, :, 1] / (igr[:, :, 0])
    igr[:, :, 2] = igr[:, :, 2] / (igr[:, :, 0])
    igr[:, :, 0] = im[:, :, 0] / (igr[:, :, 0])

    return igr

def igbr_transformation(im):
    igbr = np.zeros((im.shape[0], im.shape[1], 4), dtype=np.double)

    igbr[:,:,0] = np.sum(im,axis=2) + 1
    igbr[:, :, 1] = im[:,:,0] / igbr[:,:,0]
    igbr[:,:,2] = im[:,:,1] / igbr[:,:,0]
    igbr[:,:,3] = im[:,:,2] / igbr[:,:,0]

    igbr[:,:,0] = igbr[:,:,0] / I_NORM

    return igbr


def on_mouse(event, x, y, flag, param):
    global clicked_pos
    global im
    global border
    global scale_offset

    if event == cv2.EVENT_LBUTTONDOWN:
        crop = im[max(0, y - border):min(im.shape[0], y + border), max(0, x - border):min(im.shape[1], x + border), :]
        rescaled = rescale(crop, 2.0)

        scale_offset = (max(0, x - border), max(0, y - border))
        cv2.cv.NamedWindow('crop', cv2.cv.CV_WINDOW_AUTOSIZE)
        cv2.cv.SetMouseCallback('crop', on_mouse_scaled, param=5)
        cv2.imshow('crop', rescaled)


def on_mouse_scaled(event, x, y, flag, param):
    global clicked_pos
    global scale_offset
    global border
    global radius
    global im
    global ant_id
    global init_positions

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_pos = (scale_offset[0] + (x / 2), scale_offset[1] + (y / 2));
        c2 = get_color_around(im, clicked_pos, radius)
        c2 = np.asarray(c2, dtype=np.uint8)
        print c2

        ant_colors[ant_id, :] = c2

        cimg = np.zeros((30, 30 * ant_colors.shape[0], 3), dtype=np.uint8)
        for i in range(ant_colors.shape[0]):
            cimg[:, 30 * i:30 * (i + 1), :] = ant_colors[i, :]

        cv2.imshow('color', cimg)


        ibg_norm, i_max = normalized_ibg(im)

        h_ = CROP_SIZE
        w_ = CROP_SIZE
        x, y = np.array(clicked_pos) - np.array([CROP_SIZE / 2, CROP_SIZE / 2])

        crop_im = utils.img.get_safe_selection(im, y, x, h_, w_)
        crop_ibg = utils.img.get_safe_selection(ibg_norm, y, x, h_, w_)

        colormark, cmark_i, neigh_i, d_map, _ = get_colormark(crop_im, crop_ibg, i_max, ant_colors[ant_id, :])

        cv2.imshow('dmap', d_map)
        if not colormark:
            print "No MSER found"
            return

        init_regions[ant_id] = colormark
        init_scores[ant_id] = [cmark_i, neigh_i]

        r = (colormark.area() / 3.14)**0.5
        avg_i = np.sum(d_map[colormark.pts()[:, 0], colormark.pts()[:, 1]]) / float(colormark.area())
        min_neigh = darkest_neighbour_square(crop_im, colormark.centroid(), NEIGH_SQUARE_SIZE)

        init_means[ant_id] = {'intensity': avg_i, 'neigh': min_neigh, 'radius': r}
        crop = np.zeros((collection_cell_size, collection_cell_size, 3), dtype=np.uint8)
        crop = visualize(crop_im, 0, crop, colormark)

        colormark.set_centroid(colormark.centroid() + np.array([y, x]))
        if init_positions[ant_id] is None:
            init_positions[ant_id] = colormark.centroid()
            print 'Position set: ', init_positions[ant_id]

        cv2.imshow('colormark', crop)

        print clicked_pos


def init(path):
    global ant_id
    global im
    global ant_colors
    global ant_number
    global mser
    global init_regions
    global init_scores
    global init_positions
    global init_means

    vid = video_manager.get_auto_video_manager(path)
    im = vid.next_frame()

    mser = Mser(max_area=0.1, min_area=MSER_MIN_SIZE, min_margin=MSER_MIN_MARGIN)

    cv2.cv.NamedWindow('img', cv2.cv.CV_WINDOW_AUTOSIZE)
    cv2.cv.SetMouseCallback('img', on_mouse, param=5)
    cv2.imshow('img', im)
    cv2.moveWindow('img', 0, 0)

    ant_number = int(raw_input('What is the number of ants? '))
    ant_colors = np.zeros((ant_number, 3), dtype=np.uint8)
    init_regions = [None for i in range(ant_number)]
    init_scores = [None for i in range(ant_number)]
    init_positions = [None for i in range(ant_number)]
    init_means = [None for i in range(ant_number)]

    while ant_id < ant_number:
        # this will take last 8 bits from integer so it is number between 0 - 255
        k = cv2.waitKey(0) & 255
        if k == 32:
            print 'color for ant ' + str(ant_id) + ' was selected'
            ant_id += 1

        # r, R - random frame
        if k == 114 or k == 82:
            im = vid.random_frame()
            cv2.imshow('img', im)

        # b, B - go back in assignment
        if k == 98 or k == 66:
            if ant_id > 0:
                ant_id -= 1
                print 'moving backward in assignment, Ant id: ' + str(ant_id)

    cv2.destroyWindow('color')
    cv2.destroyWindow('crop')


def color2irg(color, i_max):
    # color comes in BGR format
    color = np.array(color, dtype=np.float)
    s = np.sum(color) + 1
    c = np.array([color[0] / s, color[1] / s, color[2] / s])
    c[0] /= i_max

    c = np.array([s/I_NORM, color[0]/s, color[1]/s, color[2]/s])

    return c


def normalized_ibg(im):
    # ibg = igr_transformation(im)
    # i_max = np.max(ibg[:, :, 0]) + 1
    # ibg[:, :, 0] /= i_max
    #
    # return ibg, i_max


    igbr = igbr_transformation(im)
    return igbr, 1


def darkest_neighbour_square(im, pt, square_size):
    squares = []
    start = [pt[0] - square_size - (square_size / 2), pt[1] - square_size - (square_size / 2)]

    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                continue

            crop = utils.img.get_safe_selection(im, start[0] + square_size * i, start[1] + square_size * j, square_size,
                                                square_size, fill_color=(255, 255, 255))

            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            s = np.sum(crop_gray)

            squares.append(s)

    id = np.argmin(squares)

    position = ''
    if id == 0:
        position = 'top-left'
    elif id == 1:
        position = 'top'
    elif id == 2:
        position = 'top-right'
    elif id == 3:
        position = 'left'
    elif id == 4:
        position = 'right'
    elif id == 5:
        position = 'bottom-left'
    elif id == 6:
        position = 'bottom'
    elif id == 7:
        position = 'bottom-right'

    # return position, squares[id]
    return squares[id] / square_size ** 2


def get_colormark(im, ibg_norm, i_max, c, ant_id = -1):
    global mser
    global init_means

    c_ibg = color2irg(c, i_max)

    dist_im = np.linalg.norm(ibg_norm - c_ibg, axis=2)
    dist_im /= np.max(dist_im)
    dist_im = np.asarray(dist_im * 255, dtype=np.uint8)

    mser.set_max_area_relative(MSER_MAX_SIZE / float(im.shape[0] * im.shape[1]))
    regions = mser.process_image(dist_im)
    groups = mser_operations.get_region_groups(regions)
    ids = mser_operations.margin_filter(regions, groups)
    regions = [regions[i] for i in ids]


    if len(regions) == 0:
        return None, -1, -1, dist_im, None

    # areas_ = [p.area() for p in regions]

    # areas = [(COLORMARK_AREA / float(p.area())) if p.area() > COLORMARK_AREA else (COLORMARK_AREA / float(p.area())) for p in regions]
    # areas = [(COLORMARK_AREA_SQROOT / float(p.area())**0.5) if p.area()**0.5 > COLORMARK_AREA_SQROOT else (float(p.area())**0.5 / COLORMARK_AREA_SQROOT) for p in regions]
    areas = [(float(p.area()) / 3.14)**0.5 for p in regions]
    # areas = [a if a < 1.0 else 0 for a in areas]

    # print areas_, areas
    avg_intensity = [np.sum(dist_im[p.pts()[:, 0], p.pts()[:, 1]]) / p.area() for p in regions];
    darkest_neighbour = [darkest_neighbour_square(im, r.centroid(), NEIGH_SQUARE_SIZE) for r in regions];

    # val = np.array(avg_intensity) + np.array(darkest_neighbour)

    i_mean = INTENSITY_MEAN
    r_mean = COLORMARK_RADIUS
    n_mean = NEIGH_MEAN
    if ant_id > -1:
        i_mean = init_means[ant_id]['intensity']
        r_mean = init_means[ant_id]['radius']
        n_mean = init_means[ant_id]['neigh']



    norm_i_ = mlab.normpdf(i_mean, i_mean, INTENSITY_S)
    norm_neigh_ = mlab.normpdf(n_mean, n_mean, NEIHG_S)

    norm_area_ = mlab.normpdf(r_mean, r_mean, COLORMARK_S)

    val_i = mlab.normpdf(np.array(avg_intensity), i_mean, INTENSITY_S) / norm_i_
    val_n = mlab.normpdf(np.array(darkest_neighbour), n_mean, NEIHG_S) / norm_neigh_
    val_a = mlab.normpdf(np.array(areas), r_mean, COLORMARK_S) / norm_area_

    val = val_i * val_n * val_a

    print areas
    print mlab.normpdf(np.array(areas), r_mean, COLORMARK_S) / norm_area_
    print val
    # val = np.array(areas)
    order = np.argsort(-val)


    print "REGIONS", len(regions)
    cols = collection_cols
    rows = len(regions) / cols + 1
    msers = np.zeros((collection_cell_size*rows, collection_cell_size*cols, 3), dtype=np.uint8)
    i = 0
    for r in regions:
        msers = visualize2(im, i, msers, r, rows, areas[i], val_a[i], avg_intensity[i], val_i[i], darkest_neighbour[i], val_n[i])
        i+=1

    # dump_i = 0
    # for id in order:
    # if dump_i == 5:
    # break
    #
    #     r = regions[id]
    #
    #     dump_i += 1
    #     c = r.centroid()
    #     cell_half = 50
    #     img = np.copy(im)
    #     pts = r.pts()
    #     img[pts[:,0], pts[:,1], :] = (255, 255, 255)
    #     crop = utils.img.get_safe_selection(img, c[0] - cell_half, c[1] - cell_half, collection_cell_size,
    #                                         collection_cell_size)
    #     ds = np.sum(dist_im[r.pts()[:, 0], r.pts()[:, 1]]) / r.area()
    #     dn = darkest_neighbour_square(im, r.centroid(), NEIGH_SQUARE_SIZE)
    #     print ds, dn, ds+dn
    #     cv2.imshow('crop', crop)
    #     cv2.waitKey(0)

    # order = np.argsort(np.array(avg_intensity))
    ids = np.asarray(order[0:1], dtype=np.int32)

    selected_r = [regions[id] for id in ids]
    print avg_intensity[ids[0]], darkest_neighbour[ids[0]]




    return selected_r[0], avg_intensity[ids[0]], darkest_neighbour[ids[0]], dist_im, msers


def visualize(img, frame_i, collection, colormark, fill_color=np.array([255, 0, 255])):
    id_in_collection = frame_i % (collection_cols * collection_rows)
    c = np.asarray(colormark.centroid(), dtype=np.int32)
    cell_half = collection_cell_size / 2

    img = np.copy(img)
    pts = colormark.pts()
    img[pts[:, 0], pts[:, 1], :] = fill_color
    crop = utils.img.get_safe_selection(img, c[0] - cell_half, c[1] - cell_half, collection_cell_size,
                                        collection_cell_size)

    cv2.putText(crop, str(frame_i), (3, 10), cv2.FONT_HERSHEY_PLAIN, 0.65, (0, 0, 0), 1, cv2.CV_AA)

    # crop[pts[:, 0] - int(c[0]) + cell_half, pts[:, 1] - int(c[1]) + cell_half, :] = fill_color

    y = (id_in_collection / collection_cols) * collection_cell_size
    x = (id_in_collection % collection_cols) * collection_cell_size

    collection[y:y + collection_cell_size, x:x + collection_cell_size, :] = crop

    return collection


def visualize2(img, frame_i, collection, colormark, collection_rows, a, va, i, vi, n, vn, fill_color=np.array([255, 0, 255])):
    id_in_collection = frame_i % (collection_cols * collection_rows)
    c = np.asarray(colormark.centroid(), dtype=np.int32)
    cell_half = collection_cell_size / 2

    img = np.copy(img)
    pts = colormark.pts()
    img[pts[:, 0], pts[:, 1], :] = fill_color
    crop = utils.img.get_safe_selection(img, c[0] - cell_half, c[1] - cell_half, collection_cell_size,
                                        collection_cell_size)
    if va < 0.01:
        va=0

    if vn < 0.01:
        vn = 0

    if vi < 0.01:
        vi = 0

    vstep = 10
    cv2.putText(crop, str(frame_i), (3, 10), cv2.FONT_HERSHEY_PLAIN, 0.65, (0, 0, 0), 1, cv2.CV_AA)
    cv2.putText(crop, str(a)[0:5]+' '+str(va)[0:5], (3, 10+vstep), cv2.FONT_HERSHEY_PLAIN, 0.65, (125, 255, 0), 1, cv2.CV_AA)
    cv2.putText(crop, str(i)[0:5]+' '+str(vi)[0:5], (3, 10+2*vstep), cv2.FONT_HERSHEY_PLAIN, 0.65, (125, 255, 0), 1, cv2.CV_AA)
    cv2.putText(crop, str(n)[0:5]+' '+str(vn)[0:5], (3, 10+3*vstep), cv2.FONT_HERSHEY_PLAIN, 0.65, (125, 255, 0), 1, cv2.CV_AA)
    cv2.putText(crop, str(va*vi*vn)[0:5], (3, 10+4*vstep), cv2.FONT_HERSHEY_PLAIN, 0.65, (125, 255, 0), 1, cv2.CV_AA)
    # crop[pts[:, 0] - int(c[0]) + cell_half, pts[:, 1] - int(c[1]) + cell_half, :] = fill_color

    y = (id_in_collection / collection_cols) * collection_cell_size
    x = (id_in_collection % collection_cols) * collection_cell_size

    collection[y:y + collection_cell_size, x:x + collection_cell_size, :] = crop

    return collection


if __name__ == "__main__":
    if FAST_START:
        # Initialization from file
        with open(output_folder + '/settings.pkl', 'rb') as f:
            settings = pickle.load(f)

        ant_number = settings['ant_number']
        ant_colors = settings['ant_colors']
        init_regions = settings['init_regions']
        init_scores = settings['init_scores']
        init_positions = settings['init_positions']
        init_means = settings['init_means']
        mser = Mser(max_area=0.1, min_area=MSER_MIN_SIZE, min_margin=MSER_MIN_MARGIN)
    else:
        init(video_file)
        settings = {'ant_number': ant_number, 'ant_colors': ant_colors, 'init_regions': init_regions,
                    'neigh_square_size': NEIGH_SQUARE_SIZE, 'init_scores': init_scores,
                    'init_positions': init_positions, 'init_means': init_means}
        with open(output_folder + '/settings.pkl', 'wb') as f:
            pickle.dump(settings, f)

    cimg = np.zeros((30, 30 * ant_colors.shape[0], 3), dtype=np.uint8)
    for i in range(ant_colors.shape[0]):
        cimg[:, 30 * i:30 * (i + 1), :] = ant_colors[i, :]

    cv2.imwrite(output_folder + '/colors.png', cimg)

    vid = video_manager.get_auto_video_manager(video_file)

    frame_i = 0

    regions = {}
    scores = {}

    dir = output_folder + '/imgs'
    if not os.path.exists(dir):
        os.makedirs(dir)

    dir = output_folder + '/collections'
    if not os.path.exists(dir):
        os.makedirs(dir)

    for a_id in range(ant_number):
        dir = output_folder + '/collections/id' + str(a_id)
        if not os.path.exists(dir):
            os.makedirs(dir)

    for a_id in range(ant_number):
        dir = output_folder + '/msers/id' + str(a_id)
        if not os.path.exists(dir):
            os.makedirs(dir)

    for a_id in range(ant_number):
        dir = output_folder + '/dmap/id' + str(a_id)
        if not os.path.exists(dir):
            os.makedirs(dir)

    dir = output_folder + '/regions/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    dir = output_folder + '/scores/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    c_w = collection_cols * collection_cell_size
    c_h = collection_rows * collection_cell_size
    collections = [np.zeros((c_h, c_w, 3), dtype=np.uint8) for i in range(ant_number)]

    previous_position = init_positions
    while frame_i < 3000:
        print frame_i
        im = vid.next_frame()

        cv2.imwrite(output_folder + '/imgs/' + str(frame_i) + '.png', im)

        ibg_norm, i_max = normalized_ibg(im)
        i_max = 1

        id_in_collection = frame_i % (collection_cols * collection_rows)
        collection_id = frame_i / (collection_cols * collection_rows)

        if id_in_collection == 0 and frame_i > 0:
            for a_id in range(ant_number):
                cv2.imwrite(output_folder + '/collections/id' + str(a_id) + '/' + str(collection_id) + '.png',
                            collections[a_id])
                collections[a_id] = np.zeros((c_h, c_w, 3), dtype=np.uint8)

                with open(output_folder + '/regions/' + str(collection_id) + '.pkl', 'wb') as f:
                    pickle.dump(regions, f)

                with open(output_folder + '/scores/' + str(collection_id) + '.pkl', 'wb') as f:
                    pickle.dump(scores, f)

        frame_cmarks = []
        frame_scores = []
        for ant_id in range(ant_number):
            h_ = CROP_SIZE
            w_ = CROP_SIZE
            y, x = previous_position[ant_id] - np.array([h_ / 2, w_ / 2])

            crop_im = utils.img.get_safe_selection(im, y, x, h_, w_)

            crop_ibg = utils.img.get_safe_selection(ibg_norm, y, x, h_, w_)

            c, cmark_i, neigh_i, dist_map, msers = get_colormark(crop_im, crop_ibg, i_max, ant_colors[ant_id, :], ant_id)
            if c:
                collections[ant_id] = visualize(crop_im, frame_i, collections[ant_id], c)
                c.set_centroid(c.centroid() + np.array([y, x]))
                previous_position[ant_id] = c.centroid()
                cv2.imwrite(output_folder + '/msers/id' + str(ant_id) + '/' + str(frame_i) + '.png', msers)

            frame_cmarks.append(c)
            frame_scores.append([cmark_i, neigh_i])
            cv2.imwrite(output_folder + '/dmap/id' + str(ant_id) + '/' + str(frame_i) + '.png', dist_map)


        regions[frame_i] = np.array(frame_cmarks)
        scores[frame_i] = np.array(frame_scores)


        # while True:
        # k = cv2.waitKey(5)
        # if k == 32:
        # break

        frame_i += 1
        im = vid.next_frame()
