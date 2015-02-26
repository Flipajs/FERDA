import math
from scipy import ndimage

__author__ = 'filip@naiser.cz'

import cv2
import numpy as np
from utils import video_manager
from skimage.transform import rescale
import my_utils
from region.mser import Mser
from region import mser_operations
import os
import utils.img
import pickle

im = None
scale_offset = (-1, -1)
clicked_pos = (-1, -1)
border = 100
radius = 1
ant_colors = None
init_regions = None
ant_id = 0
ant_number = 0
mser = None

collection_cols = 5
collection_rows = 2
collection_cell_size = 100

SQUARE_SIZE = 20


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


def ibg_transformation(im):
    ibg = np.asarray(im.copy(), dtype=np.double)

    # +1 for case of 0
    ibg[:, :, 0] = np.sum(ibg, axis=2) + 1

    ibg[:, :, 1] = ibg[:, :, 1] / (ibg[:, :, 0])
    ibg[:, :, 2] = ibg[:, :, 2] / (ibg[:, :, 0])

    return ibg


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
        colormark = get_colormark(im, ibg_norm, i_max, ant_colors[ant_id, :])
        init_regions[ant_id] = colormark
        crop = np.zeros((collection_cell_size, collection_cell_size, 3), dtype=np.uint8)
        crop = visualize(im, 0, crop, colormark)

        cv2.imshow('colormark', crop)

        print clicked_pos


def init(path):
    global ant_id
    global im
    global ant_colors
    global ant_number
    global mser
    global init_regions
    vid = video_manager.get_auto_video_manager(path)
    im = vid.move2_next()

    mser = Mser(max_area=0.0001, min_area=20, min_margin=2)

    cv2.cv.NamedWindow('img', cv2.cv.CV_WINDOW_AUTOSIZE)
    cv2.cv.SetMouseCallback('img', on_mouse, param=5)
    cv2.imshow('img', im)
    cv2.moveWindow('img', 0, 0)

    ant_number = int(raw_input('What is the number of ants? '))
    ant_colors = np.zeros((ant_number, 3), dtype=np.uint8)
    init_regions = [None for i in range(ant_number)]

    for i in range(ant_number):
        while True:
            k = cv2.waitKey(0)
            if k == 32:
                print('color for ant %d was selected', i)
                ant_id += 1
                break

            if k == 110:
                im = vid.random_frame()
                cv2.imshow('img', im)

    cv2.destroyWindow('color')
    cv2.destroyWindow('crop')


def color2ibg(color, i_max):
    color = np.array(color, dtype=np.float)
    s = np.sum(color)
    c = np.array([s, color[1] / s, color[2] / s])
    c[0] /= i_max

    return c


def normalized_ibg(im):
    ibg = ibg_transformation(im)
    i_max = np.max(ibg[:, :, 0]) + 1
    ibg[:, :, 0] /= i_max

    return ibg, i_max

def darkest_neighbour_square(im, pt, square_size):
    squares = []
    start = [pt[0]-square_size-(square_size/2), pt[1]-square_size-(square_size/2)]

    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                continue

            crop = utils.img.get_safe_selection(im, start[0]+square_size*i, start[1] + square_size*j, square_size, square_size, fill_color=(255, 255, 255))
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

    return position, squares[id]


def get_colormark(im, ibg_norm, i_max, c):
    global mser

    c_ibg = color2ibg(c, i_max)

    dist_im = np.linalg.norm(ibg_norm - c_ibg, axis=2)
    dist_im /= np.max(dist_im)
    dist_im = np.asarray(dist_im * 255, dtype=np.uint8)

    regions = mser.process_image(dist_im)
    groups = mser_operations.get_region_groups(regions)
    ids = mser_operations.margin_filter(regions, groups)
    regions = [regions[i] for i in ids]

    if len(regions) == 0:
        return None

    avg_intensity = [np.sum(dist_im[p.pts()[:, 0], p.pts()[:, 1]]) / p.area() for p in regions];
    # darkest_neighbour = [darkest_neighbour_square(im, r.centroid(), SQUARE_SIZE) for r in regions];

    # order = np.argsort(np.array(darkest_neighbour)[:,1])
    order = np.argsort(np.array(avg_intensity))
    ids = np.asarray(order[0:1], dtype=np.int32)

    selected_r = [regions[id] for id in ids]


    return selected_r[0]


def visualize(img, frame_i, collection, colormark, fill_color=np.array([255, 0, 255])):
    id_in_collection = frame_i % (collection_cols * collection_rows)
    c = np.asarray(colormark.centroid(), dtype=np.int32)
    cell_half = collection_cell_size / 2

    img = np.copy(img)
    pts = colormark.pts()
    img[pts[:,0], pts[:,1], :] = fill_color
    crop = utils.img.get_safe_selection(img, c[0] - cell_half, c[1] - cell_half, collection_cell_size,
                                        collection_cell_size)

    cv2.putText(crop, str(frame_i), (3, 10), cv2.FONT_HERSHEY_PLAIN, 0.65, (0, 0, 0), 1, cv2.CV_AA)

    # crop[pts[:, 0] - int(c[0]) + cell_half, pts[:, 1] - int(c[1]) + cell_half, :] = fill_color

    y = (id_in_collection / collection_cols) * collection_cell_size
    x = (id_in_collection % collection_cols) * collection_cell_size

    collection[y:y + collection_cell_size, x:x + collection_cell_size, :] = crop

    return collection


if __name__ == "__main__":
    output_folder = '/Users/fnaiser/Documents/colormarktests'
    video_file = '/Users/fnaiser/Documents/Camera 1_biglense2.avi'

    # Initialization from file
    with open(output_folder + '/settings.pkl', 'rb') as f:
        settings = pickle.load(f)

    ant_number = settings['ant_number']
    ant_number = 1
    ant_colors = settings['ant_colors']
    init_regions = settings['init_regions']
    mser = Mser(max_area=0.001, min_area=20, min_margin=2)

    # init(video_file)
    # settings = {'ant_number': ant_number, 'ant_colors': ant_colors, 'init_regions': init_regions}
    # with open(output_folder + '/settings.pkl', 'wb') as f:
    #     pickle.dump(settings, f)

    cimg = np.zeros((30, 30 * ant_colors.shape[0], 3), dtype=np.uint8)
    for i in range(ant_colors.shape[0]):
        cimg[:, 30 * i:30 * (i + 1), :] = ant_colors[i, :]

    cv2.imwrite(output_folder + '/colors.png', cimg)

    vid = video_manager.get_auto_video_manager(video_file)

    frame_i = 1

    regions = {}

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

    dir = output_folder + '/regions/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    c_w = collection_cols * collection_cell_size
    c_h = collection_rows * collection_cell_size
    collections = [np.zeros((c_h, c_w, 3), dtype=np.uint8) for i in range(ant_number)]

    while frame_i < 3000:
        print frame_i
        im = vid.move2_next()

        cv2.imwrite(output_folder + '/imgs/' + str(frame_i) + '.png', im)
        ibg_norm, i_max = normalized_ibg(im)

        id_in_collection = frame_i % (collection_cols * collection_rows)
        collection_id = frame_i / (collection_cols * collection_rows)
        frame_cmarks = []
        for ant_id in range(ant_number):
            h_ = 200
            w_ = 200
            y, x = init_regions[ant_id].centroid() - np.array([h_ / 2, w_ / 2])

            crop_im = utils.img.get_safe_selection(im, y, x, h_, w_)
            crop_ibg = utils.img.get_safe_selection(ibg_norm, y, x, h_, w_)

            c = get_colormark(crop_im, crop_ibg, i_max, ant_colors[ant_id, :])
            frame_cmarks.append(c)
            collections[ant_id] = visualize(im, frame_i, collections[ant_id], c)

        regions[frame_i] = np.array(frame_cmarks)

        if id_in_collection == 0 and frame_i > 0:
            for a_id in range(ant_number):
                cv2.imwrite(output_folder + '/collections/id' + str(a_id) + '/' + str(collection_id) + '.png',
                            collections[a_id])
                collections[a_id] = np.zeros((c_h, c_w, 3), dtype=np.uint8)

                with open(output_folder + '/regions/' + str(collection_id) + '.pkl', 'wb') as f:
                    pickle.dump(regions, f)

        # while True:
        # k = cv2.waitKey(5)
        # if k == 32:
        #         break

        frame_i += 1
        im = vid.move2_next()
        ibg = ibg_transformation(im)