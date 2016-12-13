from skimage.measure import moments_central, moments_hu, moments_normalized, moments
import cv2
from utils.img import get_img_around_pts, replace_everything_but_pts
import numpy as np
import math
from utils.img import rotate_img, centered_crop, get_bounding_box, endpoint_rot
from skimage.feature import local_binary_pattern
from core.id_detection.feature_manager import FeatureManager
from utils.gt.gt import GT
from utils.misc import print_progress


def get_mu_moments(img):
    m = moments(img)
    cr = m[0, 1] / m[0, 0]
    cc = m[1, 0] / m[0, 0]

    mu = moments_central(img, cr, cc)
    return mu

def get_nu_moments(img):
    mu = get_mu_moments(img)
    nu = moments_normalized(mu)

    return nu

def get_hu_moments(img):
    nu = get_nu_moments(img)
    hu = moments_hu(nu)

    features = [m_ for m_ in hu]

    return features

def __hu2str(vec):
    s = ''
    for i in vec:
        s += '\t{}\n'.format(i)

    return s

def get_basic_properties(r, p):
    f = []
    # area
    f.append(r.area())

    # contour length
    f.append(len(r.contour()))

    # major axis
    f.append(r.a_)

    # minor axis
    f.append(r.b_)

    # axis ratio
    f.append(r.a_ / r.b_)

    # axis ratio sqrt
    f.append((r.a_ / r.b_)**0.5)

    # axis ratio to power of 2
    f.append((r.a_ / r.b_)**2.0)

    img = p.img_manager.get_whole_img(r.frame_)
    crop, offset = get_img_around_pts(img, r.pts())

    pts_ = r.pts() - offset

    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGRA2GRAY)

    ###### MOMENTS #####
    # #### BINARY
    crop_b_mask = replace_everything_but_pts(np.ones(crop_gray.shape, dtype=np.uint8), pts_)
    f.extend(get_hu_moments(crop_b_mask))

    #### ONLY MSER PXs
    # in GRAY
    crop_gray_masked = replace_everything_but_pts(crop_gray, pts_)
    f.extend(get_hu_moments(crop_gray_masked))

    return f

def get_hog_features(r, p, fliplr=False):
    img = p.img_manager.get_whole_img(r.frame_)

    crop, offset = get_img_around_pts(img, r.pts(), margin=2.0)
    crop = rotate_img(crop, r.theta_)

    margin = 3

    crop = centered_crop(crop, 2 * (r.b_ + margin), 2 * (r.a_ + margin))

    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop_r = crop[:, :, 2]
    crop_g = crop[:, :, 1]
    crop_b = crop[:, :, 0]

    crops = [crop_gray, crop_r, crop_b, crop_g]

    if fliplr:
        f1 = __process_crops(crops, fliplr=False)
        f2 = __process_crops(crops, fliplr=True)

        return f1, f2
    else:
        f = __process_crops(crops, fliplr=False)

        return f

def __process_crops(crops, fliplr):
    from skimage.feature import hog

    f = []

    for crop in crops:
        if fliplr:
            crop = np.fliplr(crop)

        h, w = crop.shape

        fd = hog(crop, orientations=8, pixels_per_cell=(w, h),
                            cells_per_block=(1, 1), visualise=False)

        f.extend(fd)

        fd2 = hog(crop, orientations=8, pixels_per_cell=(w/4, h),
                            cells_per_block=(1, 1), visualise=False)

        f.extend(fd2)

    return f

def get_lbp_vect(crop):
    f = []

    h, w = crop.shape

    configs = [(24, 3), (8, 1)]
    for c in configs:
        lbp = local_binary_pattern(crop, c[0], c[1])
        lbp_ = lbp.copy()
        lbp_.shape = (lbp_.shape[0]*lbp_.shape[1], )
        h, _ = np.histogram(lbp_, bins=32, density=True)

        f += list(h)

        # subhists:
        num_parts = 3
        ls = np.linspace(0, w, num_parts+1, dtype=np.int32)
        for i in range(num_parts):
            lbp_ = lbp[:, ls[i]:ls[i+1]].copy()
            lbp_.shape = (lbp_.shape[0] * lbp_.shape[1],)

            h, _ = np.histogram(lbp_, bins=32, density=True)

            f += list(h)

    return f


def get_crop(r, p, margin=0):
    return __get_crop(r, p, margin=margin)

def __get_crop(r, p, margin=3):
    img = p.img_manager.get_whole_img(r.frame_)

    crop, offset = get_img_around_pts(img, r.pts(), margin=2.0)
    crop = rotate_img(crop, r.theta_)

    crop = centered_crop(crop, 2 * (r.b_ + margin), 2 * (r.a_ + margin))

    return crop

def get_colornames_hists(r, p, fliplr=False):
    import img_features

    crop = __get_crop(r, p)

    f2 = []

    f1 = img_features.colornames_descriptor(crop, pyramid_levels=3)
    if fliplr:
        f2 = img_features.colornames_descriptor(np.fliplr(crop), pyramid_levels=3)

    return f1, f2

def evaluate_features_performance(project, fm_names):
    gt = GT()
    gt.load(project.GT_file)

    single_region_ids = []
    single_region_ids, animal_ids = get_single_region_ids(project)
    print len(single_region_ids), len(animal_ids)

    if not isinstance(fm_names, list):
        fm_names = [fm_names]

    fms = []

    for fm_n in fm_names:
        fms.append(FeatureManager(project.working_directory, fm_n))

    import itertools

    seeds = [42, 1, 3, 30, 209]

    # Todo: guarantee min number per id class
    for test_size_ratio in [0.9, 0.95, 0.98, 0.99]:
        print "Training/Learning ratio: {}, #train: {}, #test: {}".format(test_size_ratio, int(len(animal_ids)*(1 - test_size_ratio)), int(len(animal_ids)*test_size_ratio))

        for num_f_types in range(1, len(fms)+1):
            for combination in itertools.combinations(fms, num_f_types):
                s = ""
                for fm in combination:
                    s += fm.db_path.split('/')[-1].split('.')[-2] + ' '

                print s

                X = []
                for r_id in single_region_ids:
                    f = []
                    for fm in combination:
                        _, f_ = fm[r_id]
                        f.extend(f_)

                    X.append(f[0])

                X = np.array(X)
                y = np.array(animal_ids)


                rf = RandomForestClassifier()

                from sklearn.model_selection import train_test_split

                a_num_correct = []
                a_num_test = []

                for i in range(5):
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size_ratio, random_state=seeds[i])

                    rf.fit(X_train, y_train)
                    num_correct = np.sum(rf.predict(X_test) == y_test)
                    num_test = len(y_test)

                    a_num_correct.append(num_correct)
                    a_num_test.append(num_test)


                print "Mean Correct: {}(std:{})/{}(std:{}) ({:.2%})".format(np.mean(a_num_correct),
                                                                            np.std(a_num_correct),
                                                                            np.mean(a_num_test),
                                                                            np.std(a_num_test),
                                                                            np.mean(a_num_correct)/np.mean(a_num_test))

                print

def get_idtracker_features(r, p):
    # import time

    max_d = 20
    max_i = 100
    max_c = 50
    img = p.img_manager.get_whole_img(r.frame_)
    crop, offset = get_img_around_pts(img, r.pts())

    # t1 = time.time()
    intensity_map_ = np.zeros((max_d, max_i + 1))
    contrast_map_ = np.zeros((max_d, max_c + 1))

    pts = r.pts() - offset


    ids1_ = []
    ids2_ = []

    n_p = len(pts)
    for i in range(n_p):
        ids1_.extend([i for _ in xrange(n_p - (i+1))])
        ids2_.extend(range(i+1, n_p))

    ids1_ = np.array(ids1_)
    ids2_ = np.array(ids2_)
    pts_ = np.array(pts)

    x1_ = pts_[ids1_, :]
    x2_ = pts_[ids2_, :]

    d_ = np.linalg.norm(x1_ - x2_, axis=1) -1

    x1_ = x1_[d_ < max_d]
    x2_ = x2_[d_ < max_d]
    # -1 because 0 never occurs
    d_ = d_[d_ < max_d]

    crop = np.asarray(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), dtype=np.int)

    x1_i = crop[x1_[:, 0], x1_[:, 1]]
    x2_i = crop[x2_[:, 0], x2_[:, 1]]

    i_ = x1_i + x2_i
    c_ = np.abs(x1_i - x2_i)

    for d in range(max_d):
        ids_ = np.logical_and(d_ <= d, (d-1) < d_)

        i__ = i_[ids_]
        if len(i__) == 0:
            continue

        for i in range(i__.min(), min(i__.max(), max_i)+1):
            intensity_map_[d, i] += np.sum(i__ == i)

        c__ = c_[ids_]
        for c in range(c__.min(), min(c__.max(), max_c)+1):
            contrast_map_[d, c] += np.sum(c__ == c)

    # print time.time() - t1

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(intensity_map_, aspect='auto')
    # plt.figure()
    # plt.imshow(contrast_map_, aspect='auto')

    return list(np.ravel(intensity_map_)), list(np.ravel(contrast_map_))
    # ########## slower variant
    #
    # t2 = time.time()
    # intensity_map = np.zeros((max_d, max_i + 1))
    # contrast_map = np.zeros((max_d, max_c + 1))
    #
    # for i, px1 in enumerate(pts):
    #     for px2 in pts[i+1:]:
    #         d = int(round(np.linalg.norm(px1-px2)) - 1)
    #
    #         if d >= max_d:
    #             continue
    #
    #         i1 = int(crop[px1[0], px1[1]])
    #         i2 = int(crop[px2[0], px2[1]])
    #
    #         i = min(max_i, i1 + i2)
    #         c = min(abs(i1 - i2), max_c)
    #         intensity_map[d, i] += 1
    #         contrast_map[d, c] += 1
    #
    # print time.time() - t2
    # print np.sum(intensity_map - intensity_map_), np.sum(contrast_map_ - contrast_map)
    #
    # return list(np.ravel(intensity_map)), list(np.ravel(contrast_map))

def get_single_region_ids(project):
    gt = GT()
    gt.load(project.GT_file)

    single_region_ids = []
    animal_ids = []
    match = gt.match_on_data(project, match_on='regions')

    for frame in match.iterkeys():
        for a_id, r_id in enumerate(match[frame]):
            if r_id is None:
                continue

            if match[frame].count(r_id) == 1:
                single_region_ids.append(r_id)
                animal_ids.append(a_id)

    return single_region_ids, animal_ids


if __name__ == '__main__':
    from core.project.project import Project
    import cPickle as pickle

    from sklearn.ensemble import RandomForestClassifier

    wd = '/Users/flipajs/Documents/wd/FERDA/Cam1_playground'
    p = Project()
    p.load(wd)

    from core.graph.chunk_manager import ChunkManager

    p.chm = ChunkManager()
    with open(wd + '/temp/isolation_score.pkl', 'rb') as f:
        up = pickle.Unpickler(f)
        p.gm.g = up.load()
        up.load()
        chm = up.load()
        p.chm = chm


    from core.region.region_manager import RegionManager

    p.rm = RegionManager(wd + '/temp', db_name='part0_rm.sqlite3')
    p.gm.rm = p.rm

    import matplotlib.pyplot as plt
    # for i in range(1, 7):
    #     r = p.rm[i]
    #     get_idtracker_features(r, p)

    # plt.show()


    # f1, f2 = get_idtracker_features(p.rm[1], p)
    # f1, f2 = get_idtracker_features(p.rm[2], p)
    # f1, f2 = get_idtracker_features(p.rm[3], p)
    # f1, f2 = get_idtracker_features(p.rm[4], p)
    # f1, f2 = get_idtracker_features(p.rm[5], p)
    # f1, f2 = get_idtracker_features(p.rm[6], p)
    # plt.show()

    if True:
        # p.chm.add_single_vertices_chunks(p, fra mes=range(4500))
        p.gm.update_nodes_in_t_refs()

        fm_names = ['fm_idtracker_i.sqlite3', 'fm_idtracker_i_d50.sqlite3', 'fm_idtracker_c.sqlite3', 'fm_basic.sqlite3', 'fm_colornames.sqlite3']

        if True:
            evaluate_features_performance(p, fm_names)

        if False:
            single_region_ids, _ = get_single_region_ids(p)
            # fm_basic = FeatureManager(p.working_directory, db_name='fm_basic.sqlite3')
            # fm_colornames = FeatureManager(p.working_directory, db_name='fm_colornames.sqlite3')
            fm_idtracker_i = FeatureManager(p.working_directory, db_name='fm_idtracker_i_d50.sqlite3')
            fm_idtracker_c = FeatureManager(p.working_directory, db_name='fm_idtracker_c_d50.sqlite3')

            # fms = [fm_basic, fm_colornames]
            # methods = [get_basic_properties, get_colornames_hists]

            # for r in p.rm[:]:
            #     for m, fm in zip(methods, fms):
            #         f = m(r, p)
            #         if len(f) == 2:
            #             f = f[0]
            #
            #         fm.add(r.id(), f)

            j = 0
            num_regions = len(single_region_ids)
            for r_id in single_region_ids:
                r = p.rm[r_id]
                j += 1

                f1, f2 = get_idtracker_features(r, p)

                fm_idtracker_i.add(r.id(), f1)
                fm_idtracker_c.add(r.id(), f2)

                print_progress(j, num_regions)
