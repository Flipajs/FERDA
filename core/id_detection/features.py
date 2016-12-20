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
from itertools import izip
# import pyximport; pyximport.install()
# import features2
import matplotlib.pyplot as plt
import time
from math import ceil


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

def get_hog_features_fliplr(r, p):
    f1, f2 = get_hog_features(r, p, True)

    f1.extend(f2)
    return f1

def get_hog_features(r, p, fliplr=False):
    img = p.img_manager.get_whole_img(r.frame_)

    crop, offset = get_img_around_pts(img, r.pts(), margin=2.0)
    crop = rotate_img(crop, r.theta_)

    margin = 3

    crop = centered_crop(crop, 2 * (r.b_ + margin), 2 * (r.a_ + margin))

    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    crops = [crop_gray]

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

def get_lbp(r, p):
    img = p.img_manager.get_whole_img(r.frame_)

    crop, offset = get_img_around_pts(img, r.pts(), margin=2.0)
    crop = rotate_img(crop, r.theta_)

    margin = 3

    crop = centered_crop(crop, 2 * (r.b_ + margin), 2 * (r.a_ + margin))

    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    crops = [crop_gray]
    f = __process_crops(crops, fliplr=False)

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

    return f1

def evaluate_features_performance(project, fm_names, seed=None, train_n_times=10, test_split_method='random',
                                  verbose=1, rf_class_weight=None, rf_criterion='gini'):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    gt = GT()
    gt.load(project.GT_file)

    single_region_ids, animal_ids = get_single_region_ids(project)
    if verbose:
        np.set_printoptions(precision=4)
        print len(single_region_ids), len(animal_ids)

    if not isinstance(fm_names, list):
        fm_names = [fm_names]

    fms = []

    for fm_n in fm_names:
        fms.append(FeatureManager(project.working_directory, fm_n))

    import itertools

    if seed is not None:
        np.random.seed(seed)

    seeds = np.random.randint(0, 100000, train_n_times)

    results = {'layer': 'test_size_ratio'}

    # Todo: guarantee min number per id class
    for test_size_ratio in [0.8, 0.9, 0.95, 0.99]:
        results[test_size_ratio] = {'layer': 'features'}

        if verbose:
            print
            print
            print "#########################################################"
            print "Training/Learning ratio: {}, #train: {}, #test: {}".format(test_size_ratio, int(len(animal_ids)*(1 - test_size_ratio)), int(len(animal_ids)*test_size_ratio))

        # for num_f_types in range(1, len(fms)+1):
        for num_f_types in range(1, 2):
            for combination in itertools.combinations(fms, num_f_types):
                fliplr = False

                s = ""
                for fm in combination:
                    s += fm.db_path.split('/')[-1].split('.')[-2] + ' '

                # TODO: for combinations
                if s[-6:] == 'fliplr':
                    fliplr = True

                if verbose:
                    print
                    print "##### ", s , " #####"

                results[test_size_ratio][s] = {}

                X = []
                for r_id in single_region_ids:
                    f = []
                    for fm in combination:
                        _, f_ = fm[r_id]

                        if f_[0] is None:
                            print r_id

                        f.extend(f_[0])

                    X.append(f)

                num_animals = len(set(animal_ids))

                X = np.array(X)
                y = np.array(animal_ids)

                print X.shape, y.shape

                results[test_size_ratio][s]['X_shape'] = X.shape
                results[test_size_ratio][s]['class_frequency'] = []
                results[test_size_ratio][s]['train_class_frequency'] = []

                for ai in range(num_animals):
                    results[test_size_ratio][s]['class_frequency'].append(np.sum(y == ai))

                rf = RandomForestClassifier(class_weight=rf_class_weight, criterion=rf_criterion)

                results[test_size_ratio][s]['num_correct'] = []
                results[test_size_ratio][s]['accuracy'] = []
                results[test_size_ratio][s]['class_accuracy'] = []

                for i in range(train_n_times):
                    if test_split_method == 'random':
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size_ratio, random_state=seeds[i])
                    elif test_split_method == 'equivalent_class_num':
                        # TODO:...
                        raise Exception('Not implemented yet')
                    else:
                        split_ = int(X.shape[0]*(1 - test_size_ratio))
                        X_train, X_test = X[:split_, :], X[split_:, :]
                        y_train, y_test = y[:split_], y[split_:]

                    # concatenated vector of features for left and
                    if fliplr:
                        X_train_new = []
                        y_train_new = []
                        for line, id_ in izip(X_train, y_train):
                            X_train_new.append(line[:len(line)/2])
                            X_train_new.append(line[len(line)/2:])
                            y_train_new.append(id_)
                            y_train_new.append(id_)

                        X_test_new = []
                        for line in izip(X_test):
                            X_test_new.append(line[:len(line)/2])

                    rf.fit(X_train, y_train)

                    # arr = rf.feature_importances_
                    # indices = np.where(arr > 0.000001)[0]
                    # print indices.shape
                    # arr.shape = (50, 81)
                    # y, x = np.where(arr > 0.000001)
                    #
                    # print 0.01, np.sum(arr > 0.01)
                    # print 0.001, np.sum(arr > 0.001)
                    # print 0.0001, np.sum(arr > 0.0001)
                    # print 0.00001, np.sum(arr > 0.00001)
                    # print 0.000001, np.sum(arr > 0.000001)
                    # print 0.0000001, np.sum(arr > 0.0000001)
                    #
                    # with open(p.working_directory+'/temp/'+s+'_indices.pkl', 'wb') as f:
                    #     pickle.dump((indices, y, x, rf.feature_importances_), f)
                    #
                    # plt.scatter(x, y)
                    # plt.figure()
                    # plt.imshow(arr)
                    # plt.show()

                    correct_ids = rf.predict(X_test) == y_test
                    num_correct = np.sum(correct_ids)
                    num_test = len(y_test)

                    class_accuracy = []
                    train_class_frequency = []
                    for ic in range(num_animals):
                        num_c = np.sum(y_test == ic)
                        train_num_c = np.sum(y_train == ic)
                        train_class_frequency.append(train_num_c)
                        correct_c = np.sum(np.logical_and(correct_ids, y_test == ic))
                        class_accuracy.append(correct_c / float(num_c))

                    results[test_size_ratio][s]['num_correct'].append(num_correct)
                    results[test_size_ratio][s]['accuracy'].append(num_correct / float(num_test))
                    results[test_size_ratio][s]['class_accuracy'].append(class_accuracy)
                    results[test_size_ratio][s]['train_class_frequency'].append(train_class_frequency)

                if verbose:
                    num_test = int(test_size_ratio*X.shape[0])
                    print "Mean Correct: {}(std:{})/{} ({:.2%}, std: {})".format(
                        np.mean(results[test_size_ratio][s]['num_correct']),
                        np.std(results[test_size_ratio][s]['num_correct']),
                        num_test,
                        np.mean(results[test_size_ratio][s]['accuracy']),
                        np.std(results[test_size_ratio][s]['accuracy'])
                    )

                    print "class frequency", results[test_size_ratio][s]['class_frequency']
                    print "train class frequency, mean: ", np.mean(results[test_size_ratio][s]['train_class_frequency'], axis=0), "std: ", np.std(results[test_size_ratio][s]['train_class_frequency'], axis=0)
                    print "class accuracy mean", np.mean(results[test_size_ratio][s]['class_accuracy'], axis=0), "std: ", np.std(results[test_size_ratio][s]['class_accuracy'], axis=0)

    # reset...
    np.set_printoptions()

    return results

def get_idtracker_features_sub2(r, p, debug=False):
    return get_idtracker_features(r, p, debug, sub=2)

def get_idtracker_features_sub4(r, p, debug=False):
    return get_idtracker_features(r, p, debug, sub=4)

def get_idtracker_features_sub8(r, p, debug=False):
    return get_idtracker_features(r, p, debug, sub=8)

def get_idtracker_features(r, p, debug=False, sub=1):
    # TODO: features_importance speedup...
    # TODO:


    # import time

    max_d = 50
    min_i = 20
    # max_i = 100

    # Cam1 settings
    max_i = 100
    max_c = 50

    # # zebrafish settings
    # min_i = 0
    # max_i = 210
    # max_c = 50


    # # Camera3 Settings
    # max_d = 70
    #
    # min_i = 20
    # max_i = 90
    # max_c = 40


    img = p.img_manager.get_whole_img(r.frame_)
    crop, offset = get_img_around_pts(img, r.pts())
    crop = np.asarray(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), dtype=np.int)

    # t1 = time.time()
    intensity_map_ = np.zeros((max_d, max_i + 1 - min_i), dtype=np.int)
    contrast_map_ = np.zeros((max_d, max_c + 1), dtype=np.int)

    pts = r.pts() - offset

    ids1_ = []
    ids2_ = []

    n_p = len(pts)
    for i in xrange(0, n_p):
        # ids1_.extend([i for _ in xrange(0, n_p - (i+1), sub)])
        # ids2_.extend(range(i + 1, n_p, sub))
        r = xrange(i + 1, n_p, sub)
        ids2_.extend(r)
        ids1_.extend([i] * len(r))

    ids1_ = np.array(ids1_)
    ids2_ = np.array(ids2_)

    # print ids1_.shape, ids2_.shape

    pts_ = np.array(pts)

    x1_ = pts_[ids1_, :]
    x2_ = pts_[ids2_, :]

    d_ = np.linalg.norm(x1_ - x2_, axis=1) -1

    x1_ = x1_[d_ < max_d]
    x2_ = x2_[d_ < max_d]
    # -1 because 0 never occurs
    d_ = d_[d_ < max_d]

    x1_i = crop[x1_[:, 0], x1_[:, 1]]
    x2_i = crop[x2_[:, 0], x2_[:, 1]]

    i_ = x1_i + x2_i
    c_ = np.abs(x1_i - x2_i)

    for d in range(max_d):
        ids_ = np.logical_and(d_ <= d, (d-1) < d_)

        i__ = i_[ids_]
        if len(i__) == 0:
            continue

        for i in range(max(i__.min(), min_i), min(i__.max(), max_i)+1):
            # intensity_map_[d, i - min_i] += np.sum(i__ == i)
            intensity_map_[d, i - min_i] += np.count_nonzero(i__ == i)

        c__ = c_[ids_]
        for c in range(c__.min(), min(c__.max(), max_c)+1):
            # contrast_map_[d, c] += np.sum(c__ == c)
            contrast_map_[d, c] += np.count_nonzero(c__ == c)

    # print time.time() - t1

    if debug:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(intensity_map_, aspect='auto')

        # with open(p.working_directory + '/temp/rf_f_importance.pkl') as f:
        #     y, x = pickle.load(f)

        # plt.scatter(x, y)
        # plt.show()

        # plt.figure()
        # plt.imshow(contrast_map_, aspect='auto')


    return np.ravel(intensity_map_), np.ravel(contrast_map_)
    # return list(np.ravel(intensity_map_)), list(np.ravel(contrast_map_))
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

def optimise_features(wd, fm_name):
    fm = FeatureManager(p.working_directory, db_name=wd +'/'+fm_name + '.sqlite3')
    fm_new = FeatureManager(p.working_directory, db_name=wd+'/'+ fm_name + '_opt.sqlite3')

    with open(wd+'/temp/'+fm_name+ '.pkl') as f:
        ids_, _, _, f_im = pickle.load(f)

    print np.sum(f_im > 0.00001)

    ids, fs = fm.get_all()

    # for id_, f in izip(ids, fs):
    #     f_new = f[ids_]
    #     fm_new.add(id_, f_new)


if __name__ == '__main__':
    from core.project.project import Project
    import cPickle as pickle

    wd = '/Users/flipajs/Documents/wd/FERDA/Cam1_playground'
    # wd = '/Users/flipajs/Documents/wd/FERDA/zebrafish_playground'
    # wd = '/Users/flipajs/Documents/wd/FERDA/Camera3'
    # wd = '/Users/flipajs/Documents/wd/FERDA/Sowbug3'
    p = Project()
    # p.load_hybrid(wd, state='isolation_score')
    p.load_hybrid(wd, state='eps_edge_filter')

    # optimise_features(wd, 'fm_idtracker_c_d50')
    #
    # #
    t = time.time()
    for j in range(1):
        for i in range(1, 7):
            r = p.rm[i]
            get_idtracker_features(r, p, debug=True, sub=4)

    print time.time() - t

    # plt.show()

    # test_regions = []
    single_region_ids, _ = get_single_region_ids(p)
    print len(single_region_ids)
    # print len(single_region_ids)
    # fm_idtracker_i = FeatureManager(p.working_directory, db_name='fm_idtracker_i_d50_test.sqlite3')
    # print "test"

    if False:
        # p.chm.add_single_vertices_chunks(p, fra mes=range(4500))
        p.gm.update_nodes_in_t_refs()

        if False:
            single_region_ids, _ = get_single_region_ids(p)

            fm_basic = FeatureManager(p.working_directory, db_name='fm_basic.sqlite3')
            fm_colornames = FeatureManager(p.working_directory, db_name='fm_colornames.sqlite3')
            fm_idtracker_i = FeatureManager(p.working_directory, db_name='fm_idtracker_i_d50.sqlite3')
            fm_idtracker_c = FeatureManager(p.working_directory, db_name='fm_idtracker_c_d50.sqlite3')
            fm_hog = FeatureManager(p.working_directory, db_name='fm_hog.sqlite3')
            fm_lbp = FeatureManager(p.working_directory, db_name='fm_lbp.sqlite3')

            fm_hog = FeatureManager(p.working_directory, db_name='fm_hog_fliplr.sqlite3')

            # fms = [fm_basic, fm_colornames, (fm_idtracker_i, fm_idtracker_c), fm_hog, fm_lbp]
            fms = [(fm_idtracker_i, fm_idtracker_c)]
            # fms = [fm_hog]
            # methods = [get_basic_properties, get_colornames_hists, get_idtracker_features, get_hog_features, get_lbp]
            methods = [get_idtracker_features]

            import time
            t1 = time.time()
            j = 0
            num_regions = len(single_region_ids)
            for r_id in single_region_ids:
                r = p.rm[r_id]

                j += 1

                for m, fm in zip(methods, fms):
                    if not isinstance(fm, tuple):
                        if fm[r_id][1][0] is not None:
                            continue
                    else:
                        if fm[0][r_id][1][0] is not None and fm[0][r_id][1][0] is not None:
                            continue

                    f = m(r, p)
                    if len(f) == 2:
                        f0 = f[0]
                        f1 = f[1]

                        try:
                            fm[0].add(r.id(), f0)
                            fm[1].add(r.id(), f1)
                        except Exception as e:
                            print e
                    else:
                        fm.add(r.id(), f)

                print_progress(j, num_regions)


            print "TIME: ", time.time() - t1

            # j = 0

            # for r_id in single_region_ids:
            #     r = p.rm[r_id]
            #     j += 1
            #
            #     f1, f2 = get_idtracker_features(r, p)
            #
            #     fm_idtracker_i.add(r.id(), f1)
            #     fm_idtracker_c.add(r.id(), f2)
            #
            #     print_progress(j, num_regions)


        # fm_names = ['fm_idtracker_i.sqlite3', 'fm_idtracker_i_d50.sqlite3', 'fm_idtracker_c.sqlite3', 'fm_idtracker_c_d50.sqlite3', 'fm_basic.sqlite3', 'fm_colornames.sqlite3']
        fm_names = ['fm_hog.sqlite3', 'fm_lbp.sqlite3', 'fm_idtracker_i_d50.sqlite3', 'fm_idtracker_c_d50.sqlite3', 'fm_basic.sqlite3', 'fm_colornames.sqlite3']
        fm_names = ['fm_idtracker_c_d50.sqlite3', 'fm_basic.sqlite3', 'fm_colornames.sqlite3']
        # fm_names = ['fm_hog_fliplr.sqlite3']
        fm_names = ['fm_idtracker_i_d50.sqlite3', 'fm_idtracker_c_d50.sqlite3']

        if True:
            results = evaluate_features_performance(p, fm_names, seed=42, test_split_method='random',
                                                    rf_class_weight='balanced_subsample', rf_criterion='entropy')

            print results
