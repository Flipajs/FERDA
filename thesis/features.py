from thesis_utils import load_all_projects
from core.id_detection.features import evaluate_features_performance_opt
import datetime
import cPickle as pickle

def _compute(X_data, y_data, fm_names, projects, c, out_name=None):
    results = {}

    for p_name, p in projects.iteritems():
        print "----------------- ", p_name, "-------------------------"
        results[p_name] = evaluate_features_performance_opt(p, X_data[p_name], y_data[p_name], fm_names,
                        seed=c['seed'],
                        train_n_times=c['train_n_times'],
                        test_split_method=c['test_split_method'],
                        test_split_ratio=c['test_split_ratio'],
                        rf_class_weight=c['rf_class_weight'],
                        rf_criterion=c['rf_criterion'],
                        rf_max_features=c['rf_max_features'],
                        rf_min_samples_split=c['rf_min_samples_split'],
                        rf_min_samples_leaf=c['rf_min_samples_leaf'],
                        rf_n_estimators=c['rf_n_estimators'],
                        rf_max_depth=c['rf_max_depth'])


    if out_name is None:
        dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_name = 'results/features_' + dt + '.pkl'

    with open(out_name, 'wb') as f:
        pickle.dump((results, c), f)

    print results

#
# def compute1():
#     projects = load_all_projects()
#
#     fm_names = ['fm_hog.sqlite3', 'fm_lbp.sqlite3', 'fm_idtracker_i.sqlite3', 'fm_idtracker_c.sqlite3',
#                 'fm_basic.sqlite3', 'fm_colornames.sqlite3']
#
#     c = {'seed': 42, 'test_split_method': 'random', 'rf_class_weight': 'balanced_subsample', 'rf_criterion': 'entropy'}
#     results = {}
#
#     i = 0
#
#     for p_name, p in projects.iteritems():
#         print "----------------- ", p_name, "-------------------------"
#         results[p_name] = evaluate_features_performance(p, fm_names, seed=c['seed'],
#                                                         test_split_method=c['test_split_method'],
#                                                         rf_class_weight=c['rf_class_weight'],
#                                                         rf_criterion=c['rf_criterion'])
#
#     dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     with open('results/features_' + dt + '.pkl', 'wb') as f:
#         pickle.dump((results, c), f)
#
#     print results



if __name__ == '__main__':
    wd = 'results/features_'
    cdefault = {}
    cdefault['seed'] = 42
    cdefault['test_split_method'] = 'random'
    cdefault['test_split_ratio'] = 0.95
    cdefault['rf_class_weight'] = 'balanced_subsample'
    cdefault['rf_criterion'] = 'entropy'
    cdefault['rf_max_features'] = 'auto'
    cdefault['rf_min_samples_split'] = 2
    cdefault['rf_min_samples_leaf'] = 1
    cdefault['rf_n_estimators'] = 10
    cdefault['rf_max_features'] = 0.5
    cdefault['rf_max_depth'] = None
    cdefault['train_n_times'] = 10

    c = cdefault

    # LOAD ALL FEATURES BEFORE...
    from thesis.thesis_utils import load_all_projects
    from core.id_detection.feature_manager import FeatureManager
    import numpy as np

    fm_names = ['fm_hog.sqlite3', 'fm_lbp.sqlite3', 'fm_idtracker_i.sqlite3', 'fm_idtracker_c.sqlite3',
                'fm_basic.sqlite3', 'fm_colornames.sqlite3']

    # fm_names = ['fm_hog.sqlite3', 'fm_lbp.sqlite3',
    #             'fm_basic.sqlite3', 'fm_colornames.sqlite3']

    X_data = {}
    y_data = {}

    projects = load_all_projects()

    for p_name, project in projects.iteritems():
        X_data[p_name] = {}
        y_data[p_name] = None

        from utils.gt.gt import GT
        gt = GT()
        gt.load(project.GT_file)

        single_region_ids, animal_ids = gt.get_single_region_ids(project, max_frame=5000)
        for fm_name in fm_names:
            fm = FeatureManager(project.working_directory, fm_name)
            X = []
            for r_id in single_region_ids:
                _, f_ = fm[r_id]
                X.append(f_[0])

            X_data[p_name][fm_name] = np.array(X)
        y_data[p_name] = np.array(animal_ids)

    # c = dict(cdefault)
    # _compute(X_data, y_data, fm_names, projects, c, wd + 'default')
    #
    # c = dict(cdefault)
    # c['rf_criterion'] = 'gini'
    # _compute(X_data, y_data, fm_names, projects, c, wd+'gini')
    #
    # c = dict(cdefault)
    # c['rf_min_samples_leaf'] = 2
    # _compute(X_data, y_data, fm_names, projects, c, wd+'min_samples_leaf_2')
    #
    # c = dict(cdefault)
    # c['rf_min_samples_leaf'] = 3
    # _compute(X_data, y_data, fm_names, projects, c, wd+'min_samples_leaf_3')
    #
    # c = dict(cdefault)
    # c['rf_min_samples_leaf'] = 5
    # _compute(X_data, y_data, fm_names, projects, c, wd + 'min_samples_leaf_5')

    #
    # ##### min_samples_leaf 5 vs 1
    # c = cdefault
    # c['rf_min_samples_leaf'] = 5
    # print
    # print "----------------------------------------"
    # print "MIN_SAMPLES_LEAF_5"
    # print
    # _compute(c, wd + 'min_samples_leaf_5')

    ##### min_samples_split 2 vs 10
    # c['rf_min_samples_leaf'] = 10
    # print
    # print "----------------------------------------"
    # print "MIN_F_10"
    # print
    # _compute(c, wd+'min_f_10')
    #
    #
    # ##### N_ESTIMATORS
    # c = dict(cdefault)
    # c['rf_n_estimators'] = 100
    # print
    # print "----------------------------------------"
    # print "N_ESTIMATORS_100"
    # print
    # _compute(c, wd+'n_estimators_100')


    # ##### GINI
    # c = dict(cdefault)
    # c['rf_criterion'] = 'gini'
    # print
    # print "----------------------------------------"
    # print "GINI"
    # print
    # _compute(c, wd+'gini')


    # # ##### min_samples_leaf 3 vs 1
    # c = cdefault
    # c['rf_min_samples_leaf'] = 2
    # print
    # print "----------------------------------------"
    # print "MIN_SAMPLES_LEAF_2"
    # print
    # _compute(c, wd+'min_samples_leaf_2')

    # c = cdefault
    # c['rf_min_samples_leaf'] = 3
    # print
    # print "----------------------------------------"
    # print "MIN_SAMPLES_LEAF_3"
    # print
    # _compute(c, wd+'min_samples_leaf_3')
    #
    # ##### min_samples_leaf 5 vs 1
    # c = cdefault
    # c['rf_min_samples_leaf'] = 5
    # print
    # print "----------------------------------------"
    # print "MIN_SAMPLES_LEAF_5"
    # print
    # _compute(c, wd + 'min_samples_leaf_5')





    # ## MAX DEPTH
    # c = cdefault
    # c['rf_max_depth'] = 10
    # print
    # print "----------------------------------------"
    # print "MAX_DEPTH_10"
    # print
    # _compute(c, wd + 'max_depth_10')
    #
    # ## MAX DEPTH
    # c = cdefault
    # c['rf_max_depth'] = 25
    # print
    # print "----------------------------------------"
    # print "MAX_DEPTH_25"
    # print
    # _compute(c, wd + 'max_depth_25')
    #
    # ## MAX DEPTH
    # c = cdefault
    # c['rf_max_depth'] = 50
    # print
    # print "----------------------------------------"
    # print "MAX_DEPTH_50"
    # print
    # _compute(c, wd + 'max_depth_50')
    #
    # ## MAX DEPTH
    # c = cdefault
    # c['rf_max_depth'] = 100
    # print
    # print "----------------------------------------"
    # print "MAX_DEPTH_100"
    # print
    # _compute(c, wd + 'max_depth_100')


    # ## MAX FEATURES
    # c = cdefault
    # c['rf_max_features'] = 'auto'
    # print
    # print "----------------------------------------"
    # print "MAX_FEATURES_AUTO"
    # print
    # _compute(c, wd + 'max_features_auto')
    #
    # ## MAX FEATURES
    # c = cdefault
    # c['rf_max_features'] = 0.10
    # print
    # print "----------------------------------------"
    # print "MAX_FEATURES_10"
    # print
    # _compute(c, wd + 'max_features_10')
    #
    # ## MAX FEATURES
    # c = cdefault
    # c['rf_max_features'] = 0.20
    # print
    # print "----------------------------------------"
    # print "MAX_FEATURES_20"
    # print
    # _compute(c, wd + 'max_features_20')
    #
    # ## MAX FEATURES
    # c = cdefault
    # c['rf_max_features'] = 0.30
    # print
    # print "----------------------------------------"
    # print "MAX_FEATURES_30"
    # print
    # _compute(c, wd + 'max_features_30')
    #
    # ## MAX FEATURES
    # c = cdefault
    # c['rf_max_features'] = 0.40
    # print
    # print "----------------------------------------"
    # print "MAX_FEATURES_40"
    # print
    # _compute(c, wd + 'max_features_40')
    #
    # ## MAX FEATURES
    # c = cdefault
    # c['rf_max_features'] = 0.50
    # print
    # print "----------------------------------------"
    # print "MAX_FEATURES_50"
    # print
    # _compute(c, wd + 'max_features_50_2')
    #
    # ## MAX FEATURES
    # c = cdefault
    # c['rf_max_features'] = 0.75
    # print
    # print "----------------------------------------"
    # print "MAX_FEATURES_75"
    # print
    # _compute(c, wd + 'max_features_75')
    #
    # ## MAX FEATURES
    # c = cdefault
    # c['rf_max_features'] = 1.0
    # print
    # print "----------------------------------------"
    # print "MAX_FEATURES_100"
    # print
    # _compute(c, wd + 'max_features_100')


    ##### N_ESTIMATORS
    # c = cdefault
    # c['rf_n_estimators'] = 20
    # print
    # print "----------------------------------------"
    # print "N_ESTIMATORS_20"
    # print
    # _compute(c, wd+'n_estimators_20')
    #
    #
    # ##### N_ESTIMATORS
    # c = cdefault
    # c['rf_n_estimators'] = 50
    # print
    # print "----------------------------------------"
    # print "N_ESTIMATORS_50"
    # print
    # _compute(c, wd+'n_estimators_50')
    #
    # ##### N_ESTIMATORS
    # c = cdefault
    # c['rf_n_estimators'] = 200
    # print
    # print "----------------------------------------"
    # print "N_ESTIMATORS_200"
    # print
    # _compute(c, wd+'n_estimators_200')