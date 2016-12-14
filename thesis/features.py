from thesis_utils import load_all_projects
from core.id_detection.features import evaluate_features_performance
import datetime
import cPickle as pickle

def compute():
    projects = load_all_projects()

    fm_names = ['fm_hog.sqlite3', 'fm_lbp.sqlite3', 'fm_idtracker_i_d50.sqlite3', 'fm_idtracker_c_d50.sqlite3',
                'fm_basic.sqlite3', 'fm_colornames.sqlite3']

    c = {'seed': 42, 'test_split_method': 'random', 'rf_class_weight': 'balanced_subsample', 'rf_criterion': 'entropy'}
    results = {}

    i = 0

    for p_name, p in projects.iteritems():
        print "----------------- ", p_name, "-------------------------"
        results[p_name] = evaluate_features_performance(p, fm_names, seed=c['seed'],
                                                        test_split_method=c['test_split_method'],
                                                        rf_class_weight=c['rf_class_weight'],
                                                        rf_criterion=c['rf_criterion'])

    dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open('results/features_' + dt + '.pkl', 'wb') as f:
        pickle.dump((results, c), f)

    print results


if __name__ == '__main__':
    # compute()

    pass


