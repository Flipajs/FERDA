from thesis.thesis_utils import load_all_projects
from utils.idtracker import load_idtracker_data
from utils.gt.evaluator import compare_trackers
import cPickle as pickle
from thesis.config import *


def run():
    WD = '/Users/flipajs/Documents/dev/ferda'
    ps = load_all_projects(semistate='id_classified', update_t_nodes=True, add_single_vertices=True)

    results = {}
    for nogaps in ['', '_nogaps']:
        for name in project_paths.iterkeys():
            if name not in ps:
                continue

            print name

            p = ps[name]
            path = idTracker_results_paths[name] + nogaps + '.mat'
            impath = WD+'/thesis/out/imgs/overall_' + name + nogaps + '.png'

            print path, p.working_directory, impath
            r = compare_trackers(p, path, impath=impath)

        results[name + nogaps] = r

        with open(WD+'/thesis/results/overall.pkl', 'wb') as f:
            pickle.dump(results, f)



if __name__ == '__main__':
    run()
