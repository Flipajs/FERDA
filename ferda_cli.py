"""
CLI interface for various FERDA tasks.

Features:

- save found trajectories to a MOT challenge csv file

For more help run this file as a script with --help parameter.
"""
import time

from core.project.project import Project
import pandas as pd
import numpy as np
import os
from os.path import join
import logging
import logging.config
import yaml
import subprocess
import shutil
from core.config import config


def setup_logging():
    logger_config_file = 'logging_conf.yaml'
    if os.path.isfile(logger_config_file):
        with open(logger_config_file, 'r') as fr:
            log_conf = yaml.load(fr)
        logging.config.dictConfig(log_conf)
    else:
        print('logger configuration file {} not found, using default settings'.format(logger_config_file))


setup_logging()
logger = logging.getLogger()


def results_to_mot(results):
    """
    Create MOT challenge format DataFrame out of trajectories array.

    :param results: ndarray, shape=(n_frames, n_animals, 2); coordinates are in yx order, nan when id not present
    :return: DataFrame with frame, id, x, y, width, height and confidence columns
    """
    assert results.ndim == 3
    assert results.shape[2] == 2
    results[np.isnan(results)] = -1
    objs = []
    for i in range(results.shape[1]):
        df = pd.DataFrame(results[:, i, ::-1], columns=['x', 'y'])
        df['frame'] = range(1, results.shape[0] + 1)
        df['id'] = i + 1
        df = df[['frame', 'id', 'x', 'y']]
        objs.append(df)

    df = pd.concat(objs)

    df.sort_values(['frame', 'id'], inplace=True)
    df['width'] = -1
    df['height'] = -1
    df['confidence'] = -1
    return df


def fix_legacy_project(project_path):
    import pickle
    from PyQt4 import QtGui, QtCore

    project_dirname, project_filename = Project.get_project_dir_and_file(project_path)

    chm_path = os.path.join(project_dirname, 'chunk_manager.pkl')
    if os.path.isfile(chm_path):
        with open(chm_path, 'rb') as fr:
            chm = pickle.load(fr)
        for _, ch in chm.chunks_.iteritems():
            ch.project = None
            if isinstance(ch.color, QtGui.QColor):
                ch.color = ch.color.getRgb()[:3]
        with open(chm_path, 'wb') as fw:
            pickle.dump(chm, fw, -1)
        print('Fixed chunk.color.')
    else:
        print('Can\'t find chunk manager {}'.format(chm_path))

    with open(project_filename, 'rb') as fr:
        project_pickle = pickle.load(fr)
    if isinstance(project_pickle['name'], QtCore.QString):
        project_pickle['name'] = str(project_pickle['name'])
    with open(project_filename, 'wb') as fw:
        pickle.dump(project_pickle, fw, -1)
    print('Fixed project.name.')


def run_tracking(project_dir, video_file=None, force_recompute=False, reid_model_weights_path=None, results_mot=None):
    import core.segmentation
    from core.region.clustering import is_project_cardinality_classified
    import core.graph_assembly
    import core.graph.solver
    from core.id_detection.complete_set_matching import do_complete_set_matching
    logger.info('run_tracking: segmentation')
    project = Project()
    project.load(project_dir, video_file=video_file)
    if force_recompute or not core.graph_assembly.is_assemply_completed(project):
        core.segmentation.segmentation(project_dir)
    logger.info('run_tracking: graph assembly')
    project = Project()
    project.load(project_dir, video_file=video_file)
    if force_recompute or not core.graph_assembly.is_assemply_completed(project):
        graph_solver = core.graph.solver.Solver(project)
        core.graph_assembly.graph_assembly(project, graph_solver)
        project.save()
    logger.info('run_tracking: cardinality classification')
    if force_recompute or not is_project_cardinality_classified(project):
        project.region_cardinality_classifier.classify_project(project)
        project.save()
    logger.info('run_tracking: re-identification descriptors computation')
    if force_recompute or not os.path.isfile(join(project_dir, 'descriptors.pkl')):
        assert reid_model_weights_path is not None, \
            'missing reidentification model weights, to train a model see prepare_siamese_data.py, train_siamese_contrastive_lost.py'
        from scripts.CNN.siamese_descriptor import compute_descriptors
        compute_descriptors(project_dir, reid_model_weights_path)
    logger.info('run_tracking: complete set matching')
    do_complete_set_matching(project)
    project.save()
    if results_mot is not None:
        results = project.get_results_trajectories()
        df = results_to_mot(results)
        df.to_csv(results_mot, header=False, index=False)


def run_evaluation(mot_file, gt_file, out_evaluation_file, load_python3_env_cmd=None):
    """
    Evaluate tracking results against ground truth.

    Needs to run external python 3 process (evaluation depends on motmetrics module available only for python 3).

    :param gt_file:
    :param mot_file:
    :param out_evaluation_file:
    :param load_python3_env_cmd: command to start python 3 environment
    """
    if load_python3_env_cmd is not None:
        prefix = load_python3_env_cmd + '; '
    else:
        prefix = 'source ~/.virtualenvs/ferda3/bin/activate; '

    cmd = 'python -m utils.gt.mot --load-gt {} --load-mot {}  --eval --write-eval {}'.format(
        gt_file, mot_file, out_evaluation_file)
    subprocess.check_call(prefix + cmd, shell=True, executable='/bin/bash')  # throws an exception when evaluation goes wrong
                                                     # (e.g. python3 is not available)


def run_experiment(config):
    """
    Run and evaluate an experiment.

    - copy project template
    - create new experiment dir
    - run tracking
    - evaluate results
    """
    base_experiment_name = '{}_{}'.format(time.strftime("%y%m%d_%H%M", time.localtime()),
                                          config['dataset_name'])
    if config['exp_name']:
        base_experiment_name += '_' + config['exp_name']
    base_experiment_dir = join(config['dir'], base_experiment_name)
    if not os.path.exists(base_experiment_dir):
        os.mkdir(base_experiment_dir)

    project_dir = join(config['projects_dir'], base_experiment_name)
    shutil.copytree(config['dataset']['initial_project'], project_dir)

    results_file = join(base_experiment_dir, 'results.txt')
    write_experimental_info(base_experiment_dir, config)
    run_tracking(project_dir, results_mot=results_file,
                 reid_model_weights_path=config['dataset']['reidentification_weights'])
    run_evaluation(results_file, config['dataset']['gt'], join(base_experiment_dir, 'evaluation.csv'))


def write_experimental_info(experiment_dir, config):
    with open(os.path.join(experiment_dir, 'experiment.yaml'), 'w') as fw:
        yaml.dump(config, fw)


def run_benchmarking(notebook_path='experiments/tracking/benchmarking.ipynb',
                     html_output_path='experiments/tracking/benchmarking.html'):
    import nbformat
    from nbconvert import HTMLExporter
    from nbconvert.preprocessors import ExecutePreprocessor

    with open(notebook_path) as fr:
        nb = nbformat.read(fr, as_version=4)

    # run and write notebook
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    nb, resources = ep.preprocess(nb, {'metadata': {'path': '.'}})

    with open(notebook_path, 'wt') as fw:
        nbformat.write(nb, fw)

    # codecs.open(new_fnm, 'w', encoding='utf-8').write(output)

    # write output to html
    if html_output_path:
        html_exporter = HTMLExporter()
        html_exporter.template_path = ['.']
        html_exporter.template_file = 'nocode.tpl'
        body, resources = html_exporter.from_notebook_node(nb)
        with open(html_output_path, 'wt') as fw:
            fw.write(body)


if __name__ == '__main__':
    # run_benchmarking()
    # import sys
    # sys.exit()

    import argparse

    parser = argparse.ArgumentParser(description='Convert and visualize mot ground truth and results.')
    parser.add_argument('--project', type=str, help='project directory')
    parser.add_argument('--video-file', type=str, help='project input video file')
    parser.add_argument('--save-results-mot', type=str, help='write found trajectories in MOT challenge format')
    parser.add_argument('--fix-legacy-project', action='store_true', help='fix legacy project\'s Qt dependencies')
    parser.add_argument('--run-tracking', action='store_true', help='run tracking on initilized project')
    parser.add_argument('--reidentification-weights', type=str, help='tracking: path to reidentification model weights',
                        default=None)
    parser.add_argument('--info', action='store_true', help='show project info')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--run-experiments-yaml', type=str, help='run and evaluate experiments on all datasets described in yaml file')
    parser.add_argument('--experiment-name', type=str, help='experiment name')
    args = parser.parse_args()

    if args.info:
        import core.graph_assembly
        from core.region.clustering import is_project_cardinality_classified
        project = Project(args.project)
        print('assembled: {}'.format(core.graph_assembly.is_assemply_completed(project)))
        print('cardinality classified: {}'.format(is_project_cardinality_classified(project)))
        print('descriptors computed: {}'.format(os.path.isfile(join(project.working_directory, 'descriptors.pkl'))))
        if not project.chm:
            print('chunk manager not initialized')
        else:
            print('number of chunks {}'.format(len(project.chm)))
            print('tracklet cardinality stats: {}'.format(np.bincount([tracklet.segmentation_class for tracklet in project.chm.chunk_gen()])))

    if args.fix_legacy_project:
        fix_legacy_project(args.project)

    if args.run_tracking:
        project = Project()
        project.load(args.project, video_file=args.video_file)
        run_tracking(args.project, video_file=args.video_file, reid_model_weights_path=args.reidentification_weights)

    if args.save_results_mot:
        project = Project()
        project.load(args.project, video_file=args.video_file)
        results = project.get_results_trajectories()
        df = results_to_mot(results)
        df.to_csv(args.save_results_mot, header=False, index=False)

    if args.run_experiments_yaml:
        with open(args.run_experiments_yaml, 'r') as fr:
            experiments = yaml.load(fr)
        if args.experiment_name:
            experiments['exp_name'] = args.experiment_name
        datasets = experiments['datasets']
        experiment_config = experiments.copy()
        del experiment_config['datasets']
        for dataset_name, dataset in datasets.iteritems():
            experiment_config['dataset'] = dataset
            experiment_config['dataset_name'] = dataset_name
            run_experiment(experiment_config)



