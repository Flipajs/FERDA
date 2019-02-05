"""
CLI interface for various FERDA tasks.

Features:

- save found trajectories to a MOT challenge csv file

For more help run this file as a script with --help parameter.
"""
import time

from core.project.project import Project
import numpy as np
import os
from os.path import join
import logging
import logging.config
import yaml
import subprocess
import shutil
import itertools
from collections import defaultdict
import webbrowser
from utils.experiment import Parameters, Experiment

from utils.gt.mot import results_to_mot


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


def fix_orientation(project_path):
    import jsonpickle
    import utils.load_jsonpickle
    from shutil import move
    project = Project.from_dir(project_path)
    n_swaps = project.fix_regions_orientation()
    print('Swapped {} regions orientation.'.format(n_swaps))
    move(join(project_path, 'regions.json'), join(project_path, 'regions_old_orientation.json'))
    open(join(project_path, 'regions.json'), 'w').write(jsonpickle.encode(project.rm))


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


def run_experiment(config, force_prefix=None):
    """
    Run and evaluate an experiment.

    Use force_prefix to resume experiments, e.g. force_prefix='190107_2102'.

    ferda_tracking experiments:
        - copy project template
        - create new experiment dir
        - run tracking
        - evaluate results

    single_object_tracking experiments:
        - run single object trackers on a video
        - save trajectories to results.txt
    """
    params = config.copy()  # all multiple valued parameters will create experiment batches
    del params['dir']
    del params['run']
    root_experiment = Experiment.create(config.get('exp_name'),
                                        prefix=force_prefix if force_prefix is not None else True, params=params,
                                        config={'root_experiment_dir': join(config['dir'], config['dataset_name'])})
    for experiment in root_experiment:
        print(experiment.basename)
        experiment.save_params()
        mot_results_file = join(experiment.dir, 'results.txt')
        if config['run'] == 'ferda_tracking':
            # create FERDA project template
            project_dir = join(experiment.params['projects_dir'],
                               experiment.params['dataset_name'],
                               experiment.basename)
            shutil.copytree(experiment.params['dataset']['initial_project'], project_dir)

            run_tracking(project_dir, results_mot=mot_results_file,
                         reid_model_weights_path=experiment.params['dataset']['reidentification_weights'])
        elif config['run'] == 'single_object_tracking':
            from core.interactions.detect import track_video
            track_video(experiment.params['tracker_model'], experiment.params['dataset']['initial_project'],
                        experiment.dir)
        else:
            assert False, 'unknown run: {} value in the experiments configuration'.format(config['run'])

        if 'gt' in experiment.params['dataset']:
            run_evaluation(mot_results_file, experiment.params['dataset']['gt'], join(experiment.dir, 'evaluation.csv'))


def run_benchmarks(notebook_path='experiments/tracking/benchmarking.ipynb',
                   html_output_path='experiments/tracking/benchmarking.html'):
    # command line version uses also installed themes / css:
    # jupyter-nbconvert --execute --to html experiments/tracking/benchmarking.ipynb --template=experiments/nbconvert-nocode.tpl
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
        html_exporter.template_path = ['./experiments']
        html_exporter.template_file = 'nbconvert-nocode.tpl'
        body, resources = html_exporter.from_notebook_node(nb)
        with open(html_output_path, 'wt') as fw:
            fw.write(body)
        webbrowser.open(html_output_path)


def run_visualization(experiment_names, all_experiments, gt_file, in_video_file, out_video_file):
    """
    Create visualization of tracking results overlaid on a video.

    Results of multiple experiments are visualized in a grid.

    :param experiment_names: experiments to visualize
                             list of experiment names (partial matches are supported) or "gt" or
                             indices to all_experiments (e.g. -1 for last experiment) or
                             "*" for all experiments
    :param all_experiments: list of dicts, {'mot_trajectories', 'dirname' or 'exp_name'}
    :param gt_file: ground truth mot filename
    :param in_video_file: input video filename
    :param out_video_file: output visualization filename
    """
    from utils.gt.mot import visualize_mot, load_mot
    df_mots = []
    names = []
    # fix for missing exp_name keys
    for exp in all_experiments:
        if 'exp_name' not in exp and 'dirname' in exp:
            exp['exp_name'] = exp['dirname']
    for name in experiment_names:
        if name == 'gt':
            df_mots.append(load_mot(gt_file))
            names.append('ground truth')
        elif name == '*':
            names_all = []
            for experiment in all_experiments:
                df_mots.append(load_mot(experiment['mot_trajectories']))
                names_all.append(experiment['dirname'])
            prefix = os.path.commonprefix(names_all)
            suffix = os.path.commonprefix([n[::-1] for n in names_all])
            names.extend([n[len(prefix):-len(suffix) if suffix != '' else None] for n in names_all])
        elif isinstance(name, int):
            exp = all_experiments[name]
            df_mots.append(load_mot(exp['mot_trajectories']))
            names.append(exp['exp_name'])
        else:
            matching_experiments = [e for e in all_experiments if e['exp_name'] == name]
            if len(matching_experiments) != 1:
                matching_experiments = [e for e in all_experiments if name in e['exp_name']]
            assert len(matching_experiments) == 1, \
                'experiment {} not found or ambiguous match: {}'.format(name, matching_experiments)
            df_mots.append(load_mot(matching_experiments[0]['mot_trajectories']))
            names.append(name)

    assert out_video_file is not None
    visualize_mot(in_video_file, out_video_file, df_mots, names)


def load_experiments(experiments_config, evaluation_required=False, trajectories_required=False):
    experiments = defaultdict(list)
    for directory, dirnames, filenames in \
            sorted(os.walk(experiments_config['dir']), key=lambda x: os.path.basename(x[0])):
        if directory == experiments_config['dir']:
            continue

        if 'parameters.yaml' in filenames:
            with open(join(directory, 'parameters.yaml'), 'r') as fr:
                parameters = yaml.load(fr)
            if parameters.get('dataset_name') not in experiments_config['datasets']:
                # print('skipping experiment {}, unknown dataset {}'.format(directory, parameters.get('dataset_name')))
                continue
            if 'evaluation.csv' in filenames:
                parameters['evaluation'] = join(directory, 'evaluation.csv')
            elif evaluation_required:
                continue
            if 'results.txt' in filenames:
                parameters['mot_trajectories'] = join(directory, 'results.txt')
            elif trajectories_required:
                continue
            parameters['dirname'] = os.path.basename(directory)
            if 'datetime' not in parameters:
                from datetime import datetime
                try:
                    parameters['datetime'] = datetime.strptime(parameters['dirname'][:11], "%y%m%d_%H%M")
                except ValueError:
                    try:
                        parameters['datetime'] = datetime.strptime(parameters['exp_name'][:11], "%y%m%d_%H%M")
                    except ValueError:
                        pass
            if 'name' in parameters and 'exp_name' not in parameters:
                parameters['exp_name'] = parameters['name']
            experiments[parameters['dataset_name']].append(parameters)
            # print(parameters['exp_name'])
        # else:
        #     print('no parameters.yaml in {}'.format(directory))
    for dataset, dataset_experiments in experiments.items():
        experiments[dataset] = sorted(dataset_experiments, key=lambda x: x['datetime'])
    return experiments


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convert and visualize mot ground truth and results.')
    parser.add_argument('--project', type=str, help='project directory')
    parser.add_argument('--video-file', type=str, help='project input video file')
    parser.add_argument('--save-results-mot', type=str, help='write found trajectories in MOT challenge format')
    parser.add_argument('--fix-legacy-project', action='store_true', help='fix legacy project\'s Qt dependencies')
    parser.add_argument('--fix-orientation', action='store_true', help='fix single tracklets regions orientation')
    parser.add_argument('--run-tracking', action='store_true', help='run tracking on initilized project')
    parser.add_argument('--reidentification-weights', type=str, help='tracking: path to reidentification model weights',
                        default=None)
    parser.add_argument('--info', action='store_true', help='show project info')
    # parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--run-experiments-yaml', type=str, help='run and evaluate experiments on all datasets described in yaml file')
    parser.add_argument('--experiment-name', nargs='?', type=str, help='experiment name')
    parser.add_argument('--force-experiment-prefix', type=str, default=None, help='force output directory prefix, use to continue experiments')
    parser.add_argument('--run-visualizations-yaml', type=str, help='visualize experiments described in yaml file')
   # parser.add_argument('--run-benchmarks', action='store_true', help='run benchmarks and store results to a html file')
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

    if args.fix_orientation:
        fix_orientation(args.project)

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
            experiments_config = yaml.load(fr)
        if args.experiment_name:
            experiments_config['exp_name'] = args.experiment_name
        datasets = experiments_config['datasets']
        experiment_config = experiments_config.copy()
        del experiment_config['datasets']
        for dataset_name, dataset in datasets.iteritems():
            experiment_config['dataset'] = dataset
            experiment_config['dataset_name'] = dataset_name
            run_experiment(experiment_config, args.force_experiment_prefix)

    if args.run_visualizations_yaml:
        with open(args.run_visualizations_yaml, 'r') as fr:
            experiments_config = yaml.load(fr)
        experiments = load_experiments(experiments_config, trajectories_required=True)
        for dataset_name, dataset in experiments_config['datasets'].iteritems():
            if 'visualize_experiments' in dataset:
                print(dataset_name)
                run_visualization(dataset['visualize_experiments'], experiments[dataset_name],
                                  dataset['gt'], dataset['video'],
                                  join(experiments_config['dir'],
                                       time.strftime("%y%m%d_%H%M", time.localtime()) + '_visualization.mp4'))

    # if args.run_benchmarks:
    #     run_benchmarks()

