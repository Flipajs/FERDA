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
from core.region.region_manager import RegionManager
from utils.gt.mot import results_to_mot
import sys


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


def fix_orientation(project):
    n_swaps = project.fix_regions_orientation()
    print('Swapped {} regions orientation.'.format(n_swaps))


def run_tracking(project, force_recompute=False, reid_model_weights_path=None):
    import core.segmentation
    import core.graph_assembly
    import core.graph.solver
    from scripts.CNN.siamese_descriptor import compute_descriptors
    from core.id_detection.complete_set_matching import do_complete_set_matching
    if force_recompute:
        project.next_processing_stage = 'segmentation'
    if project.next_processing_stage == 'segmentation':
        logger.info('run_tracking: segmentation')
        core.segmentation.segmentation(project.working_directory)
        project.next_processing_stage = 'assembly'
        project.save()
    if project.next_processing_stage == 'assembly':
        logger.info('run_tracking: graph assembly')
        core.graph_assembly.graph_assembly(project)
        project.next_processing_stage = 'cardinality_classification'
        project.save()
    if project.next_processing_stage == 'cardinality_classification':
        logger.info('run_tracking: cardinality classification')
        project.region_cardinality_classifier.classify_project(project)
        project.next_processing_stage = 're-identification'
        project.save()
    if project.next_processing_stage == 're-identification':
        # not os.path.isfile(join(project.working_directory, 'descriptors.pkl')):
        logger.info('run_tracking: re-identification descriptors computation')
        assert reid_model_weights_path is not None, \
            'missing reidentification model weights, to train a model see prepare_siamese_data.py, train_siamese_contrastive_lost.py'
        compute_descriptors(project.working_directory, reid_model_weights_path)
        project.next_processing_stage = 'complete_sets_matching'
        project.save()
    if project.next_processing_stage == 'complete_sets_matching':
        logger.info('run_tracking: complete set matching')
        do_complete_set_matching(project)
        project.next_processing_stage = 'export_results'
        project.save()


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
            if os.path.exists(mot_results_file):
                print('{} already exists, skipping.'.format(mot_results_file))
            else:
                # create FERDA project template
                project_dir = join(experiment.params['projects_dir'],
                                   experiment.params['dataset_name'],
                                   experiment.basename)
                if not os.path.isdir(project_dir):
                    shutil.copytree(experiment.params['dataset']['initial_project'], project_dir)

                project = Project.from_dir(project_dir, regions_optional=True, graph_optional=True, tracklets_optional=True)
                run_tracking(project, reid_model_weights_path=experiment.params['dataset']['reidentification_weights'])
                save_results_mot(project, mot_results_file)
        elif config['run'] == 'single_object_tracking':
            from core.interactions.detect import track_video
            track_video(experiment.params['tracker_model'], experiment.params['dataset']['initial_project'],
                        experiment.dir)
        else:
            assert False, 'unknown run: {} value in the experiments configuration'.format(config['run'])

        if 'gt' in experiment.params['dataset']:
            evaluation_file = join(experiment.dir, 'evaluation.csv')
            if os.path.exists(evaluation_file):
                print('{} already exists, skipping.'.format(evaluation_file))
            else:
                run_evaluation(mot_results_file, experiment.params['dataset']['gt'], evaluation_file)


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


def load_experiments(experiments_dir, evaluation_required=False, trajectories_required=False):
    """
    Recursively search for directories with experiments.

    - metadata is loaded from experiment.yaml or parameters.yaml
    - evaluation from evaluation.csv
    - trajectories from results.txt
    - if experiment_name is not present in metadata directory basename is used

    :param experiments_dir: top level directory
    :param evaluation_required:
    :param trajectories_required:
    :return: list of experiments
    """
    import os
    import yaml
    from os.path import join
    import warnings

    experiments = []
    for directory, dirnames, filenames in \
            sorted(os.walk(experiments_dir), key=lambda x: os.path.basename(x[0])):
        if directory == experiments_dir:  # skip top level
            continue

        if 'experiment.yaml' in filenames:
            with open(join(directory, 'experiment.yaml'), 'r') as fr:
                metadata = yaml.load(fr)
        elif 'parameters.yaml' in filenames:
            with open(join(directory, 'parameters.yaml'), 'r') as fr:
                metadata = yaml.load(fr)
        else:
                metadata = {}

        if 'evaluation.csv' in filenames:
            metadata['evaluation_filename'] = join(directory, 'evaluation.csv')
        elif evaluation_required:
            continue

        if 'trajectories.txt' in filenames:
            metadata['mot_trajectories_filename'] = join(directory, 'trajectories.txt')
        elif trajectories_required:
            continue

        if not metadata:
            warnings.warn('no experiment found in {}'.format(directory))
            continue

        if 'experiment_name' not in metadata:
            metadata['experiment_name'] = os.path.basename(directory)

        if 'datetime' not in metadata:
            from datetime import datetime
            try:
                metadata['datetime'] = datetime.strptime(metadata['experiment_name'][:6], "%y%m%d")
                metadata['datetime'] = datetime.strptime(metadata['experiment_name'][:11], "%y%m%d_%H%M")
            except ValueError:
                try:
                    metadata['datetime'] = datetime.strptime(os.path.basename(directory)[:6], "%y%m%d")
                    metadata['datetime'] = datetime.strptime(os.path.basename(directory)[:11], "%y%m%d_%H%M")
                except ValueError:
                    if 'datetime' not in metadata or not metadata['datetime']:
                        warnings.warn('Can\'t parse datetime in {}.'.format(directory))

        experiments.append(metadata)

    try:
        experiments = sorted(experiments, key=lambda x: x['datetime'])
    except KeyError:
        warnings.warn('Can\'t sort experiments, some datetime was not parsed.')
    return experiments


def save_results_mot(project, out_filename):
    results = project.get_results_trajectories()
    df = results_to_mot(results)
    df.to_csv(out_filename, header=False, index=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convert and visualize mot ground truth and results.',
                                     epilog='use --project and --project-save-dir to convert project format')
    parser.add_argument('--project', type=str, help='load project from directory (old or new format)')
    parser.add_argument('--project-save-dir', type=str, help='save project to directory')
    parser.add_argument('--video-file', type=str, help='project input video file')
    parser.add_argument('--save-results-mot', type=str, help='write found trajectories in MOT challenge format')
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
            print('Processing {} dataset.'.format(dataset_name))
            run_experiment(experiment_config, args.force_experiment_prefix)
        sys.exit(0)

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
        sys.exit(0)

    project = Project.from_dir(args.project, video_file=args.video_file,
                               regions_optional=True, graph_optional=True, tracklets_optional=True)

    if args.info:
        print('next processing stage: ' + project.next_processing_stage)
        if not project.chm:
            print('chunk manager not initialized')
        else:
            print('number of chunks {}'.format(len(project.chm)))
            print('tracklet cardinality stats: {}'.format(np.bincount([tracklet.segmentation_class for tracklet in project.chm.chunk_gen()])))

    if args.fix_orientation:
        fix_orientation(project)

    if args.run_tracking:
        run_tracking(project, reid_model_weights_path=args.reidentification_weights)

    if args.save_results_mot:
        save_results_mot(project, args.save_results_mot)

    if args.project_save_dir:
        project.save(args.project_save_dir)

    # if args.run_benchmarks:
    #     run_benchmarks()

