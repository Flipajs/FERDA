"""
CLI interface for various FERDA tasks.

Features:

- save found trajectories to a MOT challenge csv file

For more help run this file as a script with --help parameter.
"""
from core.project.project import Project
import pandas as pd
import numpy as np
import os


def results_to_mot(results):
    """
    Create MOT challenge format DataFrame out of trajectories array.

    :param results: ndarray, shape=(n_frames, n_animals, 2); coordinates are in yx order, nan when id not present
    :return: DataFrame with frame, id, x, y, width, height and confidence columns
    """
    results[np.isnan(results)] = -1
    objs = []
    for i, _ in enumerate(project.animals):
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


def run_tracking(project_path, video_file=None):
    _, project_filename = Project.get_project_dir_and_file(project_path)
    import core.segmentation
    import core.graph_assembly
    import core.graph.solver
    n_parts = core.segmentation.segmentation(project_filename)
    project = Project()
    project.load(project_filename, video_file=video_file)
    graph_solver = core.graph.solver.Solver(project)
    core.graph_assembly.graph_assembly(project, graph_solver, n_parts)
    project.save()
    project.region_cardinality_classifier.classify_project(project)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convert and visualize mot ground truth and results.')
    parser.add_argument('project', type=str, help='project file or directory')
    parser.add_argument('--video-file', type=str, help='project input video file')
    parser.add_argument('--save-results-mot', type=str, help='write found trajectories in MOT challenge format')
    parser.add_argument('--fix-legacy-project', action='store_true', help='fix legacy project\'s Qt dependencies')
    parser.add_argument('--run-tracking', action='store_true', help='run tracking on initilized project')
    args = parser.parse_args()

    if args.save_results_mot:
        project = Project()
        project.load(args.project, video_file=args.video_file)
        results = project.get_results_trajectories()
        df = results_to_mot(results)
        df.to_csv(args.save_results_mot, header=False, index=False)

    if args.fix_legacy_project:
        fix_legacy_project(args.project)

    if args.run_tracking:
        project = Project()
        project.load(args.project, video_file=args.video_file)
        run_tracking(args.project, video_file=args.video_file)
