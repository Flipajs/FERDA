"""
CLI interface for various FERDA tasks.

Features:

- save found trajectories to a MOT challenge csv file

For more help run this file as a script with --help parameter.
"""
from core.project.project import Project
import pandas as pd
import numpy as np


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


def fix_chunk_manager_color(project):
    import pickle
    from PyQt4 import QtGui
    chm_path = project.working_directory + '/chunk_manager.pkl'
    with open(chm_path, 'rb') as f:
        chm = pickle.load(f)

    for _, ch in chm.chunks_.iteritems():
        ch.project = None
        assert isinstance(ch.color, QtGui.QColor)
        ch.color = ch.color.getRgb()[:3]
    with open(chm_path, 'wb') as f:
        pickle.dump(chm, f, -1)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convert and visualize mot ground truth and results.')
    parser.add_argument('project', type=str, help='project file or directory')
    parser.add_argument('--video-file', type=str, help='project input video file')
    parser.add_argument('--save-results-mot', type=str, help='write found trajectories in MOT challenge format')
    parser.add_argument('--fix-chunk-manager-color', action='store_true', help='fix legacy project\'s ChunkManager.color')
    args = parser.parse_args()

    project = Project()
    project.load(args.project, video_file=args.video_file)

    if args.save_results_mot:
        results = project.get_results_trajectories()
        df = results_to_mot(results)
        df.to_csv(args.save_results_mot, header=False, index=False)

    if args.fix_chunk_manager_color:
        fix_chunk_manager_color(project)
