import sys
import numpy as np
from core.graph.region_chunk import RegionChunk


def ferda_trajectories_dict(project, frame_limits_start=0, frame_limits_end=-1):
    """
    frame_limits_end = -1 means - no limit
    Args:
        project:
        start_frame:
        end_frame:

    Returns:

    """
    num_animals = len(project.animals)

    trajectories = {}

    if frame_limits_end < 0:
        from utils.video_manager import get_auto_video_manager
        v = get_auto_video_manager(project)
        frame_limits_end = v.total_frame_count()

    for frame in range(frame_limits_start, frame_limits_end):
        trajectories[frame] = [None for i in range(num_animals)]

    for t in project.chm.chunk_gen():
        # if len(t.P) == 1 and len(t.P.union(t.N)) == num_animals:

        rch = RegionChunk(t, project.gm, project.rm)
        for r in rch.regions_gen():
            frame = r.frame()

            if frame_limits_start > frame:
                continue

            if frame_limits_end <= frame:
                break

            if len(t.P) == 1:
                ids = list(t.P)
            else:
                ids = list(set(range(len(project.animals))) - t.N)

            for id_ in ids:
                trajectories[frame][id_] = np.array((r.centroid()[0], r.centroid()[1]))

    return trajectories