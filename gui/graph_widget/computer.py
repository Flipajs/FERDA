from __future__ import division
from __future__ import unicode_literals
from past.utils import old_div
from heapq import *

from gui.graph_widget.graph_line import LineType

__author__ = 'Simon Mandlik'

def sort_edges(edges, used_frames_sorted):
    chunk_dict = {}
    heap = []

    for edge in edges:
        length = used_frames_sorted.index(edge.region_to.frame_) - used_frames_sorted.index(edge.region_from.frame_)
        if edge.type == LineType.TRACKLET or edge.type == LineType.PARTIAL_TRACKLET:
            if length in list(chunk_dict.keys()):
                chunk = chunk_dict.pop(length)
                if isinstance(chunk, list):
                    chunk.append(edge)
                    chunk_dict[length] = chunk
                else:
                    chunk_dict[length] = [chunk, edge]
            else:
                chunk_dict[length] = edge
        elif edge.type == LineType.LINE:
            frame0 = used_frames_sorted.index(edge.region_from.frame_)
            heappush(heap, ((len(used_frames_sorted) - length)
                            + ((used_frames_sorted[len(used_frames_sorted) - 1] - frame0) + old_div((1 - edge.sureness), float(len(used_frames_sorted)))), edge))

    chunks = get_list_from_dict(chunk_dict)
    lines = []

    while heap:
        lines.append(heappop(heap)[1])

    result = chunks + lines
    return result


def get_list_from_dict(dictionary):
    lengths = list(dictionary.keys())
    result = []
    for length in sorted(lengths, reverse=True):
        value = dictionary[length]
        if isinstance(value, list):
            for edge in value:
                result.append(edge)
        else:
            result.append(value)
    return result