from heapq import *
from scipy.linalg._matfuncs_inv_ssq import fractional_matrix_power

__author__ = 'Simon Mandlik'

# deprecated due to horrible performance on big data
#
# class LinePriorityQueue:
#
#     def __init__(self, used_frames_sorted):
#         self.q = []
#         self.used_frames_sorted = used_frames_sorted
#
#     def add(self, line):
#         self.q.append(line)
#         index = len(self.q) - 1
#         while index > 0 and self.bigger(line, self.q[(index - 1) // 2]):
#             (self.q[index], self.q[(index - 1) // 2]) = (self.q[(index - 1) // 2], self.q[index])
#             index = (index - 1) // 2
#
#     def pop(self):
#         if len(self.q) == 1:
#             return self.q.pop()
#         else:
#             ret = self.q[0]
#             self.q[0] = self.q.pop(-1)
#             self.bubble()
#             return ret
#
#     def is_empty(self):
#         return len(self.q) == 0
#
#     def bubble(self):
#         index = 0
#         while index < len(self.q) - 1:
#             first_child = index * 2 + 1
#             second_child = index * 2 + 2
#             if first_child >= len(self.q):
#                 break
#             elif second_child > len(self.q) - 1:
#                 child = first_child
#             else:
#                 child = first_child if self.bigger(self.q[first_child], self.q[second_child]) else second_child
#             if not self.bigger(self.q[index], self.q[child]):
#                 (self.q[index], self.q[child]) = (self.q[child], self.q[index])
#                 index = child
#             else:
#                 break
#
# def bigger(self, line1, line2):
#         length1 = self.used_frames_sorted.index(line1[1].frame_) - self.used_frames_sorted.index(line1[0].frame_)
#         length2 = self.used_frames_sorted.index(line2[1].frame_) - self.used_frames_sorted.index(line2[0].frame_)
#         if length1 != length2:
#             return length1 > length2
#         elif line1[0].frame_ != line2[0].frame_:
#             return line1[0].frame_ < line2[0].frame_
#         else:
#             return line1[3] >= line2[3]


def sort_edges(edges, used_frames_sorted):
    partial = []
    chunk_dict = {}
    heap = []

    for edge in edges:
        if edge[2] == "partial":
            partial.append(edge)
        else:
            length = used_frames_sorted.index(edge[1].frame_) - used_frames_sorted.index(edge[0].frame_)
            if edge[2] == "chunk":
                if length in chunk_dict.keys():
                    chunk = chunk_dict.pop(length)
                    if isinstance(chunk, list):
                        chunk.append(edge)
                        chunk_dict[length] = chunk
                    else:
                        chunk_dict[length] = [chunk, edge]
                else:
                    chunk_dict[length] = edge
            elif edge[2] == "line":
                frame0 = used_frames_sorted.index(edge[0].frame_)
                heappush(heap, ((len(used_frames_sorted) - length)
                                + ((used_frames_sorted[len(used_frames_sorted) - 1] - frame0) + (1 - edge[3])
                                   / float(len(used_frames_sorted))), edge))

    print("Getting result")
    chunks = get_list_from_dict(chunk_dict)
    lines = []
    import time
    time1 = time.time()
    print("Popping from queue")
    while heap:
        lines.append(heappop(heap)[1])
    time2 = time.time()
    print("Popping from queue took {0} seconds".format(time2 - time1))

    result = list(reversed(chunks)) + lines + partial
    return result


def get_list_from_dict(dictionary):
    lengths = dictionary.keys()
    result = []
    for length in lengths:
        value = dictionary[length]
        if isinstance(value, list):
            for edge in value:
                result.append(edge)
        else:
            result.append(value)
    return result
