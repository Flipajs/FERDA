__author__ = 'Simon Mandlik'


class LinePriorityQueue:

    def __init__(self, used_frames_sorted):
        self.q = []
        self.used_frames_sorted = used_frames_sorted

    def add(self, line):
        self.q.append(line)
        index = len(self.q) - 1
        while index > 0 and self.bigger(line, self.q[(index - 1) // 2]):
            (self.q[index], self.q[(index - 1) // 2]) = (self.q[(index - 1) // 2], self.q[index])
            index = (index - 1) // 2

    def pop(self):
        if len(self.q) == 1:
            return self.q.pop()
        else:
            ret = self.q[0]
            self.q[0] = self.q.pop(-1)
            self.bubble()
            return ret

    def is_empty(self):
        return len(self.q) == 0

    def bubble(self):
        index = 0
        while index < len(self.q) - 1:
            first_child = index * 2 + 1
            second_child = index * 2 + 2
            if first_child >= len(self.q):
                break
            elif second_child > len(self.q) - 1:
                child = first_child
            else:
                child = first_child if self.bigger(self.q[first_child], self.q[second_child]) else second_child
            if not self.bigger(self.q[index], self.q[child]):
                (self.q[index], self.q[child]) = (self.q[child], self.q[index])
                index = child
            else:
                break

    def bigger(self, line1, line2):
        length1 = self.used_frames_sorted.index(line1[1].frame_) - self.used_frames_sorted.index(line1[0].frame_)
        length2 = self.used_frames_sorted.index(line2[1].frame_) - self.used_frames_sorted.index(line2[0].frame_)
        if length1 != length2:
            return length1 > length2
        elif line1[0].frame_ != line2[0].frame_:
            return line1[0].frame_ < line2[0].frame_
        else:
            return line1[3] >= line2[3]


def sort_edges(edges, used_frames_sorted,):
    partial = []
    chunk_dict = {}
    lines_queue = LinePriorityQueue(used_frames_sorted)

    for edge in edges:
        if edge[2] == "partial":
            partial.append(edge)
        else:
            if edge[2] == "chunk":
                length = used_frames_sorted.index(edge[1].frame_) - used_frames_sorted.index(edge[0].frame_)
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
                lines_queue.add(edge)

    chunks = get_list_from_dict(chunk_dict)
    lines = []
    while not lines_queue.is_empty():
        lines.append(lines_queue.pop())

    # for line in lines:
    #     print("Start: " + str(line[0].frame_))
    #     print("End: " + str(line[1].frame_))
    #     print("Sureness: " + str(line[3]))
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
