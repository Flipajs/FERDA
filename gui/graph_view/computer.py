__author__ = 'Simon Mandlik'


def sort_edges(edges, regions, used_frames_sorted,):
    partial = []
    lines = []
    chunks = []

    for edge in edges:
        if edge[2] == "partial":
            print("partial")
            partial.append(edge)
        else:
            length = used_frames_sorted.index(edge[1].frame_) - used_frames_sorted.index(edge[0].frame_)
            if edge[2] == "chunk":
                # or edge[3] == 1
                chunks.insert(length, edge)
            elif edge[2] == "line":
                # or edge[3] < 1
                lines.insert(length, edge)

    #testovaci ucely
    for a in [3, 10, 13, 14]:
        edge = lines.pop(a)
        partial.append((None, edge[1], "partial", 0))
    for b in [1, 5, 7, 12]:
        edge = lines.pop(b)
        partial.append((edge[0], None, "partial", 0))

    result = list(reversed(chunks)) + list(reversed(lines)) + partial
    return result

if __name__ == '__main__':
    for a in [1,2,3]:
        for b in [1,2,3]:
            print a, b