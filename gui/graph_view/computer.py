__author__ = 'Simon Mandlik'


def sort_edges(edges, regions, used_frames_sorted,):
    partial = []
    lines = []
    chunks = []

    for edge in edges:
        if edge[2] == "partial":
            partial.append(edge)
        else:
            length = used_frames_sorted.index(edge[1].frame_) - used_frames_sorted.index(edge[0].frame_)
            if edge[2] == "chunk":
                chunks.insert(length, edge)
            elif edge[2] == "line":
                lines.insert(length, edge)

    #TODO smazat - testovaci ucely
    for a in [3, 10, 13, 14]:
        edge = lines.pop(a)
        partial.append((None, edge[1], "partial", 0))
        print(edge[1].frame_)
    for b in [1, 5, 7, 12]:
        edge = lines.pop(b)
        partial.append((edge[0], None, "partial", 0))
        print(edge[0].frame_)

    result = list(reversed(chunks)) + list(reversed(lines)) + partial
    return result
