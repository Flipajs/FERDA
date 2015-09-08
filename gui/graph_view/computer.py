__author__ = 'Simon Mandlik'


def sort_edges(edges, regions, used_frames_sorted,):
    """
    Sorts given edges according to their length, partial edges are considered the shortest.
    Chunks are the longest by default and lines in the middle.

    Requires [(node1, node2, type, sure)] input.
    If either node1 or node2 is not passed or one of these nodes is not part of the graph,
    edge is considered as partial edge.
    If no sureness is given, edge is considered as 0% sure.
    If no type is given, edge is considered as line with minimal sureness.
    :return:
    """

    result = []
    partial = []
    lines = []
    chunks = []

    for edge in edges:
        if (edge[0] is None or edge[0] not in regions) or (edge[1] is None or edge[1] not in regions):
            edge[2] = "partial"
            partial.append(edge)
        else:
            length = used_frames_sorted.index(edge[0]._frame) - used_frames_sorted.index(edge[1]._frame)
            if edge[3] == "chunk" or edge[4] == "1":
                edge[2] = "chunk"
                chunks.insert(edge, length)
            elif edge[3] == "line" or int(edge[4]) < 1:
                edge[2] = "line"
                lines.insert(edge, length)

    result.append(chunks, lines, partial)
    return result


