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
    partial = []
    lines = []
    chunks = []

    for edge in edges:
        if edge[0] is None or not (edge[0] in regions) or edge[1] is None or not (edge[1] in regions) or edge[2] == "partial":
            new_edge_tuple = edge[:2] + ("partial",) + edge[3:]
            partial.append(new_edge_tuple)
        else:
            length = used_frames_sorted.index(edge[1].frame_) - used_frames_sorted.index(edge[0].frame_)
            if edge[2] == "chunk":
                # or edge[3] == 1
                new_edge_tuple = edge[:2] + ("chunk",) + edge[3:]
                chunks.insert(length, new_edge_tuple)
            elif edge[2] == "line":
                # or edge[3] < 1
                new_edge_tuple = edge[:2] + ("line",) + edge[3:]
                lines.insert(length, new_edge_tuple)

    result = [] + chunks + lines + partial
    return result
