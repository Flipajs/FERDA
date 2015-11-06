__author__ = 'Simon Mandlik'


def sort_edges(edges, used_frames_sorted,):
    partial = []
    chunk_dict = {}
    line_dict = {}

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
                if length in line_dict.keys():
                    line = line_dict.pop(length)
                    if isinstance(line, list):
                        line.append(edge)
                        line_dict[length] = line
                    else:
                        line_dict[length] = [line, edge]
                else:
                    line_dict[length] = edge

    chunks = get_list_from_dict(chunk_dict)
    lines = get_list_from_dict(line_dict)

    result = list(reversed(chunks)) + list(reversed(lines)) + partial
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