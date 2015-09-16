__author__ = 'Simon Mandlik'


def sort_edges(edges, regions, used_frames_sorted,):
    partial = []
    chunk_dict = {}
    line_dict = {}

    for edge in edges:
        if edge[2] == "partial":
            partial.append(edge)
        else:
            length = used_frames_sorted.index(edge[1].frame_) - used_frames_sorted.index(edge[0].frame_)
            if edge[2] == "chunk":
                if chunk_dict.has_key(length):
                    chunk = chunk_dict.pop(length)
                    if isinstance(chunk, list):
                        chunk.append(edge)
                        chunk_dict[length] = chunk
                    else:
                        chunk_dict[length] = [chunk, edge]
                else:
                    chunk_dict[length] = edge
            elif edge[2] == "line":
                if line_dict.has_key(length):
                    line = line_dict.pop(length)
                    if isinstance(line, list):
                        line.append(edge)
                        line_dict[length] = line
                    else:
                        line_dict[length] = [line, edge]
                else:
                    line_dict[length] = edge

    # #TODO smazat - testovaci ucely
    # for a in [3, 10, 13, 14]:
    #     edge = lines.pop(a)
    #     partial.append((None, edge[1], "partial", 0))
    #     print(edge[1].frame_)
    # for b in [1, 5, 7, 12]:
    #     edge = lines.pop(b)
    #     partial.append((edge[0], None, "partial", 0))
    #     print(edge[0].frame_)

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