from core.project.project import Project

project = Project()
project.load("/home/sheemon/FERDA/projects/Cam1_/cam1.fproj")


def get_chunks_regions(ch):
    chunk = project.chm[ch]
    chunk_start = chunk.start_frame(project.gm)
    chunk_end = chunk.end_frame(project.gm)
    while chunk_start <= chunk_end:
        yield project.gm.region(chunk[chunk_start])
        chunk_start += 1


def get_matrix(chunk):
    matrix = []
    for region in get_chunks_regions(chunk):
        matrix.append(get_region_vector(region))
    return matrix


def get_region_vector(region):
    vector = []
    print(region.contour())
    print(region.centroid())
    return vector


if __name__ == '__main__':
    chunks = project.gm.chunk_list()
    print(get_matrix(chunks[0]))
