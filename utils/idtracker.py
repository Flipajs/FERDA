import numpy as np

def load_idtracker_data(path, project, gt):
    try:
        import scipy.io as sio
        data = sio.loadmat(path)
        data = data['trajectories']
        print(len(data))

        permutation_data = []

        for frame in range(len(data)):
            i = 0
            for x, y in data[frame]:
                if np.isnan(x):
                    continue

                i += 1

            if i == len(project.animals):
                break

        print("permutation search in frame {}".format(frame))

        # frame = 0
        for id_, it in enumerate(data[frame]):
            x, y = it[0], it[1]
            permutation_data.append((frame, id_, y, x))

        perm = gt.get_permutation(permutation_data)

        return data, perm
    except IOError:
        print("idtracker data was not loaded: {}".format(path))
        return None, None