import pickle
import jsonpickle
import numpy as np

# in_file = 'test/project/Sowbug3_cut_300_frames/region_cardinality_clustering.pkl'
# out_file = 'test/project/Sowbug3_cut_300_frames/region_cardinality_clustering.json'

in_file = 'test/project/Sowbug3_cut_300_frames/descriptors.pkl'
out_file = 'test/project/Sowbug3_cut_300_frames/descriptors.npz'

# with open(in_file, 'rb') as fr:
#     data = pickle.load(fr)  # , fix_imports=True, encoding='latin1')
#

# open(out_file, 'w').write(jsonpickle.encode(data, keys=True, warn=True))

# json_data = open(out_file, 'r').read()
# obj = jsonpickle.decode(json_data, keys=True)
# pass