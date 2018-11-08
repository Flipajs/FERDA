from __future__ import print_function
import sys, os, re, random
from scipy import misc
import tqdm
import sys
import h5py
import numpy as np

from keras.models import model_from_json
import string
from keras.models import Model

def classify_imgs(imgs, ids, results_map, dist_map):
    global DATA_DIR, MODEL_NAME, ADD_DIST_MAP

    json_file = open(DATA_DIR + "/" + MODEL_NAME + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(DATA_DIR + "/" + MODEL_NAME + ".h5")
    classification_model = loaded_model

    if ADD_DIST_MAP:
        classification_model = Model(input=[classification_model.inputs[0]],
                  output=[classification_model.output, classification_model.layers[2].input])

    # classification_model.summary()

    y_pred = classification_model.predict(imgs, verbose=1)

    if ADD_DIST_MAP:
        for i in range(len(imgs)):
            id_ = int(ids[i])
            results_map[id_] = y_pred[0][i, :]
            dist_map[id_] = y_pred[1][i, :]
    else:
        for i in range(len(imgs)):
            id_ = int(ids[i])
            results_map[id_] = y_pred[i, :]

if __name__ == '__main__':
    DATA_DIR = sys.argv[1]
    MODEL_NAME = sys.argv[2]
    ADD_DIST_MAP = False

    BGR_FORMAT = True
    if len(sys.argv) > 3:
        BGR_FORMAT = bool(string.atoi(sys.argv[3]))

    if len(sys.argv) > 4:
        ADD_DIST_MAP = bool(string.atoi(sys.argv[4]))

    print("SWAP BGR?: ", BGR_FORMAT)

    imgs = []
    names = []

    pattern = re.compile(r"(.)*\imgs.h5")

    results_map = {}
    dist_map = {}

    i = 0

    for fname in tqdm.tqdm(os.listdir(DATA_DIR+ '/test')):
        if pattern.match(fname):
            i += 1
            # if i < 28:
            #     continue
            # if i > 35:
            #     break

            n = fname[:10]
            with h5py.File(DATA_DIR + '/test/' + fname, 'r') as hf:
                imgs = hf['data'][:]
                print(imgs.shape)
            with h5py.File(DATA_DIR + '/test/' + n + 'ids.h5', 'r') as hf:
                ids = hf['data'][:]
                print(ids.shape)

            # we need to reverse image channels
            if BGR_FORMAT:
                imgs = imgs[:, :, :, ::-1]

            classify_imgs(imgs, ids, results_map, dist_map)

    import pickle
    with open(DATA_DIR+'/softmax_results_map.pkl', 'wb') as f:
        pickle.dump(results_map, f)

    if ADD_DIST_MAP:
        with open(DATA_DIR+'/softmax_dist_map.pkl', 'wb') as f:
            pickle.dump(dist_map, f)
