import sys, os, re, random
from scipy import misc
import tqdm
import sys
import h5py
import numpy as np

from keras.models import model_from_json

def classify_imgs(imgs, results_map):
    global DATA_DIR, MODEL_NAME

    json_file = open(DATA_DIR + "/" + MODEL_NAME + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(DATA_DIR + "/" + MODEL_NAME + ".h5")
    classification_model = loaded_model

    y_pred = classification_model.predict(imgs, verbose=1)

    for i in range(len(imgs)):
        id_ = int(names[i][1][:-4])
        results_map[id_] = y_pred[i, :]


if __name__ == '__main__':
    DATA_DIR = sys.argv[1]
    MODEL_NAME = sys.argv[2]

    imgs = []
    names = []

    pattern = re.compile(r"(.)*\.jpg")

    results_map = {}

    for fname in tqdm.tqdm(os.listdir(DATA_DIR+ '/test')):
        if pattern.match(fname):
            with h5py.File(DATA_DIR + '/test/' + fname, 'r') as hf:
                imgs = hf['data'][:]
                classify_imgs(results_map)

    import pickle
    with open(DATA_DIR+'/softmax_results_map.pkl', 'wb') as f:
        pickle.dump(results_map, f)
