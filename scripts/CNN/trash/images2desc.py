from __future__ import print_function
import numpy as np
import sys, os, re, random
import h5py
import string
from scipy import misc
import tqdm
from scipy import misc
import tqdm
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Need a path to folder with images as a parameter...")

    WD = sys.argv[1]

    imgs = []
    ids = []
    classes = []

    pattern = re.compile(r"(.)*\.jpg")
    for dir_name in tqdm.tqdm(os.listdir(WD)):
        i = 0
        if os.path.isdir(WD+'/'+dir_name):
            try:
                class_id = string.atoi(dir_name)
            except:
                class_id = -1

            for fname in os.listdir(WD+'/'+dir_name):
                if pattern.match(fname):
                    img = misc.imread(WD + '/' + dir_name + '/' + fname)
                    imgs.append(img)
                    ids.append(fname[:-4])
                    classes.append(class_id)

                    i += 1

    imgs = np.array(imgs)
    print(imgs.shape)

    # load json and create model
    json_file = open('vision_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("vision_model.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    results = loaded_model.predict(imgs, verbose=1)
    print(results.shape)

    with h5py.File(WD+'/results.h5', 'w') as hf:
        hf.create_dataset("data", data=results)

    with h5py.File(WD + '/classes.h5', 'w') as hf:
        hf.create_dataset("data", data=classes)

    with h5py.File(WD + '/ids.h5', 'w') as hf:
        hf.create_dataset("data", data=ids)

