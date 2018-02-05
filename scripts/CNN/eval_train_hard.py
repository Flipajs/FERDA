import h5py
import os
import argparse
import sys
import string
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train siamese CNN with HARD loss (https://github.com/DagnyT/hardnet/)')

    parser.add_argument('--datadir', type=str,
                        default='/Users/flipajs/Documents/wd/FERDA/CNN_desc_training_data_zebrafish',
                        help='path to dataset')
    parser.add_argument('--num_negative', type=int, default=1,
                        help='name used for saving intermediate results')

    args = parser.parse_args()


    with h5py.File(args.datadir + '/pred.h5', 'r') as hf:
        pred = hf['data'][:]

    with h5py.File(args.datadir + '/imgs_test_hard_'+str(args.num_negative)+'.h5', 'r') as hf:
        X_test = hf['data'][:]

    with h5py.File(args.datadir + '/labels_test_hard_'+str(args.num_negative)+'.h5', 'r') as hf:
        y_test = hf['data'][:]


    pos_d = []
    neg_d = []
    for i in range(pred.shape[0]/(2+args.num_negative)):
        p1 = pred[i*(2 + args.num_negative)]
        p2 = pred[i*(2 + args.num_negative) + 1]
        n = pred[i*(2 + args.num_negative) + 2]

        pos_d.append(np.linalg.norm(p1-p2))
        neg_d.append(min(np.linalg.norm(n-p1), np.linalg.norm(n-p2)))

    from scipy import stats

    print stats.describe(pos_d)
    print stats.describe(neg_d)

    corr = np.sum(np.array(pos_d) < np.array(neg_d))
    print "{}, {:.2%}".format(corr, corr/float(len(pos_d)))

