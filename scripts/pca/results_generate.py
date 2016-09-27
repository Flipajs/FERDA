from PyQt4 import QtGui
from matplotlib import gridspec
import os
import numpy as np
import matplotlib.pyplot as plt

import sys

from scripts.pca.eigen_widget import EigenWidget

# This module takes care of viewing results and saving important figures

def generate_eigen_ants_figure(project, ants, number_of_eigen_v):
    f = plt.figure(figsize=(number_of_eigen_v / 6 + 1, 6))
    gs1 = gridspec.GridSpec(number_of_eigen_v / 6 + 1, 6)
    gs1.update(wspace=0.3, hspace=0.1)
    for i in range(number_of_eigen_v):
        ax = plt.subplot(gs1[i])
        ax.plot(np.append(ants[i, ::2], ants[i, 0]), np.append(ants[i, 1::2], ants[i, 1]))
        ax.set_title("Eigenant #{0}".format(i + 1))
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes], visible=False)
    fig = plt.gcf()
    fig.suptitle('Dim reduction: {0}'.format(number_of_eigen_v), fontsize=23)
    plt.axis('equal')
    f.set_size_inches(30, 20)
    fold = os.path.join(project.working_directory, 'pca_results')
    if not os.path.exists(fold):
        os.mkdir(fold)
    f.savefig(os.path.join(fold, 'eigen_ants'), dpi=f.dpi)
    plt.ioff()


def generate_ants_image(X, X_R, X_C, r, c, i, fold):
    f = plt.figure(figsize=(r, c))
    gs1 = gridspec.GridSpec(r, c)
    gs1.update(wspace=0.025, hspace=0.05)
    for j in range(len(X)):
        ax1 = plt.subplot(gs1[j])
        plt.axis('on')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
        ax1.plot(np.append(X[j, ::2], X[j, 0]), np.append(X[j, 1::2], X[j, 1]), c='r')
        ax1.scatter(np.append(X[j, ::2], X[j, 0]), np.append(X[j, 1::2], X[j, 1]), c='r')
        ax1.plot(np.append(X_R[j, ::2], X_R[j, 0]), np.append(X_R[j, 1::2], X_R[j, 1]), c='b')
        ax1.scatter(np.append(X_R[j, ::2], X_R[j, 0]), np.append(X_R[j, 1::2], X_R[j, 1]), c='b')
        ax1.plot(np.arange(len(X_C[j, :])) + 1, X_C[j, :], c='g')

    # red_patch = mpatches.Patch(color='red', label='original')
    # blue_patch = mpatches.Patch(color='blue', label='reconstructed')
    # f.legend(handles=[red_patch], labels=[])
    f.set_size_inches(30, 20)
    f.savefig(os.path.join(fold, str(i)), dpi=f.dpi)
    plt.ioff()


def generate_ants_reconstructed_figure(project, X, X_R, X_C, rows, columns):
    number_in_pic = rows * columns
    fold = os.path.join(project.working_directory, 'pca_results')
    if not os.path.exists(fold):
        os.mkdir(fold)
    i = 0
    while X.shape[0] != 0:
        generate_ants_image(X[:number_in_pic, :], X_R[:number_in_pic, :], X_C[:number_in_pic, :], rows, columns, i,
                            fold)
        X = np.delete(X, range(number_in_pic), axis=0)
        X_R = np.delete(X_R, range(number_in_pic), axis=0)
        X_C = np.delete(X_C, range(number_in_pic), axis=0)
        i += 1
    generate_ants_image(X, X_R, X_C, rows, columns, i, fold)


def view_ant(pca, eigen_ants, eigen_values, ant):
    app = QtGui.QApplication(sys.argv)
    w = EigenWidget(pca, eigen_ants, eigen_values, ant)
    w.showMaximized()
    w.close_figures()
    app.exec_()
