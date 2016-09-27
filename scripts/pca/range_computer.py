import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

class RangeComputer:

    def __init__(self, X, number_of_data, number_of_eigen_v):
        self.X = X
        self.X_pca_comp = np.zeros((X.shape[0], number_of_data * 2))
        for i in range(X.shape[0]):
            self.X_pca_comp[i] = X[i].flatten()
        self.number_of_data = number_of_data
        self.number_of_eigen_v = number_of_eigen_v
        # TODO ???
        self.min_k = 3
        self.max_k = number_of_data / 2 - 1
        self.results = np.zeros((number_of_data))

        self.s = []
        self._compute()
        print max(self.s)
        print min(self.s)

    def get_optimal_ks(self):
        return self.results

    def get_optimal_k(self, i):
        return self.results[i]

    def _compute(self):
        for i in range(self.number_of_data):
            self._compute_for_i(i)

    def _compute_for_i(self, i):
        max = -100000000
        for k in range(self.min_k, self.max_k + 1):
            pca_i = PCA(k)
            rolled = np.roll(self.X_pca_comp, 2*(self.number_of_data - i), axis=1)
            pca_i.fit(rolled)
            prob = self._compute_for_k(i, k, pca_i, rolled)
            if prob > max:
                max = prob
                self.results[i] = k

    def _compute_for_k(self, i, k, pca_i, rolled):
        cuts = rolled[:, range(2*(- k), 2*(k) + 2)]
        # TODO ???
        pca = PCA(k)
        f_coordinates = pca.fit_transform(cuts)
        scores = pca.score_samples(cuts)
        reconstructed = np.dot(f_coordinates, pca_i.components_) + pca_i.mean_

        for l in range(scores.shape[0]):
            # plt.plot(reconstructed[l, ::2], reconstructed[l, 1::2], c='g')
            # plt.plot(self.X_pca_comp[l, ::2], self.X_pca_comp[l, 1::2], c='y')
            # plt.plot(rolled[l, ::2], rolled[l, 1::2], c='r')
            # plt.plot(cuts[l, ::2], cuts[l, 1::2], c ='b')
            # plt.axis('equal')
            # plt.show()
            scores[l] = err_function(scores[l], reconstructed[l], rolled[l], self.s)

        # self.results[i, k] = np.mean(scores)
        return np.mean(scores)


def err_function(log_score, reconstructed, original, scores):
        scores.append(pow(2,log_score))

        return 0










