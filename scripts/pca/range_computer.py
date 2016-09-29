import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


class OptimalRange:
    def __init__(self, X, sizes, number_of_data, number_of_eigen_v):
        self.X = X
        self.sizes = sizes
        self.X_pca_comp = np.zeros((X.shape[0], number_of_data * 2))
        for i in range(X.shape[0]):
            self.X_pca_comp[i] = X[i].flatten()
        self.number_of_data = number_of_data
        self.number_of_eigen_v = number_of_eigen_v

        # TODO ???
        self.min_k = 3
        self.max_k = number_of_data / 2 - 1
        self.results = np.zeros((number_of_data))

        self.pdf = {}
        self._compute_pdf()
        self.s = float(sum(self.pdf.values()))
        self._compute()

        # k = sorted(self.pdf.keys())
        # plt.plot(k, [self.pdf[x] for x in k])
        # plt.scatter(k, [self.pdf[x] for x in k])
        # plt.show()
        # plt.savefig('/home/simon/Desktop/gaus.jpg')

    def get_optimal_ks(self):
        return self.results

    def get_optimal_k(self, i):
        return self.results[i]

    def _compute_pdf(self):
        for i in range(self.number_of_data):
            for k in range(self.min_k, self.max_k + 1):
                pca_i = PCA(k)
                rolled = np.roll(self.X_pca_comp, 2 * (self.number_of_data - i), axis=1)
                pca_i.fit(rolled)
                cuts = rolled[:, range(2 * (- k), 2 * (k) + 2)]
                # TODO ???
                pca = PCA(k)
                f_coordinates = pca.fit_transform(cuts)
                scores = pca.score_samples(cuts)
                reconstructed = np.dot(f_coordinates, pca_i.components_) + pca_i.mean_

                for l in range(scores.shape[0]):
                    std = 0
                    for i in range(len(reconstructed[l])):
                        std += pow(reconstructed[l][i] - rolled[l][i], 2)
                    std /= float(len(reconstructed[l]) * self.sizes[l])
                    std = round(std, 3)
                    self.pdf[std] = self.pdf.get(std, 0) + 1

    def _compute(self):
        for i in range(self.number_of_data):
            # print self._compute_for_i(i)
            self._compute_for_i(i)

    def _compute_for_i(self, i):
        max = -100000000
        ret = 1
        for k in range(self.min_k, self.max_k + 1):
            pca_i = PCA(k)
            rolled = np.roll(self.X_pca_comp, 2 * (self.number_of_data - i), axis=1)
            pca_i.fit(rolled)
            prob = self._compute_for_k(i, k, pca_i, rolled)
            if prob > max:
                max = prob
                ret = k
                self.results[i] = k
        return ret, max

    def _compute_for_k(self, i, k, pca_i, rolled):
        cuts = rolled[:, range(2 * (- k), 2 * (k) + 2)]
        # TODO ???
        pca = PCA(k)
        f_coordinates = pca.fit_transform(cuts)
        scores = pca.score_samples(cuts)
        reconstructed = np.dot(f_coordinates, pca_i.components_) + pca_i.mean_

        for l in range(scores.shape[0]):
            scores[l] = self.err_function(scores[l], reconstructed[l], rolled[l], l)
            # plt.plot(reconstructed[l, ::2], reconstructed[l, 1::2], c='g')
            # plt.scatter(reconstructed[l, ::2], reconstructed[l, 1::2], c='g')
            # plt.plot(self.X_pca_comp[l, ::2], self.X_pca_comp[l, 1::2], c='y')
            # plt.scatter(self.X_pca_comp[l, ::2], self.X_pca_comp[l, 1::2], c='y')
            # plt.plot(rolled[l, ::2], rolled[l, 1::2], c='r')
            # plt.scatter(rolled[l, ::2], rolled[l, 1::2], c='r')
            # plt.plot(cuts[l, ::2], cuts[l, 1::2], c ='b')
            # plt.scatter(cuts[l, ::2], cuts[l, 1::2], c ='b')
            # plt.axis('equal')
            # plt.show()
        return np.mean(scores)

    def err_function(self, log_score, reconstructed, original, l):
        std = 0
        for i in range(len(reconstructed)):
            std += pow(reconstructed[i] - original[i], 2)
        std /= float(len(reconstructed) * self.sizes[l])
        std = round(std, 3)
        if std in self.pdf:
            l = self.pdf[std] / self.s
        else:
            l = 10e-10
        # TODO zaklad logaritmu?
        # print "Fit: {0}, PCA: {1}".format(l, pow(2, log_score))
        return l + pow(2, log_score)
