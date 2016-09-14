def pca_basis(X):
    # implementation trick
    T = X.T.dot(X)
    eigenValues, eigenVectors = np.linalg.eig(T)

    # sorting
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]

    # normalization
    Y = X.dot(eigenVectors)
    for i in range(Y.shape[1]):
        Y[:, i] = Y[:, i] / norm(Y[:, i])

    eigenValues = eigenValues/np.sum(eigenValues)

    return Y, eigenValues


def reconstruct(X, Y, X_mean):
    m, num_samples = X.shape

    # Apply the liner combination and add the mean image
    Z = np.zeros((Y.shape[0], X.shape[1]))
    for i in range(num_samples):
        temp = (Y[:, m - 1] * X[m - 1, i] + X_mean).real
        Z[:, i] = temp.reshape((Z.shape[0], ))

    return Z


def compact_representation(X, Y, m):
    W = Y[:, :m].T.dot(X)
    return W


def get_pca(chunks, number_of_data, number_of_eigen_v, chm, gm):
    matrix = []
    i = 1
    for ch in chunks:
        print "Chunk #{0}".format(i)
        i+= 1
        for vector in get_matrix(ch, number_of_data, chm, gm):
            # plt.plot(vector[::2], vector[1::2])
            # plt.show()
            matrix.append(vector)

    print "Constructing eigen ants"

    matrix = np.matrix(matrix)
    X = matrix.T

    X_mean = np.mean(X, axis=1)

    # center the data
    X = X-X_mean

    eigenAnts, eigenValues = pca_basis(X)

    X_c = compact_representation(X, eigenAnts, number_of_eigen_v)
    Z = reconstruct(X_c, eigenAnts[:, :number_of_eigen_v], X_mean)

    # print matrix.T - Z

    return eigenAnts


def get_eigenfaces(m, number_of_eigen_v):
    covariance_matrix = m.T.dot(m)
    eigenvalues, eigenvectors = eig(covariance_matrix)
    index = eigenvalues.argsort()[::-1]
    eigenfaces = eigenvectors[:, index]
    eigenfaces = eigenfaces[:, :number_of_eigen_v]
    eigenfaces = m.dot(eigenfaces)
    return eigenfaces