import numpy as np

def feature_descriptor(data, block_division=(2, 2), pyramid_levels=3, histogram_num_bins=8, histogram_density=False):
    """
    Computes histograms on different pyramid levels based on grid.
    Parameters
    ----------
    data : 2D np.array() image
    block_division : tuple(uint, uint), (3, 2) means on next pyramid level divide block height by 3 and width by 2
        (produces 6x more blocks)
    pyramid_levels : uint, 2 means on the first level compute a histogram of the whole image, then divide it according
        to block_division.
    histogram_num_bins : uint
    histogram_density : bool, If ``False``, the result will contain the number of samples in
        each bin. If ``True``, the result is the value of the
        probability *density* function at the bin, normalized such that
        the *integral* over the range is 1 (numpy.histogram is used)
    Returns
    -------
    np.array of shape(n, ) histogram values arranged level by level, in each level row by row in given block arrangement

    """

    pyramid_levels = int(pyramid_levels)
    if pyramid_levels < 1:
        raise Exception("number of pyramid levels cannot by smaller than 1")

    h, w = data.shape

    # allocate feature vector
    t_ = np.float if histogram_density else np.uint
    n = 0
    current_block_num = np.array((1, 1))
    for lvl in range(pyramid_levels):
        n += np.product(current_block_num) * histogram_num_bins
        current_block_num = np.multiply(current_block_num, np.array(block_division))

    features = np.empty((n, ), dtype=t_)

    current_block_num = np.array((1, 1))
    # pointer
    fp_ = 0
    for lvl in range(pyramid_levels):
        rs = np.linspace(0, h, current_block_num[0] + 1, dtype=np.int32)
        cs = np.linspace(0, w, current_block_num[1] + 1, dtype=np.int32)

        for row in range(current_block_num[0]):
            for col in range(current_block_num[1]):
                # TODO: is there a better way without .copy()? we cannot change shape for non-contiguous array...
                data_block = data[rs[row]:rs[row+1], cs[col]:cs[col+1]].copy()
                data_block.shape = (data_block.shape[0]*data_block.shape[1], )
                hist_, _ = np.histogram(data_block, bins=histogram_num_bins, density=histogram_density)
                features[fp_:fp_+histogram_num_bins] = np.asarray(hist_, dtype=t_)

                fp_ += histogram_num_bins

        # divide into smaller blocks
        current_block_num = np.multiply(current_block_num, np.array(block_division))

    return features