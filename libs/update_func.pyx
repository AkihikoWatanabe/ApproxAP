# coding=utf-8

from const import SIGMOID_RANGE, DIM
import numpy as np
cimport numpy as np

DTYPE_INT8 = np.int8
DTYPE_FLOAT32 = np.float32
ctypedef np.int8_t DTYPE_INT8_t
ctypedef np.float32_t DTYPE_FLOAT32_t

cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)
def approx_ap( \
        np.ndarray[DTYPE_INT8_t, ndim=2, mode="c"] x_list, \
        np.ndarray[DTYPE_INT8_t, ndim=1, mode="c"] y_list, \
        np.ndarray[DTYPE_FLOAT32_t, ndim=1, mode="c"] weight, \
        float eta, \
        int alpha, \
        int beta):
    """ Calculate gradient for ApproxAP and update weight.
    Params:
        x_list(np.ndarray): list of np.ndarray of feature vectors.
        y_list(np.ndarray): list of np.ndarray of labels corresponding to each feature vector
        weight(np.ndarray): weight vector
        eta(float): learning rate
        alpha(int): scaling constant for approximated position function
        beta(int): scaling constant for approximated truncation function
    Returns:
        weight(float): updated weight
    """ 

    def logistic(float s_xy, int alpha):
        """ logistic function
        Params:
            s_xy(float): difference of score between x-th and y-th item
            alpha(float): scaling constant
        Returns:
            logistic(float): value of logistic function by given s_xy and alpha
        """
        cdef float x = -alpha * s_xy
    
        if x <= -SIGMOID_RANGE:
            return 1e-15
        elif x >= SIGMOID_RANGE:
            return 1.0 - 1e-15
    
        cdef float _logistic = np.exp(x) / (1.0 + np.exp(x))
    
        return _logistic
    
    def diff_logistic(float s_xy, int alpha):
        """ differential of logistic function
        Params:
            s_xy(float): difference between x-th and y-th item
            alpha(int): scaling constant for approximated position function
        Returns:
            diff_logistic(float): value of differential of logistic function
        """
        cdef float _diff_logistic = (1.0 - logistic(s_xy, alpha)) * logistic(s_xy, alpha)
    
        return _diff_logistic
    
    cdef int N = x_list.shape[0] # # of cases
    cdef int F_LEN = weight.shape[0] # # of feature dim
    cdef int D = 0 # # of relevance items
    cdef np.ndarray[DTYPE_FLOAT32_t, ndim=1, mode="c"] s = \
                                        np.zeros([N, ], dtype=DTYPE_FLOAT32)
    cdef np.ndarray[DTYPE_FLOAT32_t, ndim=1, mode="c"] pihat_arr = \
                                        np.zeros([N,], dtype=DTYPE_FLOAT32)
    cdef np.ndarray[DTYPE_INT8_t, ndim=3, mode="c"] x_xy_arr = \
                                        np.zeros([N, N, F_LEN], dtype=DTYPE_INT8)
    cdef int x_xy
    cdef np.ndarray[DTYPE_FLOAT32_t, ndim=2, mode="c"] diff_logistic_arr = \
                                        np.zeros([N, N], dtype=DTYPE_FLOAT32)
    cdef np.ndarray[DTYPE_FLOAT32_t, ndim=2, mode="c"] diff_logistic_pihat_arr = \
                                        np.zeros([N, N], dtype=DTYPE_FLOAT32)
    cdef np.ndarray[DTYPE_FLOAT32_t, ndim=2, mode="c"] dJ_dpihat_x_arr = \
                                        np.zeros([N, N], dtype=DTYPE_FLOAT32)
    cdef np.ndarray[DTYPE_FLOAT32_t, ndim=2, mode="c"] dJ_dpihat_y_arr = \
                                        np.zeros([N, N], dtype=DTYPE_FLOAT32)
    cdef np.ndarray[DTYPE_FLOAT32_t, ndim=2, mode="c"] gradient_of_pihat_arr = np.zeros([N, F_LEN], dtype=DTYPE_FLOAT32)

    cdef np.ndarray[DTYPE_FLOAT32_t, ndim=1, mode="c"] gradient = \
                                        np.zeros([F_LEN, ], dtype=DTYPE_FLOAT32)
    cdef np.ndarray[DTYPE_FLOAT32_t, ndim=1, mode="c"] first_term = \
                                        np.zeros([F_LEN, ], dtype=DTYPE_FLOAT32)
    cdef np.ndarray[DTYPE_FLOAT32_t, ndim=1, mode="c"] second_term = \
                                        np.zeros([F_LEN, ], dtype=DTYPE_FLOAT32)

    cdef float _sum = 0.0
    cdef np.ndarray[DTYPE_FLOAT32_t, ndim=1, mode="c"] _sum_arr = \
                                        np.zeros([F_LEN, ], dtype=DTYPE_FLOAT32)
    cdef np.ndarray[DTYPE_FLOAT32_t, ndim=1, mode="c"] _gradient_of_J = \
                                        np.zeros([F_LEN, ], dtype=DTYPE_FLOAT32)

    # calculate score for each case
    for i in xrange(N):
        for j in xrange(F_LEN):
            s[i] += weight[j] * x_list[i, j]

    # calculate # of relevance items
    for i in xrange(N):
        D += y_list[i]

    """
    for speed up
    """
    for x in xrange(N):
        # calculate pihat
        _sum = 0.0
        for y in xrange(N):
            if x == y:
                continue
            _sum += logistic(s[x] - s[y], alpha)
        _sum += 1.0
    
        pihat_arr[x] = _sum
 
    for x in xrange(N):
        for y in xrange(x, N):
            if x == y:
                continue
            # calculate x_xy
            for i in xrange(F_LEN):
                x_xy = -alpha * (x_list[x, i] - x_list[y, i])
                x_xy_arr[x, y, i] = x_xy
                x_xy_arr[y, x, i] = -x_xy

            # calculate diff_logistic
            diff_logistic_arr[x, y] = diff_logistic(s[x] - s[y], alpha)
            diff_logistic_pihat_arr[x, y] = diff_logistic(pihat_arr[x] - pihat_arr[y], beta)

            # calculate dJ_dpihat
            dJ_dpihat_x_arr[x, y] = -1.0 / pihat_arr[y] * beta * diff_logistic_pihat_arr[x, y]
            dJ_dpihat_y_arr[x, y] = -1.0 / (pihat_arr[y] ** 2) * logistic(pihat_arr[x] - pihat_arr[y], beta) - dJ_dpihat_x_arr[x, y]

    for x in xrange(N):
        # calculate gradient of pihat
        _sum_arr = np.zeros([F_LEN, ], dtype=DTYPE_FLOAT32)
        for y in xrange(N):
            if y == x:
                continue
            for i in xrange(F_LEN):
                _sum_arr[i] += diff_logistic_arr[x, y] * x_xy_arr[x, y, i]

        gradient_of_pihat_arr[x] = _sum_arr

    """
    calculate gradient
    """ 
    for y in xrange(N):
        # calculate first_term for gradient
        for i in xrange(F_LEN):
            first_term[i] += y_list[y] / (pihat_arr[y] ** 2) * gradient_of_pihat_arr[y, i]

        # calculate second_term for gradient
        for x in xrange(N):
            if x == y:
                continue
            # calculate gradient of J
            for i in xrange(F_LEN):
                if x_xy_arr[x, y, i] == 0:
                    continue
                _gradient_of_J[i] = dJ_dpihat_y_arr[x, y] * gradient_of_pihat_arr[y, i] + dJ_dpihat_x_arr[x, y] * gradient_of_pihat_arr[x, i]

            for i in xrange(F_LEN):
                second_term[i] += y_list[y] * y_list[x] * _gradient_of_J[i]

    # calculate gradient and update weight
    for i in xrange(F_LEN):
        gradient[i] = -1.0 / D * first_term[i] + 1.0 / D * second_term[i]
        weight[i] += eta * gradient[i]

    return weight
