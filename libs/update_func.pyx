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
def approx_ap(np.ndarray[DTYPE_FLOAT32_t, ndim=2] x_list, np.ndarray[DTYPE_INT8_t, ndim=1] y_list, np.ndarray[DTYPE_FLOAT32_t, ndim=1] weight, float eta, int alpha, int beta):
    """ Calculate gradient for ApproxAP and update weight.
    Params:
        x_list(csr_matrix): list of csr_matrix of feature vectors.
        y_list(np.ndarray): list of np.ndarray of labels corresponding to each feature vector
        weight(csr_matrix): weight vector
        eta(float): learning rate
        alpha(int): scaling constant for approximated position function
        beta(int): scaling constant for approximated truncation function
    Returns:
        weight(float): updated weight
    """

    def s_xy(float sx, float sy):
        """ difference of score between x and y
        Params:
            sx(float): score of x-th item
            sy(float): score of y-th item
        Returns:
            s_xy(float): difference of score between sx and sy
        """
        cdef float s_xy = sx - sy

        return s_xy

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

    def pihat(int x, np.ndarray[DTYPE_FLOAT32_t, ndim=1] s, int alpha):
        """ approximated position function
        Params:
            x(int): item index
            s(list): score list of items
            alpha(int): scaling constant for approximated position function
        Returns:
            pihat(float): value of approximated position function
        """
        cdef int len_s = len(s)
        cdef float _sum = 0.0

        for y in xrange(len_s):
            if x == y:
                continue
            _sum += logistic(s_xy(s[x], s[y]), alpha)
        _sum += 1.0

        return _sum

    def gradient_of_pihat(int x, np.ndarray[DTYPE_FLOAT32_t, ndim=1] s, np.ndarray[DTYPE_FLOAT32_t, ndim=2] x_list, int alpha, int F_LEN):
        """ gradient of approximated position function. See appendix B.1 equation (42).
        Params:
            x(int): item index
            s(list): list of score of items
            N(int): # of items
            x_list(csr_matrix): csr_matrix of features
        Returns:
            gradient_of_pihat(float): gradient of approximated position function
        """
        cdef int len_s = len(s)
        cdef np.ndarray[DTYPE_FLOAT32_t, ndim=1] _sum = np.zeros([F_LEN, ], dtype=DTYPE_FLOAT32)
        cdef np.ndarray[DTYPE_FLOAT32_t, ndim=1] _diff_logistic = np.zeros([len_s, ], dtype=DTYPE_FLOAT32)

        for y in xrange(len_s):
            if y == x:
                continue

            _diff_logistic[y] = diff_logistic(s_xy(s[x], s[y]), alpha)

            for i in xrange(F_LEN):
                _sum[i] += _diff_logistic[y] * (x_list[x, i] - x_list[y, i])
        
        for i in xrange(F_LEN):
            _sum[i] *= -alpha 

        return _sum

    def dJ_dpihat_x(int x, int y, np.ndarray[DTYPE_FLOAT32_t, ndim=1] s, int alpha, int beta):

        cdef float pihat_x = pihat(x, s, alpha)
        cdef float pihat_y = pihat(y, s, alpha)
        cdef float _diff_logistic = diff_logistic(pihat_x - pihat_y, beta)
        cdef float _dJ_dpihat_x = -1.0 / pihat_y * beta * _diff_logistic

        return _dJ_dpihat_x         

    def dJ_dpihat_y(int x, int y, np.ndarray[DTYPE_FLOAT32_t, ndim=1] s, int alpha, int beta):

        cdef float pihat_x = pihat(x, s, alpha)
        cdef float pihat_y = pihat(y, s, alpha)
        cdef float _logistic = logistic(pihat_x - pihat_y, beta)
        cdef float _diff_logistic = diff_logistic(pihat_x - pihat_y, beta)
        cdef float _dJ_dpihat_y = -1.0 / (pihat_y ** 2) * _logistic + 1.0 / pihat_y * beta * _diff_logistic

        return _dJ_dpihat_y

    def gradient_of_J(int x, int y, np.ndarray[DTYPE_FLOAT32_t, ndim=1] s, int alpha, int beta, np.ndarray[DTYPE_FLOAT32_t, ndim=2] x_list, int F_LEN):
        """ gradient of J. See appendix B.2 equation (45).
        Params:
            x(int): item index
            y(int): item index
            s(list): score of items
            alpha(int): scaling constant for approximated position function
            beta(int): scaling constant for approximated truncation function
            N(int): # of items
            x_list(csr_matrix): csr_matrix of features
        Returns:
            gradient_of_J(float): value of gradient of J by given args.
        """
        
        cdef np.ndarray[DTYPE_FLOAT32_t, ndim=1] _gradient_of_J = np.zeros([F_LEN, ], dtype=DTYPE_FLOAT32)
        cdef np.ndarray[DTYPE_FLOAT32_t, ndim=1] _gradient_of_pihat_x = gradient_of_pihat(x, s, x_list, alpha, F_LEN)
        cdef np.ndarray[DTYPE_FLOAT32_t, ndim=1] _gradient_of_pihat_y = gradient_of_pihat(y, s, x_list, alpha, F_LEN)
        cdef float _dJ_dpihat_x = dJ_dpihat_x(x, y, s, alpha, beta)
        cdef float _dJ_dpihat_y = dJ_dpihat_y(x, y, s, alpha, beta)

        for i in xrange(F_LEN):
            _gradient_of_J[i] = _dJ_dpihat_y * _gradient_of_pihat_y[i] + _dJ_dpihat_x * _gradient_of_pihat_x[i]

        return _gradient_of_J

    cdef int N = x_list.shape[0] # # of cases
    cdef int F_LEN = weight.shape[0] # # of feature dim

    # calculate score for each case
    cdef np.ndarray[DTYPE_FLOAT32_t, ndim=1] s = np.zeros([N, ], dtype=DTYPE_FLOAT32)
    for i in xrange(N):
        for j in xrange(F_LEN):
            s[i] += weight[j] * x_list[i, j]

    # calculate # of relevance items
    cdef int D = 0
    for i in xrange(N):
        D += y_list[i]
    
    cdef np.ndarray[DTYPE_FLOAT32_t, ndim=1] gradient = np.zeros([F_LEN, ], dtype=DTYPE_FLOAT32)
    cdef np.ndarray[DTYPE_FLOAT32_t, ndim=1] first_term = np.zeros([F_LEN, ], dtype=DTYPE_FLOAT32)
    cdef np.ndarray[DTYPE_FLOAT32_t, ndim=1] second_term = np.zeros([F_LEN, ], dtype=DTYPE_FLOAT32)
    cdef np.ndarray[DTYPE_FLOAT32_t, ndim=1] _gradient_of_pihat_y
    cdef float _pihat_y
    cdef np.ndarray[DTYPE_FLOAT32_t, ndim=1] _gradient_of_J

    for y in xrange(N):

        # calculate first_term for gradient
        _gradient_of_pihat_y = gradient_of_pihat(y, s, x_list, alpha, F_LEN)
        _pihat_y = pihat(y, s, alpha)

        for i in xrange(F_LEN):
            first_term[i] += y_list[y] / (_pihat_y ** 2) * _gradient_of_pihat_y[i]

        # calculate second_term for gradient
        for x in xrange(N):
            if x == y:
                continue

            _gradient_of_J = gradient_of_J(x, y, s, alpha, beta, x_list, F_LEN)

            for i in xrange(F_LEN):
                second_term[i] += y_list[y] * y_list[x] * _gradient_of_J[i]

    # calculate gradient and update weight
    for i in xrange(F_LEN):
        gradient[i] = -1.0 / D * first_term[i] + 1.0 / D * second_term[i]
        weight[i] += eta * gradient[i]

    return weight
