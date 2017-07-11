# coding=utf-8

from predictor import Predictor
from rank_metrics import mean_reciprocal_rank, mean_average_precision
import math

def approx_ap(x_list, y_list, weight, eta, alpha, beta):
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

    def s_xy(sx, sy):
        """ difference of score between x and y
        Params:
            sx(float): score of x-th item
            sy(float): score of y-th item
        Returns:
            s_xy(float): difference of score between sx and sy
        """
        return sx - sy

    def logistic(s_xy, alpha):
        """ logistic function
        Params:
            s_xy(float): difference of score between x-th and y-th item
            alpha(float): scaling constant
        Returns:
            logistic(float): value of logistic function by given s_xy and alpha
        """
        return math.exp(-alpha * s_xy) / (1.0 + math.exp(-alpha * s_xy))

    def pihat(x, s, alpha):
        """ approximated position function
        Params:
            x(int): item index
            s(list): score list of items
            alpha(int): scaling constant for approximated position function
        Returns:
            pihat(float): value of approximated position function
        """
        return 1.0 + sum([logistic(s_xy(s[x], s[y]), alpha) for y in range(N) if x!=y])

    def gradient_of_pihat(x, s, N, x_list):
        """ gradient of approximated position function. See appendix B.1 equation (42).
        Params:
            x(int): item index
            s(list): list of score of items
            N(int): # of items
            x_list(csr_matrix): csr_matrix of features
        Returns:
            gradient_of_pihat(float): gradient of approximated position function
        """

        def diff_logistic(s_xy, alpha):
            """ differential of logistic function
            Params:
                s_xy(float): difference between x-th and y-th item
                alpha(int): scaling constant for approximated position function
            Returns:
                diff_logistic(float): value of differential of logistic function
            """
            return math.exp(alpha * s_xy) / ((1.0 + math.exp(alpha * s_xy)) ** 2)

        return -alpha * sum([diff_logistic(s_xy(s[x], s[y]), alpha) * (x_list[x] - x_list[y]) for y in range(N) if y!=x])

    def gradient_of_J(x, y, s, alpha, beta, N, x_list):
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
        dJ_dpihat_x = lambda x, y, s, alpha, beta: \
                -1.0 / pihat(y, s, alpha) * beta * math.exp(beta * ((pihat(x, s, alpha) - pihat(y, s, alpha)))) / \
                ((1.0 + math.exp(beta * (pihat(x, s, alpha) - pihat(y, s, alpha)))) ** 2)
        dJ_dpihat_y = lambda x, y, s, alpha, beta: \
                -1.0 / (pihat(y, s, alpha) ** 2) * 1.0 / (1.0 + math.exp(beta * (pihat(x, s, alpha) - pihat(y, s, alpha)))) + \
                1.0 / pihat(y, s, alpha) * beta * math.exp(beta * (pihat(x, s, alpha) - pihat(y, s, alpha))) / \
                ((1.0 + math.exp(beta * (pihat(x, s, alpha) - pihat(y, s, alpha)))) ** 2)

        return dJ_dpihat_y(x, y, s, alpha, beta) * gradient_of_pihat(y, s, N, x_list) + \
                dJ_dpihat_x(x, y, s, alpha, beta) * gradient_of_pihat(x, s, N, x_list)

    predictor = Predictor()

    N = x_list.shape[0] # # of items
    r_i = lambda i:y_list[i] # relevance score for i-th item
    s = predictor.predict(x_list, weight) # score of items
    D = sum([1.0 if relevance==1 else 0 for relevance in y_list])
    
    gradient = -1.0 / D * sum([r_i(y) / (pihat(y, s, alpha) ** 2) * gradient_of_pihat(y, s, N, x_list) for y in range(N)]) + \
            1.0 / D * sum([r_i(y) * r_i(x) * gradient_of_J(x, y, s, alpha, beta, N, x_list) for y in range(N) for x in range(N) if x!=y])
    weight += eta * gradient

    return weight
