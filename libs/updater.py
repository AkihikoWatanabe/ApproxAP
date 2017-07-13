# coding=utf-8

"""
A python implementation of ApproxAP.
"""

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from tqdm import tqdm

from update_func_approxap import approx_ap

class Updater():
    """ This class support ApproxAP updater.
    """

    def __init__(self, eta=0.01, alpha=10, beta=1):
        """
        Params:
            eta(float): learning rate 
            alpha(int): scaling constant for approximated position function
            beta(int): scaling constant for approximated truncation function
        """
        self.eta = eta
        self.alpha = alpha
        self.beta = beta

    def __get_shuffled_qids(self, x_dict, y_dict, epoch):
        """
        Params:
            x_dict(dict): dict of csr_matrix of feature vectors.
            y_dict(dict): dict of np.ndarray of labels corresponding to each feature vector
            epoch(int): current epoch number (the number is used for seed of random)
        Returns:
            qids(np.array): shuffled qids
        """

        qids = np.asarray(x_dict.keys())
        N = len(qids) # # of qids
        np.random.seed(epoch) # set seed for permutation
        perm = np.random.permutation(N)

        return qids[perm]

    def update(self, x_dict, y_dict, weight):
        """ Update weight parameter using ApproxAP.
        Params:
            x_dict(dict): dict of csr_matrix of feature vectors.
            y_dict(dict): dict of np.ndarray of labels corresponding to each feature vector
            weight(Weight): class of weight
        """
        assert len(x_dict) == len(y_dict), "invalid # of qids"
        
        qids = self.__get_shuffled_qids(x_dict, y_dict, weight.epoch)
        w = weight.get_dense_weight()
        for qid in tqdm(qids):
            w = approx_ap(x_dict[qid].toarray(), y_dict[qid], w, self.eta, self.alpha, self.beta)
            exit()
        weight.set_weight(sp.csr_matrix(w))
        weight.epoch += 1
