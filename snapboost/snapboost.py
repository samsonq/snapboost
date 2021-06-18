import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from hnbm import HNBM
import warnings
warnings.filterwarnings("ignore")


class SnapBoost(HNBM):
    """
    A particular realization of a HNBM that uses decision trees and kernel ridge regressors
    Args:
        num_iterations (int): number of boosting iterations
        learning_rate (float): learning rate
        p_tree (float): probability of selecting a tree at each iteration
        min_max_depth (int): minimum maximum depth of a tree in the ensemble
        max_max_depth (int): maximum maximum depth of a tree in the ensemble
        alpha (float): L2-regularization penalty in the ridge regression
        gamma (float): RBF-kernel parameter
        mode (string): classification or regression
    """
    def __init__(self, num_iterations=100, learning_rate=0.1, p_tree=0.8,
                 min_max_depth=4, max_max_depth=8, alpha=1.0, gamma=1.0, mode="classification"):

        super().__init__(num_iterations, learning_rate, mode)

        # Insert decision tree base learners
        depth_range = range(min_max_depth,  1+max_max_depth)
        for d in depth_range:
            self.base_learners.append(DecisionTreeRegressor(max_depth=d, random_state=42))
            self.probabilities.append(p_tree/len(depth_range))

        # Insert kernel ridge base learner
        self.base_learners.append(KernelRidge(alpha=alpha, kernel='rbf', gamma=gamma))
        self.probabilities.append(1.0-p_tree)
