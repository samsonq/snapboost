import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, log_loss
from utils import MeanSquaredError, Logistic
import warnings
warnings.filterwarnings("ignore")


class HNBM:
    """
    Heterogeneous Newton Boosting Machine
    Args:
        num_iterations (int): number of boosting iterations
        learning_rate (float): learning rate
        mode (string): classification or regression
    Attributes:
        ensemble_ (list): Ensemble after training
    """
    def __init__(self, num_iterations, learning_rate, mode="classification"):
        assert mode in ["classification", "regression"], "Invalid mode: specify 'classification' or 'regression'."
        self.mode = mode
        self.loss_ = MeanSquaredError if mode == "classification" else Logistic
        self.num_iterations_ = num_iterations
        self.learning_rate_ = learning_rate
        self.base_learners_ = []  # list of base learners
        self.probabilities_ = []  # list of sampling probabilities
        self.ensemble_ = []

    def fit(self, X, y):
        """
        Train the model
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Labels
        """
        z = np.zeros(X.shape[0])
        self.ensemble_ = []
        for _ in tqdm(range(0, self.num_iterations_)):
            g, h = self.loss_.compute_derivatives(y, z)
            base_learner = clone(np.random.choice(self.base_learners_, p=self.probabilities_))
            base_learner.fit(X, -np.divide(g, h), sample_weight=h)
            z += base_learner.predict(X) * self.learning_rate_
            self.ensemble_.append(base_learner)

    def predict(self, X):
        """
        Predict using the model
        Args:
            X (np.ndarray): Feature matrix
        """
        preds = np.zeros(X.shape[0])
        for learner in tqdm(self.ensemble_):
            preds += self.learning_rate_ * learner.predict(X)
        return preds

    def evaluate(self, X, y):
        """
        Evaluate trained model
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Labels
        """
        preds = self.predict(X)
        if self.mode == "classification":
            loss = log_loss(y, 1.0 / (1.0 + np.exp(-preds)))
            print("Log Loss: %.4f" % loss)
        else:
            loss = np.sqrt(mean_squared_error(y, preds))
            print("RMSE: %.4f" % loss)
        return loss
