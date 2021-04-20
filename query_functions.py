from sklearn.base import BaseEstimator
import numpy as np
from modAL.utils.data import modALinput


def query_function_wrapper(query_function):
    """
    Some queries functions need X0 (the initial/current train data) and others not.
    This function makes those that don't need, have the same interface/parameters.
    """
    return lambda classifier, X, X0, n_instances, **kwargs: query_function(classifier, X, n_instances, **kwargs)


def random_sampling(classifier: BaseEstimator, X: modALinput, X0, n_instances: int = 1, **kwargs) -> np.ndarray:
    """
    Gets n_instances samples from X, randomly chosen.
    """
    return np.random.permutation(range(len(X)))[:n_instances]
