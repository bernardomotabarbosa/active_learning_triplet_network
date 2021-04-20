import numpy as np
from modAL import ActiveLearner
from scipy.spatial import distance


def query(model, Xpool, X_train, Y_train, query_size):

    transform_x0 = model['transformer'].transform(X_train)

    # creates an array of indices, sorted by unique element
    idx_sort = np.argsort(Y_train)

    # sorts records array so all unique elements are together
    sorted_Y_train = Y_train[idx_sort]

    # returns the unique values, the index of the first occurrence of a value, and the count for each element
    vals, idx_start, count = np.unique(sorted_Y_train, return_counts=True, return_index=True)

    # splits the indices into separate arrays
    res = np.split(idx_sort, idx_start[1:])

    list_centroid = []

    for r in range(len(res)):
        list_centroid.append(np.mean(transform_x0[res[r]], axis=0))

    transform_xpool = model['transformer'].transform(Xpool)

    distances = distance.cdist(transform_xpool, np.array(list_centroid))
    distances.sort(axis=1)
    distances = distances[:, :2]
    distances = distances[:, 1] - distances[:, 0]

    return distances.argsort()[:query_size]


def distance_active_learning(model, X0, Y0, Xpool, Ypool, Xtest, Ytest, query_size, query_budget, metrics, classifier_name):

    inital_len_x0 = len(X0)

    model.fit(X0, Y0)

    preds = model.predict(Xtest)
    Result = {'test_' + name: [m(Ytest, preds)] for name, m in metrics.items()}
    queried_samples = [0]
    queried_idxs = []
    while (query_budget > 0):
        if (query_budget < query_size):
            query_size = query_budget

        print(f'----------------------------------- my query_budget {query_budget} --------------------------------------------')

        query_idx = query(model, Xpool, X0, Y0, query_size)

        X0 = np.concatenate((X0, Xpool[query_idx]), axis=0)
        Y0 = np.concatenate((Y0, Ypool[query_idx]), axis=0)

        model.fit(X0, Y0)

        Xpool = np.delete(Xpool, query_idx, axis=0)
        Ypool = np.delete(Ypool, query_idx, axis=0)

        preds = model.predict(Xtest)
        for name, m in metrics.items():
            Result['test_' + name].append(m(Ytest, preds))

        queried_samples.append(queried_samples[-1] + query_size)
        queried_idxs.append(np.array(query_idx))
        query_budget -= query_size

    Result['queried samples'] = queried_samples
    Result['queried idxs'] = queried_idxs

    scores = {name: np.array(r) for name, r in Result.items()}
    scores['queried samples'] += inital_len_x0

    Results = {}
    Results[classifier_name] = scores

    return Results


def active_learning(active_estimator: ActiveLearner, Xpool, Ypool, Xtest, Ytest, query_size: int, query_budget: int, metrics):
    """
    Peforms active learning with speficied estimator and dataset.
    Args:
        metrics (dict): str->callback function.

    Returns:
        A dictionary where the key is the name of a metric (from parameter `metrics`) or statistic about the experiment.
        The value of the dict is a numpy array of size ceil(query_budget/query_size) + 1`
    """
    preds = active_estimator.predict(Xtest)
    Result = {'test_'+name: [m(Ytest, preds)] for name, m in metrics.items()}
    queried_samples = [0]
    queried_idxs = []
    while(query_budget > 0):
        if(query_budget < query_size):
            query_size = query_budget
        query_idx, _ = active_estimator.query(Xpool, X0=active_estimator.X_training,
                                              n_instances=query_size)  # TODO: see active_estimator.X_training
        active_estimator.teach(Xpool[query_idx], Ypool[query_idx])
        Xpool = np.delete(Xpool, query_idx, axis=0)
        Ypool = np.delete(Ypool, query_idx, axis=0)
        preds = active_estimator.predict(Xtest)
        for name, m in metrics.items():
            Result['test_'+name].append(m(Ytest, preds))
        queried_samples.append(queried_samples[-1]+query_size)
        queried_idxs.append(np.array(query_idx))
        query_budget -= query_size
        print(f'----------------------------------- modAL query_budget {query_budget} --------------------------------------------')
    Result['queried samples'] = queried_samples
    Result['queried idxs'] = queried_idxs
    return {name: np.array(r) for name, r in Result.items()}