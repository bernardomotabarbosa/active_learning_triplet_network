from active_learning import active_learning, distance_active_learning
import numpy as np
import pandas as pd
import torch
from rpdbcs.datahandler.dataset import readDataset
import skorch
from modAL import ActiveLearner
from modAL.models import BayesianOptimizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tripletnet.networks import TripletNetwork, lmelloEmbeddingNet
from tripletnet.datahandler import BalancedDataLoader
from tripletnet.callbacks import LoadEndState
from tripletnet.TripletNetClassifierMCDropout import TripletNetClassifierMCDropout
import itertools
from tempfile import mkdtemp
from shutil import rmtree
from adabelief_pytorch import AdaBelief
from rpdbcs_yaml import load_yaml
from tripletnet.utils import PipelineExtended
import sklearn
from pathlib import Path
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from siamese_triplet.networks import ClassificationNet
import time
import random

RANDOM_STATE = 1
np.random.seed(RANDOM_STATE)
torch.cuda.manual_seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

DEEP_CACHE_DIR = mkdtemp()
PIPELINE_CACHE_DIR = mkdtemp()


def load_rpdbcs_data(data_dir, nsigs=100000):
    """
    Signal are normalized with respect to 37.2894Hz.
    Multi-label samples are discarded here.
    """
    df_d = readDataset(data_dir / 'freq.csv', data_dir / 'labels.csv',
                       remove_first=100, nsigs=nsigs, npoints=10800, dtype=np.float32, use_cache=False)
    df_d.discardMultilabel()
    targets, _ = df_d.getMulticlassTargets()
    df_d.normalize(37.28941975, n_jobs=0)

    return df_d


def get_base_classifiers(pre_pipeline=None):
    """
    Gets all traditional machine learning classifier that will be use in the experiments.
    They will be used in both TripletNet space and the Hand-crafted space.
    """

    clfs = []
    rf = RandomForestClassifier(n_estimators=1000, random_state=RANDOM_STATE, n_jobs=-1)
    rf_param_grid = {'max_features': [2, 3, 4, 5]}
    clfs.append(("RF", rf, rf_param_grid))

    if pre_pipeline is not None:
        return [(cname, Pipeline([pre_pipeline, ('base_clf', c)]), {"base_clf__%s" % k: v for k, v in pgrid.items()})
                for cname, c, pgrid in clfs]

    return clfs


def get_call_backs():
    """
    Callbacks used by the neural network.
    One of the callbacks is monitoring and saving the best epoch (lowest non zero triplets), 
        so that at the end of training the best is loaded and actually used for predictions.
    """
    checkpoint_callback = skorch.callbacks.Checkpoint(dirname=DEEP_CACHE_DIR,
                                                      monitor='non_zero_triplets_best')

    callbacks = [('non_zero_triplets', skorch.callbacks.PassthroughScoring(name='non_zero_triplets', on_train=True))]
    callbacks += [checkpoint_callback, LoadEndState(checkpoint_callback)]
    return callbacks


def create_neural_classifier():
    """
    Common neural net classifier.
    """

    optimizer_parameters = {'weight_decay': 1e-4, 'lr': 1e-3,
                            'eps': 1e-16, 'betas': (0.9, 0.999),
                            'weight_decouple': True, 'rectify': False}
    optimizer_parameters = {"optimizer__" + key: v for key, v in optimizer_parameters.items()}
    optimizer_parameters['optimizer'] = AdaBelief

    checkpoint_callback = skorch.callbacks.Checkpoint(dirname=DEEP_CACHE_DIR,
                                                      monitor='train_loss_best')
    callbacks = [checkpoint_callback, LoadEndState(checkpoint_callback)]

    parameters = {
        'callbacks': callbacks,
        'device': 'cuda',
        'max_epochs': 300,
        'train_split': None,
        'batch_size': 80,
        'iterator_train': BalancedDataLoader, 'iterator_train__num_workers': 0, 'iterator_train__pin_memory': False,
        'module__embedding_net': lmelloEmbeddingNet(8), 'module__n_classes': 5}
    parameters = {**parameters, **optimizer_parameters}
    convnet = skorch.NeuralNetClassifier(ClassificationNet, **parameters)
    return 'convnet', convnet, {}


def get_deep_transformers():
    """
    Constructs and returns Triplet Networks.
    """
    optimizer_parameters = {'weight_decay': 1e-4, 'lr': 1e-3,
                            'eps': 1e-16, 'betas': (0.9, 0.999),
                            'weight_decouple': True, 'rectify': False}
    optimizer_parameters = {"optimizer__" + key: v for key, v in optimizer_parameters.items()}
    optimizer_parameters['optimizer'] = AdaBelief

    parameters = {
        'callbacks': get_call_backs(),
        'device': 'cuda',
        'module': lmelloEmbeddingNet,
        'max_epochs': 300,
        'train_split': None,
        'batch_size': 80,
        'iterator_train': BalancedDataLoader, 'iterator_train__num_workers': 0, 'iterator_train__pin_memory': False,
        'margin_decay_value': 0.75, 'margin_decay_delay': 100}
    parameters = {**parameters, **optimizer_parameters}
    deep_transf = []
    tripletnet = TripletNetwork(module__num_outputs=8, init_random_state=100, **parameters)

    tripletnet_param_grid = {'batch_size': [80],
                             'margin_decay_delay': [500],
                             'module__num_outputs': [8]}
    deep_transf.append(("tripletnet", tripletnet, tripletnet_param_grid))

    return deep_transf


def get_metrics():
    """
    args:
        labels_names (dict): mapping from label code (int) to label name (str).
    returns:
        A dictionary where key is the name of the metric and its value is a callback function receiving two parameters.
    """
    scoring = {'accuracy': accuracy_score,
               'f1_macro': lambda p1, p2: f1_score(p1, p2, average='macro'),
               'precision': lambda p1, p2: precision_score(p1, p2, average='macro'),
               'recall': lambda p1, p2: recall_score(p1, p2, average='macro')}

    return scoring


def build_grid_search(t, base_classif, base_classif_param_grid):
    gridsearch_sampler = StratifiedShuffleSplit(n_splits=1, test_size=0.11, random_state=RANDOM_STATE)
    base_classif = GridSearchCV(base_classif, base_classif_param_grid, cv=gridsearch_sampler, n_jobs=-1)
    clf = PipelineExtended([('transformer', t),
                            ('base_classifier', base_classif)],
                           memory=PIPELINE_CACHE_DIR)
    return clf


def combine_transformer_classifier(transformers, base_classifiers):
    """
    Combines the TripletNetwork with a base classifier (ex: K-NN) to form a scikit-learn Pipeline.

    returns:
        A scikit-learn :class:`~sklearn.pipeline.Pipeline`.
    """

    for transf, base_classif in itertools.product(transformers, base_classifiers):
        transf_name, transf, transf_param_grid = transf
        base_classif_name, base_classif, base_classif_param_grid = base_classif
        classifier = build_grid_search(transf, base_classif, base_classif_param_grid)

        yield '%s + %s' % (transf_name, base_classif_name), classifier


def split_active_learning(X, Y, init_train_size, test_size, withdrawn_category):
    sampler0 = StratifiedShuffleSplit(n_splits=1, train_size=init_train_size, random_state=RANDOM_STATE)
    idxs_ini, idxs_others = next(sampler0.split(X, Y))
    x_0, y_0 = X[idxs_ini], Y[idxs_ini]
    xo, yo = X[idxs_others], Y[idxs_others]

    sampler_pool = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    idxs_pool, idxs_test = next(sampler_pool.split(xo, yo))
    x_pool, y_pool = xo[idxs_pool], yo[idxs_pool]
    x_test, y_test = xo[idxs_test], yo[idxs_test]

    if withdrawn_category != 'all':

        indexs_to_remove = np.where(y_0 == withdrawn_category)[0].tolist()
        idx_to_stay = random.sample(indexs_to_remove, k=5)
        for id in range(len(idx_to_stay)):
            indexs_to_remove.remove(idx_to_stay[id])

        y_0 = np.delete(y_0, [tuple(np.array(indexs_to_remove))])

        a0, a1, a2 = x_0.shape
        mask = np.ones_like(x_0, dtype=bool)
        mask[indexs_to_remove, np.arange(a1), :] = False
        x_0 = x_0[mask].reshape((-1, a1, a2))

    return x_0, y_0, x_pool, y_pool, x_test, y_test


def iterateActiveLearners(estimator: sklearn.base.BaseEstimator, x_0, y_0, query_strategies, estimator_name):
    """
    Transforms a scikit-learn :class:`~sklearn.base.BaseEstimator`
    into an active learner of :class:`modAL.ActiveLearner`.

    returns:
        Generator where each item is a tuple of (str,:class:`modAL.ActiveLearner`).
    """
    for qstrat_name, qstrat in query_strategies.items():
        new_estimator_name = "%s [%s]" % (estimator_name, qstrat_name)
        aclearner = ActiveLearner(estimator=estimator, X_training=x_0, y_training=y_0,
                                  query_strategy=qstrat)
        yield new_estimator_name, aclearner


def run_active_learning(classifier: sklearn.base.BaseEstimator, x_0, y_0, x_pool, y_pool, x_test, y_test,
                        query_strategies, query_size, budget, scoring, classifier_name, withdrawn_category,
                        early):
    results = {}
    for estimator_name, aclearner in iterateActiveLearners(classifier, x_0, y_0, query_strategies, classifier_name):
        scores, percent_each_class = active_learning(aclearner, x_pool, y_pool, x_test, y_test, query_size, budget,
                                                     scoring)
        scores['queried samples'] += len(x_0)
        results[estimator_name] = scores
        f = open(
            f"results/{withdrawn_category}/percent_{classifier_name}_{early}_{withdrawn_category}_{estimator_name}.txt",
            "w")
        f.write(str(percent_each_class))
        f.close()
    return results


def main(initial_configs, d):
    early = time.time()
    global DEEP_CACHE_DIR, PIPELINE_CACHE_DIR

    query_strategies = initial_configs.query_strategies

    x = np.expand_dims(d.asMatrix()[:, :6100], axis=1)  # Transforms shape (n,10800) to (n,1,6100).

    y, y_names = d.getMulticlassTargets()

    scoring = get_metrics()

    print(x.shape)
    unique, counts = np.unique(y, return_counts=True)
    print(dict(zip(unique, counts)))

    x_0, y_0, x_pool, y_pool, x_test, y_test = split_active_learning(x, y,
                                                                     init_train_size=initial_configs.init_train_size,
                                                                     test_size=initial_configs.test_size,
                                                                     withdrawn_category=initial_configs.withdrawn_category)

    # All classifiers scales features to mean=0 and std=1.
    base_classifiers = get_base_classifiers(('normalizer', StandardScaler()))

    results = {}  # All results are stored in this dict. The keys are the name of the classifiers.
    transformers = get_deep_transformers()

    for classifier_name, classifier in combine_transformer_classifier(transformers, base_classifiers):
        print(classifier_name, classifier)
        r, percent_each_class = distance_active_learning(classifier, x_0, y_0, x_pool, y_pool, x_test, y_test,
                                                         initial_configs.query_size, initial_configs.budget,
                                                         scoring, classifier_name)
        results.update(r)
        f = open(
            f"results/{initial_configs.withdrawn_category}/percent_{classifier_name}_{early}_{initial_configs.withdrawn_category}_topmargin.txt",
            "w")
        f.write(str(percent_each_class))
        f.close()
        r_2 = run_active_learning(classifier, x_0, y_0, x_pool, y_pool, x_test, y_test,
                                  query_strategies, initial_configs.query_size,
                                  initial_configs.budget,
                                  scoring, classifier_name, initial_configs.withdrawn_category,
                                  early)
        results.update(r_2)
        print(results)

    # Saving results
    results_asmatrix = []
    for classif_name, result in results.items():
        print("===%s===" % classif_name)
        queried_samples = result['queried samples']
        for rname, rs in result.items():
            if rname.startswith('test_') or 'time' in rname:
                if rname.startswith('test_'):
                    metric_name = rname.split('_', 1)[-1]
                else:
                    metric_name = rname
                print("%s: %f" % (metric_name, rs[-1]))
                for i, r in enumerate(rs):
                    results_asmatrix.append((classif_name, metric_name, i, queried_samples[i], r))

    if initial_configs.save_file is not None:
        df = pd.DataFrame(results_asmatrix,
                          columns=['classifier name', 'metric name', 'step', 'train size', 'value'])
        df.to_csv(
            f'results/{initial_configs.withdrawn_category}/dfs/' + str(initial_configs.save_file).replace('.csv', f'{str(early).replace(".", "")}_'
                                                                              f'{initial_configs.withdrawn_category}.csv'),
            index=False)
    rmtree(PIPELINE_CACHE_DIR)
    rmtree(DEEP_CACHE_DIR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--initial_configs', type=Path, required=True, help="YAML initial_configs file")
    parser.add_argument('-i', '--inputdata', type=Path, required=True, help="Input directory of dataset")
    parser.add_argument('-o', '--outfile', type=Path, required=True, help="Output csv file containing all the results.")
    args = parser.parse_args()
    initial_configs = load_yaml(args.initial_configs, args.inputdata, args.outfile)
    D = load_rpdbcs_data(initial_configs.dataset_path)
    main(initial_configs, D)

# python validation.py -i C:/Users/UserVert/Desktop/all/data_classified_v6 -o results.csv -c experiment_configs.yaml
