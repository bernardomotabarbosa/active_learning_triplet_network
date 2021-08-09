# Description

Active learning is the subset of machine learning in which a learning algorithm can query a user interactively to label data with the desired outputs. In active learning, the algorithm proactively selects the subset of examples to be labeled next from the pool of unlabeled data. The fundamental belief behind the active learner algorithm concept is that an ML algorithm could potentially reach a higher level of accuracy while using a smaller number of training labels if it were allowed to choose the data it wants to learn from.

Therefore, active learners are allowed to interactively pose queries during the training stage. These queries are usually in the form of unlabeled data instances and the request is to a human annotator to label the instance. This makes active learning part of the human-in-the-loop paradigm, where it is one of the most powerful examples of success.

In this repository I developed a triple neural network with the Random Forest classifier to apply the powerful technique of active learning in order to improve the quality of the classification with less need for data, bringing better results, advancing in terms of efficiency, saving time and reducing of costs. The data are from defects in submerged centrifugal pumps from the company Petrobras.

# Installation

Recommended to use python 3.6.9

```
git clone https://github.com/bernardomotabarbosa/active_learning_triplet_network.git
cd active_learning_triplet_network
pip install -r requirements.txt
pip install git+http://gitlab.ninfa.inf.ufes.br/ninfa-ufes/deep-rpdbcs#subdirectory=src/python
```

# Usage
```
python validation.py -i C:/Users/UserVert/Desktop/all/data_classified_v6 -o results.csv -c experiment_configs.yaml
```
The experiment_configs.yaml file contains the experimental settings.

# Result

![Result AL/RF](https://github.com/bernardomotabarbosa/active_learning_triplet_network/blob/master/results/charts/RF.png?raw=true)

This active learning methodology applied to deep modeling of BCS system failure detection and diagnosis was developed with the objective of reducing the expense of time and financial resources used to achieve the desired results with the modeling. Empirical evidence shows that active learning using acquisition functions such as Maximum Entropy or Top-Two-Margin is able to improve the performance of a model with a much smaller amount of data compared to a random selection of data to be labeled by an expert in the field. Furthermore, I found strong evidence that this technique is also useful in the training scenario of the model, considering a new defect that is still unrepresentative in the mass of training data.
