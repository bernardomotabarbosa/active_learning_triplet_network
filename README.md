# Installation

Recommended to use python 3.6.9

```
git clone https://github.com/bernardomotabarbosa/active_learning_repo.git
cd active_learning
pip install -r requirements.txt
pip install git+http://gitlab.ninfa.inf.ufes.br/ninfa-ufes/deep-rpdbcs#subdirectory=src/python
```

# Usage
```
python validation.py -i C:/Users/UserVert/Desktop/all/data_classified_v6 -o results.csv -c experiment_configs.yaml
```
The experiment_configs.yaml file contains the experimental settings.

# Result

![Result AL/RF](https://github.com/bernardomotabarbosa/active_learning/blob/master/results/charts/RF.png?raw=true)
