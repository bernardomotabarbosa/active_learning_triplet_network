import yaml
from modAL.uncertainty import entropy_sampling
import query_functions
from query_functions import query_function_wrapper

_query_dict = {
    'random': query_functions.random_sampling,
    'entropy': query_function_wrapper(entropy_sampling)
}

_attrs = [
    'init_train_size',
    'test_size',
    'query_size',
    'budget',
    'withdrawn_category',
    'query_strategies'
]


class RALConfig:
    def __init__(self, dataset_path, save_file, **kwargs):
        super().__init__()

        # Check invalid keys
        attr_set = set(_attrs)
        kwkeyset = set(kwargs.keys())
        if not (attr_set & kwkeyset == attr_set):
            raise RuntimeError("YAML key error.")

        for key in _attrs:
            value = kwargs[key]
            setattr(self, key, value)

        # query strategies functions
        fs = {i: _query_dict[i] for i in self.query_strategies}
        self.query_strategies = fs

        self.dataset_path = dataset_path
        self.save_file = save_file

    def __repr__(self):
        repr_ = ""
        for i in _attrs:
            repr_ += "%s: %s\n" % (i, repr(getattr(self, i)))
        return repr_


def load_yaml(yaml_file, dataset_path, save_file):

    with open(yaml_file) as yaml_stream:
        al_file = yaml.load(yaml_stream, Loader=yaml.FullLoader)
        return RALConfig(dataset_path, save_file, **al_file)