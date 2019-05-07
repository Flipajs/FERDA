import os
import sys
import time
from itertools import product
from os.path import join
import yaml


class Parameters(dict):
    def load(self, filename):
        self.update(yaml.load(open(filename)))

    def save(self, filename):
        yaml.dump(dict(self), open(filename, 'w'))

    def get_batches(self):
        """
        Get batch iterator as a cartesian product of all parameters values.

        :return: iterator over sets parameter values, e.g. [[1, 1, 1], [1, 1, 2], [1, 1, 3]]
        """
        values = list(self.values())
        # turn scalars into tuples for product to work
        for i, value in enumerate(values):
            if not (isinstance(value, list) or isinstance(value, tuple)):
                values[i] = (value,)
        return product(*values)

    def get_multiple_value_keys(self):
        """
        Return keys with more than one value.

        :return: list of keys with multiple values
        """
        multiple_value_keys = []
        for key, value in self.items():
            if (isinstance(value, list) or isinstance(value, tuple)) and len(value) > 1:
                multiple_value_keys.append(key)
        return multiple_value_keys


class Experiment(object):
    def __init__(self):
        self.basename = ''
        self.config = {}
        self.dir = '.'
        self.tensorboard = False
        self.tensor_board_dir = None
        self.params = Parameters()
        self.files = {}  # arbitrary loaded files

    def __str__(self):
        return self.basename

    @classmethod
    def create(cls, name=None, prefix=True, params=None, config=None, tensorboard=False):
        """

        :param name: experiment name
        :param prefix: if True prefix experiment dir with current datetime, if str use specified prefix, else no prefix
        :param params: experimental parameters (loss_alpha, epochs, batch_size)
        :param config: experiment configuration (root_experiment_dir, root_tensor_board_dir)
        :param tensorboard: if True, initialize tensor_board_dir
        """
        experiment = cls()
        if prefix is True:
            experiment.basename = time.strftime("%y%m%d_%H%M", time.localtime())
        elif isinstance(prefix, str):
            experiment.basename = prefix
        if name is not None:
            if len(experiment.basename) != 0:
                experiment.basename += '_'
            experiment.basename += name
        assert len(experiment.basename)
        if config is not None:
            experiment.config = config
        experiment.dir = join(experiment.config.get('root_experiment_dir', '.'), experiment.basename)
        experiment.tensorboard = tensorboard
        if tensorboard:
            experiment.tensor_board_dir = join(experiment.config.get('root_tensor_board_dir', '.'), experiment.basename)
        experiment._create_dir()
        if params is not None:
            experiment.params.update(params)
        return experiment

    def _create_dir(self):
        try:
            os.makedirs(self.dir)
        except OSError:
            pass

    @classmethod
    def from_dir(cls, experiment_dir, *load_filenames):
        import pandas as pd
        experiment = cls()
        experiment.dir = experiment_dir
        experiment.params.load(join(experiment.dir, 'parameters.yaml'))
        experiment.basename = os.path.basename(os.path.normpath(experiment.dir))
        experiment.files = {}
        for filename in load_filenames:
            ext = os.path.splitext(filename)[1]
            try:
                if ext == '.csv':
                    experiment.files[filename] = pd.read_csv(join(experiment.dir, filename))
                elif ext == '.yaml':
                    experiment.files[filename] = yaml.load(open(join(experiment.dir, filename)))
                else:
                    experiment.files[filename] = open(join(experiment.dir, filename)).read()
            except IOError:
                experiment.files[filename] = None
        return experiment

    def load_experiments_recursive(self, *load_filenames):
        experiments = []
        for directory, dirnames, files in \
                sorted(os.walk(self.dir), key=lambda x: os.path.basename(x[0])):
            if directory == self.dir:
                continue
            if 'parameters.yaml' in files:
                experiments.append(Experiment.from_dir(directory, *load_filenames))
            else:
                pass
        return experiments

    def save_argv(self):
        with open(join(self.dir, 'arguments.txt'), 'w') as fw:
            fw.writelines('\n'.join(sys.argv))

    def save_params(self):
        self.params.save(join(self.dir, 'parameters.yaml'))

    def __iter__(self):
        """
        Iterate over subexperiments.

        :return: Experiment
        """
        changing_keys = self.params.get_multiple_value_keys()
        if len(changing_keys) == 0:
            yield self
        else:
            for batch_values in self.params.get_batches():
                parameters = dict(zip(self.params.keys(), batch_values))
                name_parts = []
                for key in changing_keys:
                    value = parameters[key]
                    if '/' in value:
                        # it's a path, transform it
                        value = os.path.basename(value)
                        # value = value.replace('/', '-')
                    name_parts.append('{}_{}'.format(key, value))

                config = dict(self.config)
                config['root_experiment_dir'] = self.dir
                config['root_tensor_board_dir'] = self.tensor_board_dir
                yield Experiment().create('_'.join(name_parts), prefix=False, params=parameters, config=config,
                                          tensorboard=self.tensorboard)


if __name__ == '__main__':
    # see root_experiment_dir for created directories
    config = {'root_experiment_dir': '.'}

    # create single experiment
    parameters = {'speed': 1,
                  'capacity': 0.5,
                  'type': 'aaa'}
    experiment = Experiment.create('experiment_single', params=parameters, config=config)
    print(experiment.basename)
    experiment.save_params()
    # ... do computation, save results to experiment.dir

    # create multiple (batch / grid search) experiments
    parameters = {'speed': [1, 2, 3],
                  'capacity': 0.5,
                  'type': ['aaa', 'bbb']}
    experiment_batch = Experiment.create('experiment_batch', params=parameters, config=config)
    for experiment in experiment_batch:
        print(experiment.basename)
        experiment.save_params()
        # ... do computation, save results to experiment.dir

    # load experiment and results
    experiment_dir = '.'
    experiment_dir = '/home/matej/prace/ferda/experiments/181211_0928_mobilenet_batch'
    experiment = Experiment.from_dir(experiment_dir, 'results.csv')

    # load multiple experiments and results
    root_experiment = Experiment.from_dir(experiment_dir)
    experiments = root_experiment.load_experiments_recursive('results.csv', 'results_train.csv')
    [str(exp) for exp in experiments]
    # ['loss_alpha_0.0',
    #  'loss_alpha_0.1',
    #  ...
    #  'loss_alpha_0.8',
    #  'loss_alpha_0.9']
    import pandas as pd
    df_results = pd.concat([exp.files['results.csv'] for exp in experiments], ignore_index=True)
    df_results['loss_alpha'] = pd.DataFrame([exp.params['loss_alpha'] for exp in experiments])
    df_results
    #     xy MAE  angle MAE  loss_alpha
    # 0  1.486147  25.170076         0.0
    # 1  1.429566   2.736373         0.1
    # ...
    # 7  1.745380   1.957427         0.7
    # 8  1.978683   1.976512         0.8




