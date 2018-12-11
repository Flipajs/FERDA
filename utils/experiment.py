import os
import sys
import time
from itertools import product
from os.path import join


class Experiment(object):
    def __init__(self, name=None, prefix_datetime=True, params=None, config=None, tensorboard=True):
        """

        :param name: experiment name
        :param prefix_datetime: prefix experiment dir with current datetime
        :param params: experimental parameters (loss_alpha, epochs, batch_size)
        :param config: experiment configuration (root_experiment_dir, root_tensor_board_dir)
        :param tensorboard: if True, initialize tensor_board_dir
        """
        if prefix_datetime:
            self.basename = time.strftime("%y%m%d_%H%M", time.localtime()) + '_'
        else:
            self.basename = ''
        if name is not None:
            self.basename += name
        assert len(self.basename)
        if config is None:
            config = {}
        self.config = config
        root_dir = self.config.get('root_experiment_dir', '.')
        self.dir = join(root_dir, self.basename)
        root_tb_dir = self.config.get('root_tensor_board_dir', '.')
        self.tensorboard = tensorboard
        if tensorboard:
            self.tensor_board_dir = join(root_tb_dir, self.basename)
        else:
            self.tensor_board_dir = None
        self._create_dir()

        if params is None:
            self.params = {}
        else:
            self.params = params

    def _create_dir(self):
        try:
            os.makedirs(self.dir)
        except OSError:
            pass

    def write_argv(self):
        with open(join(self.dir, 'arguments.txt'), 'w') as fw:
            fw.writelines('\n'.join(sys.argv))

    def _get_batches(self):
        """
        Get batch iterator as a cartesian product of all parameters values.

        :return:
            batches - iterator over sets parameter values, e.g. [[1, 1, 1], [1, 1, 2], [1, 1, 3]]
            multiple_value_keys - set of parameters keys with multiple values
        """
        keys, values = zip(*self.params.items())
        values = list(values)
        single_value_keys = []
        multiple_value_keys = []
        for i, (key, value) in enumerate(zip(keys, values)):
            if not (isinstance(value, list) or isinstance(value, tuple)):
                values[i] = (value,)
                single_value_keys.append(key)
            else:
                multiple_value_keys.append(key)

        return product(*values), multiple_value_keys

    def __iter__(self):
        """
        Iterate over subexperiments.

        :return: Experiment
        """
        batches, changing_keys = self._get_batches()
        if len(changing_keys) == 0:
            yield self
        else:
            for batch_values in batches:
                parameters = dict(zip(self.params.keys(), batch_values))
                name = '_'.join(['{}_{}'.format(key, parameters[key]) for key in changing_keys])
                config = dict(self.config)
                config['root_experiment_dir'] = self.dir
                config['root_tensor_board_dir'] = self.tensor_board_dir
                yield Experiment(name, prefix_datetime=False, params=parameters, config=config,
                                 tensorboard=self.tensorboard)


if __name__ == '__main__':
    # see root_experiment_dir for created directories
    config = {'root_experiment_dir': '.'}
    import yaml
    # single experiment
    parameters = {'a': 1,
                  'b': 0.5,
                  'c': 'aaa'}
    experiment = Experiment('experiment_single', params=parameters, config=config, tensorboard=False)
    print(experiment.basename)
    yaml.dump(experiment.params, open(join(experiment.dir, 'parameters.yaml'), 'w'))
    # ... do computation, save results to experiment.dir

    # batch experiment
    parameters = {'a': [1, 2, 3],
                  'b': 0.5,
                  'c': ['aaa', 'bbb']}
    experiment_batch = Experiment('experiment_batch', params=parameters, config=config, tensorboard=False)
    for experiment in experiment_batch:
        print(experiment.basename)
        yaml.dump(experiment.params, open(join(experiment.dir, 'parameters.yaml'), 'w'))
        # ... do computation, save results to experiment.dir

