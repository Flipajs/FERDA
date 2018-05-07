import yaml
import multiprocessing

with open('config.yaml', 'r') as fr:
    config = yaml.load(fr)

if config['parallelization']['processes_num'] == -1:
    config['parallelization']['processes_num'] = multiprocessing.cpu_count() - 1

