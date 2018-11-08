from __future__ import unicode_literals
import yaml
import multiprocessing

with open('config.yaml', 'r') as fr:
    config = yaml.load(fr)

