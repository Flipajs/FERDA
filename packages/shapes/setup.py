#!/usr/bin/env python
# coding=utf-8

from setuptools import setup

setup(
    name='shapes',
    version='0.1',
    description='2D shapes abstraction',
    author='Matěj Šmíd',
    author_email='m@matejsmid.cz',
#    url='https://github.com/bbcrd/audio-offset-finder',
#    license='Apache License 2.0',
    packages=['shapes'],
    install_requires=[
        'opencv-python',
        'numpy',
    ],
)

