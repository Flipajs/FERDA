# README

## FERDA - Fast Extremal Region Detector of Ants

- version 2.2.10
- FERDA is a multi object tracker developed mainly to meet the needs of biologists analysing video sequences of their animal experiments.

## Installation

### Conda

Install Conda: https://conda.io.

Setup Conda environment with GUI support:

`$ conda env create -f conda.yml`

Setup Conda environment for batch processing (no GUI):

`$ conda env create -f conda_noqt.yml`

### Manual

* python 2.7.\*
* [opencv](http://opencv.org) for python (pycv) builded with FFMPEG support (2.4.12)
* [PyQt4](https://www.riverbankcomputing.com/software/pyqt/download)
* [graph-tool](https://pypi.python.org/pypi/graph-tool) (2.26)

`$ pip install -r requirements.txt`

## Quickstart

Activate environment if necesary:

`$ conda activate ferda` or `$ conda activate ferda_noqt`

Start GUI:

`$ python main.py`

Note: when nothing is happening check the progress bars in the terminal.

Or use command line interace:

`$ python ferda_cli.py --help`
    
### Automatic Tracking    

1. new project
2. set video range, tracking arena and settings (sane defaults can be accepted)
3. teach algorithm to distinguish single and multiple animals
    - sort out the regions into 4 categories using the buttons under the regions, the classifier will learn on the go
    - inspect the results in the *results viewer* by checking the *t-class* checker box
    - the animals should be correctly classified as single, multiple, etc.
4. train algorithm to distinguish animal identities
    - pick *id detection* tab
    - click *compute features*
    - click *auto init*
    - click *learn/restart classifier*
    - now decide N tracklets by setting the *certainity eps* and the number *N* and hitting *do N steps*
    - repeat the last point as long as are the new decisions without much errors 


### Creating Ground Truth

The usual goal is to assign an id to all tracklets of a reasonable size (e.g. > 10 frames). No id is assigned to short tracklets and to groups of animals.

1. start with project with trained *region classifier* without any identified tracklets
   (correcting all errors in automatic results tends to be slower than creating ground 
   truth from scratch)
2. play the video from the beginning until are the animals separated
3. click on an animal, press *d* and select an id, the assignment propagates 
   in the corresponding tracklet
4. skip to either end of the tracklet using "go to tracklet end" (ctrl + e), "go to tracklet group end" (ctrl + shift + e) or "tracklet start" (ctrl + s), "tracklet group start" (ctrl + shift + s)
5. play the video until there is a tracklet of a reasonable length for the animal (pay attention to the cyan tracklet length marker on the id bar)
6. go to 3. until the animal is marked in the whole sequence
7. go to the frame where are all animals separated
8. go to 3. and annotate next animal

## Keyboard Shortcuts

settings dialog
: ctrl + ,

### Results Viewer

step forward
: n

step backward
: b

play / pause
: space

increase step (step value is visible next to FPS display)
: 2

decrease step
: 1

assign id to tracklet
: d

assign id to tracklet (advanced)
: shift + d

show / hide all animals overlays
: h

## Unit Tests

Run unit tests:

`python -m unittest discover -v`

## Architecture

Key components:

core/parallelization.py
- extraction of animal containing regions using MSER algorithm
- construction of region graph (nodes are regions in both space and time, edges are possible transitions)

## Known Issues

`AttributeError: 'Vertex' object has no attribute 'in_neighbors'`

- solution: upgrade Graph-tool to version > 2.26

`OpenCV Error: Assertion failed (chunk.m_size <= 0xFFFF)`

- opencv 3.4.1 introduced the issue https://github.com/opencv/opencv/issues/11126
- solution: downgrade opencv to version 3.3.1 or upgrade to opencv with the mentioned issue solved

## Team

* author: Filip Naiser, CTU Prague <mailto:filip@naiser.cz>
* author: Matěj Šmíd, CTU Prague <mailto:smidm@cmp.felk.cvut.cz>
* collaborator: Barbara Casillas Perez, Cremer Group IST Austria 
* supervisor: Prof. Jiří Matas, CTU Prague

## Acknowledgement

* Dita Hollmannová - long term intern in CMP CTU Prague. Databases, visualisations, segmentation.
* Šimon Mandlík - long term intern in CMP CTU Prague. Visualisations, PCA of ant shapes, region fitting problem.
* Michael Hlaváček - 1 month intern in CMP CTU Prague, he was working on early version of results visualiser.

