# README

## FERDA - Fast Extremal Region Detector of Ants

- version 2.2.10
- FERDA is a multi object tracker developed mainly to meet the needs of biologists analysing video sequences of their animal experiments.

## How do I get set up?

### Dependencies:

* python 2.7.\*
* [opencv](http://opencv.org) for python (pycv) builded with FFMPEG support (2.4.12)
* [PyQt4](https://www.riverbankcomputing.com/software/pyqt/download)
* [graph-tool](https://pypi.python.org/pypi/graph-tool) (2.22)

    $ pip install -r requirements.txt

## Quickstart

- when nothing is happening check the progress bars in the terminal

    $ python main.py
    
### Automatic Tracking    

1. new project
2. set video range, tracking arena and settings (sane defaults can be accepted)
3. teach algorithm to distinguish single and multiple animals
    - pick *region classifier* tab and click *find candidates...* button
    - sort out the regions into 4 categories using the buttons under the regions, the classifier will learn on the go
    - when satisfied click on *classify tracklets*
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

## Team

* **author:** Filip Naiser (naisefil@cmp.felk.cvut.cz)
* **collaborators:** Barbara Casillas Perez, 
* **supervisor:** Prof. Jiří Matas

## Acknowledgement

* Dita Hollmannová - long term intern in CMP. Databases, visualisations, segmentation.
* Šimon Mandlík - long term intern in CMP. Visualisations, PCA of ant shapes, region fitting problem.
* Michael Hlaváček - 1 month intern in CMP, he was working on early version of results visualiser.
