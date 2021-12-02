import csv
import cv2
import numpy as np
from abc import ABCMeta, abstractmethod
from os.path import basename, splitext
import datetime
from pycococreatortools import pycococreatortools
import json


class IO(metaclass=ABCMeta):
    @abstractmethod
    def add_item(self, image): pass

    @abstractmethod
    def read_item(self): pass

    def close(self):
        pass


class ImageIOFile(IO):
    def __init__(self, filename_template):
        self.filename_template = filename_template
        self.next_idx = 0

    def next_filename(self):
        return self.filename_template.format(idx=self.next_idx)

    def add_item(self, image):
        cv2.imwrite(self.next_filename(), image)
        self.next_idx += 1

    def read_item(self):
        return cv2.imread(self.next_filename())
        self.next_idx += 1


class ImageIOHdf5(IO):
    def __init__(self, dataset):
        self.dataset = dataset
        self.next_idx = 0

    def add_item(self, image):
        self.dataset[self.next_idx] = image
        self.next_idx += 1

    def read_item(self):
        image = self.dataset[self.next_idx]
        self.next_idx += 1
        return image


class DataIOCSV(IO):
    def __init__(self, filename, columns=None):
        if columns is not None:
            self.csv_file = open(filename, 'w')
            self.writer = csv.DictWriter(self.csv_file, fieldnames=columns)
            self.writer.writeheader()
            self.reader = None
        else:
            assert False
            # df = pd.read_csv(filename)
            self.reader = None
            self.writer = None

    def add_item(self, data):
        if isinstance(data, dict):
            data = [data]
        for d in data:
            self.writer.writerow(d)

    def read_item(self):
        assert False

    def close(self):
        self.csv_file.close()


class DataIOVot(IO):
    def __init__(self, filename_template, image_filename_template=None, image_shape=None):
        self.template = '<annotation>\n\t<folder>GeneratedData_Train</folder>' \
                 '<filename>{filename}</filename>' \
                 '<path>{path}</path>' \
                 '<source>' \
                 '<database>Unknown</database>' \
                 '</source>' \
                 '<size>' \
                 '<width>{width}</width>' \
                 '<height>{height}</height>' \
                 '<depth>{depth}</depth>' \
                 '</size>' \
                 '<segmented>0</segmented>\n' \
                 '{annotation}</annotation>'
        self.bbox_template = '<object><name>{name}</name><bndbox>' \
                             '<xmin>{xmin}</xmin><xmax>{xmax}</xmax>' \
                             '<ymin>{ymin}</ymin><ymax>{ymax}</ymax></bndbox>' \
                             '<truncated>{truncated}</truncated>' \
                             '<difficult>0</difficult>' \
                             '{points}</object>'
        self.point_template = '<point id="{id}"><x>{x}</x><y>{y}</y></point>'
        self.next_idx = 0
        self.filename_template = filename_template
        if image_filename_template is not None and image_shape is not None:
            # writing
            self.image_filename_template = image_filename_template
            self.template_data = {'height': image_shape[0], 'width': image_shape[1],
                                  'depth': image_shape[2], 'path': ''}
        else:
            assert False

    def add_item(self, data):
        if isinstance(data, dict):
            data = [data]
        assert isinstance(data, list)

        bboxes_str = []
        for bbox in data:
            point_annotations = self.point_template.format(id=0, x=bbox['p0_x'] + 1, y=bbox['p0_y'] + 1) + \
                                self.point_template.format(id=1, x=bbox['p1_x'] + 1, y=bbox['p1_y'] + 1)  # coordinates are 1-based
            for key, val in list(bbox.items()):
                if key in ['xmin', 'xmax', 'ymin', 'ymax']:
                    bbox[key] = val + 1  # coordinates are 1-based
            bboxes_str.append(self.bbox_template.format(points=point_annotations, **bbox))
        xml_str = self.template.format(annotation=''.join(bboxes_str),
                                        filename=self.image_filename_template.format(idx=self.next_idx),
                                        **self.template_data)
        open(self.filename_template.format(idx=self.next_idx), 'w').write(xml_str)
        self.next_idx += 1

    def read_item(self):
        annotation = open(self.filename_template.format(idx=self.next_idx), 'w').read()
        # parse xml
        assert False

    def write_imageset(self, out_filename, idx_range=None):
        if idx_range is None:
            idx_range = list(range(self.next_idx))
        imageset_str = '\n'.join([splitext(basename(self.filename_template.format(idx=i)))[0] for i in idx_range])
        open(out_filename, 'w').write(imageset_str)


class DataIOCoco(IO):
    def __init__(self, json_filename, img_filename_func, image_size):
        self.INFO = {
            "description": "Example Dataset",
            "url": "https://github.com/waspinator/pycococreator",
            "version": "0.1.0",
            "year": 2018,
            "contributor": "waspinator",
            "date_created": datetime.datetime.utcnow().isoformat(' ')
        }
        self.LICENSES = [
            {
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License",
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
            }
        ]
        self.CATEGORIES = [
            {
                'id': 1,
                'name': 'ant',
                'supercategory': 'animal',
                'keypoints': ['head', 'tail'],
                'skeleton': [[2, 1]]
            },
            {
                'id': 2,
                'name': 'zebrafish',
                'supercategory': 'animal',
            },
            {
                'id': 3,
                'name': 'sowbug',
                'supercategory': 'animal',
            },
        ]
        self.coco_output = {
            "info": self.INFO,
            "licenses": self.LICENSES,
            "categories": self.CATEGORIES,
            "images": [],
            "annotations": []
        }
        class_id = [x['id'] for x in self.CATEGORIES if x['name'] == 'ant'][0]
        self.category_info = {'id': class_id, 'is_crowd': False}
        self.next_idx = 0
        self.json_filename = json_filename
        self.img_filename_func = img_filename_func
        self.image_size = image_size

    def add_item(self, data):
        if isinstance(data, dict):
            data = [data]
        assert isinstance(data, list)

        image_info = pycococreatortools.create_image_info(
            self.next_idx, self.img_filename_func(), self.image_size)
        self.coco_output["images"].append(image_info)

        segmentation_id = 1
        for ann in data:
            annotation_info = pycococreatortools.create_annotation_info(
                segmentation_id, self.next_idx, self.category_info, ann['mask'], self.image_size, tolerance=2,
                keypoints=np.array([[ann['p0_x'], ann['p0_y']], [ann['p1_x'], ann['p1_y']]]))
            assert annotation_info is not None
            self.coco_output["annotations"].append(annotation_info)
            segmentation_id += 1
            # for key, val in bbox.items():
            #     if key in ['xmin', 'xmax', 'ymin', 'ymax']:
        self.next_idx += 1

    def read_item(self):
        assert False, 'not implemented'

    # def write_imageset(self, out_filename, idx_range=None):
    #     if idx_range is None:
    #         idx_range = range(self.image_id)
    #     imageset_str = '\n'.join([splitext(basename(self.filename_template.format(idx=i)))[0] for i in idx_range])
    #     open(out_filename, 'w').write(imageset_str)

    def close(self):
        with open(self.json_filename, 'w') as output_json_file:
            json.dump(self.coco_output, output_json_file)


class Dataset(object):
    def __init__(self, image_io=None, data_io=None):
        """

        :param image_io: single or list of multiple ImageIO descendant(s) or None
        :param data_io: DataIO descendant or None
        """
        if image_io is None:
            self.image_ios = None
        else:
            if not isinstance(image_io, list):
                image_io = [image_io]
            self.image_ios = image_io
        self.data_io = data_io

    def is_dummy(self):
        return self.image_ios is None and self.data_io is None

    def add_item(self, image=None, data=None):
        if self.image_ios is not None:
            if self.data_io is not None:
                self.data_io.add_item(data)
            if not isinstance(image, list):
                image = [image]
            for image_io, single_img in zip(self.image_ios, image):
                image_io.add_item(single_img)

    def read_item(self):
        return self.image_ios.read_item(), self.data_io.read_item()

    def close(self):
        if self.image_ios is not None:
            for image_io in self.image_ios:
                image_io.close()
        if self.data_io is not None:
            self.data_io.close()

    @property
    def next_idx(self):
        if self.image_ios is not None and self.data_io is not None:
            assert self.image_ios[0].next_idx == self.data_io.next_idx
            return self.image_ios[0].next_idx
        elif self.image_ios is not None:
            return self.image_ios[0].next_idx
        elif self.data_io is not None:
            return self.data_io.next_idx
        else:
            assert False, 'image and data io is None'

    @next_idx.setter
    def next_idx(self, value):
        if self.data_io is not None:
            self.data_io.next_idx = value
        if self.image_ios is not None:
            for io in self.image_ios:
                io.next_idx = value
