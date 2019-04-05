import csv
import cv2
import numpy as np
from abc import ABCMeta, abstractmethod


class IO:
    __metaclass__ = ABCMeta

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

    def add_item(self, image):
        cv2.imwrite(self.filename_template.format(idx=self.next_idx), image)
        self.next_idx += 1

    def read_item(self):
        return cv2.imread(self.filename_template.format(idx=self.next_idx))
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
        self.template = '<annotation><folder>GeneratedData_Train</folder>' \
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
                 '<segmented>0</segmented>' \
                 '{annotation}</annotation>'
        self.bbox_template = '<object><name>{name}</name><bndbox>' \
                             '<xmin>{xmin}</xmin><xmax>{xmax}</xmax>' \
                             '<ymin>{ymin}</ymin><ymax>{ymax}</ymax></bndbox>{points}</object>'
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
            point_annotations = self.point_template.format(id=0, x=bbox['p0_x'], y=bbox['p0_y']) + \
                                self.point_template.format(id=1, x=bbox['p1_x'], y=bbox['p1_y'])
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
            if not isinstance(image, list):
                image = [image]
            for image_io, single_img in zip(self.image_ios, image):
                image_io.add_item(single_img)
        if self.data_io is not None:
            self.data_io.add_item(data)

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
