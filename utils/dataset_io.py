import csv
import cv2
import numpy as np


class ImageIOFile(object):
    def __init__(self, filename_template):
        self.filename_template = filename_template
        self.next_idx = 0

    def add_item(self, image):
        cv2.imwrite(self.filename_template.format(idx=self.next_idx), image)
        self.next_idx += 1

    def read_item(self):
        return cv2.imread(self.filename_template.format(idx=self.next_idx))
        self.next_idx += 1

    def close(self):
        pass


class MultiImageIOFile(object):
    def __init__(self, filename_template, names):
        self.filename_template = filename_template
        self.names = names
        self.next_idx = 0

    def add_item(self, images):
        for name, image in zip(self.names, images):
            cv2.imwrite(
                self.filename_template.format(idx=self.next_idx, name=name),
                image)
        self.next_idx += 1

    def read_item(self):
        images = []
        for name in self.names:
            images.append(cv2.imread(self.filename_template.format(
                idx=self.next_idx, name=name)))
        self.next_idx += 1
        return images

    def close(self):
        pass


class ImageIOHdf5(object):
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


class MultiImageIOHdf5(object):
    def __init__(self, h5group, dataset_names, shapes):
        self.datasets = []
        if len(shapes) != len(dataset_names) or not isinstance(shapes, list):
            shapes = [shapes] * len(dataset_names)
        for name, shape in zip(dataset_names, shapes):
            self.datasets.append(h5group.create_dataset(name, shape, np.uint8))
        self.next_idx = 0

    def add_item(self, images):
        for dataset, image in zip(self.datasets, images):
            dataset[self.next_idx] = image
        self.next_idx += 1

    def read_item(self):
        images = [dataset[self.next_idx] for dataset in self.datasets]
        self.next_idx += 1
        return images

    def close(self):
        pass


class DataIOCSV(object):
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
        self.writer.writerow(data)

    def read_item(self):
        assert False

    def close(self):
        self.csv_file.close()


class DataIOVot(object):
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
                             '<ymin>{ymin}</ymin><ymax>{ymax}</ymax></bndbox></object>'
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
        annotation = self.template.format(annotation=''.join([self.bbox_template.format(**bbox) for bbox in data]),
                                          filename=self.image_filename_template.format(idx=self.next_idx),
                                          **self.template_data)
        open(self.filename_template.format(idx=self.next_idx), 'w').write(annotation)
        self.next_idx += 1

    def read_item(self):
        annotation = open(self.filename_template.format(idx=self.next_idx), 'w').read()
        # parse xml
        assert False

    def close(self):
        pass


class Dataset(object):
    def __init__(self, image_io, data_io):
        self.image_io = image_io
        self.data_io = data_io

    def add_item(self, image, data):
        self.image_io.add_item(image)
        self.data_io.add_item(data)

    def read_item(self):
        return self.image_io.read_item(), self.data_io.read_item()

    def close(self):
        self.image_io.close()
        self.data_io.close()

    @property
    def next_idx(self):
        assert self.image_io.next_idx == self.data_io.next_idx
        return self.image_io.next_idx


class DummyDataset(object):
    def add_item(self, image, data):
        pass

    def close(self):
        pass