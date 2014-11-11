__author__ = 'filip@naiser.cz'

from numpy import *
import cv2
import math

class CropSaver():
    def __init__(self):
        self.out_path = '/home/flipajs/~dump/crop/'
        self.crop_size = -1
        self.increase_factor = 1.8

    def estimate_crop_size(self, ants, increase_factor = -1):
        max_a = 0
        for a in ants:
            x = a.state.head.x - a.state.back.x
            y = a.state.head.y - a.state.back.y
            d = math.sqrt(x*x + y*y)

            if d > max_a:
                max_a = d

        if increase_factor > -1:
            max_a *= increase_factor
        else:
            max_a *= self.increase_factor

        self.crop_size = int(math.ceil(max_a))
        if self.crop_size%2 == 1:
            self.crop_size += 1

        print self.crop_size

    def save(self, ants, img, frame_num):
        border = self.crop_size
        img_ = zeros((shape(img)[0] + 2 * border, shape(img)[1] + 2 * border, 3), dtype=uint8)
        img_[border:-border, border:-border] = img.copy()
        collection = zeros((self.crop_size, self.crop_size * len(ants), 3), dtype=uint8)

        for ant_id in range(len(ants)):
            ant = ants[ant_id]

            x = ant.state.position.x
            y = ant.state.position.y

            img_small = img_[border + y - self.crop_size / 2:border + y + self.crop_size / 2,
                             border + x - self.crop_size / 2:border + x + self.crop_size / 2].copy()

            collection[:, ant_id*self.crop_size:(ant_id+1)*self.crop_size, :] = img_small


        cv2.imwrite(self.out_path+str(frame_num)+".png", collection)

    def visualize(self, ants, img, frame_num):
        img_ = img.copy()

        for ant_id in range(len(ants)):
            ant = ants[ant_id]

            x = ant.state.position.x
            y = ant.state.position.y

            cv2.rectangle(img_, (int(x - self.crop_size / 2), int(y - self.crop_size / 2)),
                          (int(x + self.crop_size / 2), int(y + self.crop_size / 2)), (255, 0, 0), 0)


        cv2.imshow("ant_crops", img_)
        cv2.imwrite(self.out_path+str(frame_num)+"_crops.png", img_)