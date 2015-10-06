__author__ = 'fnaiser'


from core.arena.model import Model

class PaintMask(Model):
    def __init__(self, im_height, im_width):
        super(PaintMask, self).__init__(im_height, im_width)

    def set_mask(self, data):
        self.mask_ = data
        data.dtype = "uint8"

        self.mask_idx_ = (self.mask_ == 0)