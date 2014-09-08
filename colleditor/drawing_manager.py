__author__ = 'flipajs'


class DrawingManager:
    # class which holds multiple EditablePixmaps and manages
    # which one will be used

    def __init__(self):
        self.layers = {}
        self.active_layer = None

    def add_layer(self, name, layer):
        self.layers[name] = layer
        self.layers[name].update_pixmap()

        if not self.active_layer:
            self.active_layer = name

    def remove_layer(self, name):
        try:
            del self.layers[name]
        except KeyError:
            pass

    def drawing_signal(self, pts, erase=False):
        try:
            layer = self.layers[self.active_layer]
            if erase:
                layer.remove_points(pts)
            else:
                layer.add_points(pts)

        except KeyError:
            pass

    def activate(self, name):
        self.active_layer = name

    def get_layer(self, name):
        return self.layers[name]