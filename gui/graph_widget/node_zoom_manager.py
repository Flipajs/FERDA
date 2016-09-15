class NodeZoomManager():

    def __init__(self):
        self.toggled = []

    def show_zoom_all(self):
        for node in self.toggled:
            self.show_zoom(node)

    def show_zoom(self, node):
        if not node.toggled:
            self.toggled.append(node)
            node.toggle()

    def hide_zoom_all(self):
        for node in self.toggled:
            self.hide_zoom(node)

    def hide_zoom(self, node):
        if node.toggled:
            node.toggle()

    def add(self, item):
        self.toggled.append(item)

    def remove_all(self):
        self.hide_zoom_all()
        self.toggled = []

    def remove(self, item):
        self.toggled.remove(item)

    def clear(self):
        self.toggled = []


