from PyQt6 import QtGui, QtWidgets
from gui.gui_utils import SelectableQLabel

__author__ = 'fnaiser'


class ImgGridWidget(QtWidgets.QWidget):
    def __init__(self, scrolling=True, cols=5, element_width=300):
        super(ImgGridWidget, self).__init__()

        self.scrolling = scrolling
        self.element_width = element_width
        self.cols = cols
        self.id = 0

        self.grid = QtWidgets.QGridLayout()
        self.grid.setSpacing(1)
        self.grid.setContentsMargins(0, 0, 0, 0)

        self.grid_widget = QtWidgets.QWidget()
        self.grid_widget.setLayout(self.grid)

        self.items = []

        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().setSpacing(0)
        self.layout().setMargin(0)

        self.layout().addWidget(self.grid_widget)
        if self.scrolling:
            self.scroll_ = QtWidgets.QScrollArea()
            self.scroll_.setWidget(self.grid_widget)
            self.scroll_.setWidgetResizable(True)
            self.set_width_()

            self.layout().addWidget(self.scroll_)

    def _widget(self, item):
        if issubclass(type(item), SelectableQLabel):
            return item
        else:
            return item.widget

    def reshape(self, cols, element_width=100):
        self.cols = cols
        self.element_width = element_width

        grid2 = QtWidgets.QGridLayout()

        for i, item in enumerate(self.items):
            row = i / cols
            col = i % cols
            grid2.addWidget(self._widget(item), row, col)

        QtWidgets.QWidget().setLayout(self.grid)
        self.grid_widget.setLayout(grid2)
        self.grid = grid2
        self.grid.setSpacing(1)
        self.grid.setContentsMargins(0, 0, 0, 0)

        self.set_width_()

    def add_item(self, item, append=True):
        """
        Add items to the grid.

        :param item: SelectableQLabel subclass or object with object.widget subclass of SelectableQLabel
        :param append: if False, add item only to grid widget not to list of items
        """
        if append:
            self.items.append(item)

        row = self.id / self.cols
        col = self.id % self.cols

        self.grid.addWidget(self._widget(self.items[self.id]), row, col)

        self.set_width_()

        self.id += 1

    def delete_item(self, item):
        try:
            self.items.remove(item)
        except ValueError:
            pass

        self.redraw()

    def delete_all(self):
        self.items = []
        self.redraw()

    def redraw(self):
        # remove all widgets
        for i in reversed(list(range(self.grid.count()))):
            widget_to_remove = self.grid.itemAt(i).widget()
            # remove it from the layout list
            self.grid.removeWidget(widget_to_remove)
            # remove it from the gui
            widget_to_remove.setParent(None)

        self.id = 0
        for item in self.items:
            self.add_item(item, False)

    def set_width_(self):
        w = self.cols*self.element_width + (self.cols + 1)
        if self.scrolling:
            vscroll_shint = self.scroll_.verticalScrollBar().sizeHint()
            self.scroll_.setFixedWidth(w + vscroll_shint.width())
        else:
            self.setMaximumWidth(w)
            self.setMinimumWidth(w)

    def get_selected(self):
        return [item.id_ for item in self.items if self._widget(item).selected]

    def get_selected_items(self):
        return [item for item in self.items if self._widget(item).selected]

    def get_unselected(self):
        return [item.id_ for item in self.items if not self._widget(item).selected]

    def get_unselected_items(self):
        return [item for item in self.items if not self._widget(item).selected]

    def swap_selection(self):
        for item in self.items:
            self._widget(item).set_selected(not self._widget(item.selected))

    def deselect_all(self):
        for item in self.items:
            self._widget(item).set_selected(False)

    def select_all(self):
        for item in self.items:
            self._widget(item).set_selected(True)

    def select_all_until_first(self):
        for item in self.items:
            if not self._widget(item.selected):
                self._widget(item).set_selected(True)
            else:
                break
