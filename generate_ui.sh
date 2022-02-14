#!/bin/sh

pyuic6 gui/tracking_widget.ui -o gui/generated/ui_tracking_widget.py -x
pyuic6 gui/graph_widget.ui -o gui/generated/ui_graph_widget.py -x
pyuic6 gui/create_project_page.ui -o gui/generated/ui_create_project_page.py -x
pyuic6 gui/landing_tab.ui -o gui/generated/ui_landing_tab.py
