__author__ = 'flipajs'

import time


class LogCategories:
    GRAPH_EDIT = 0
    USER_ACTION = 1
    OTHERS = 2
    DEBUG_INFO = 3


class ActionNames:
    # GRAPH_EDIT
    ADD_NODE = 'add_node'
    REMOVE_NODE = 'remove_node'
    ADD_EDGE = 'add_edge'
    REMOVE_EDGE = 'remove_edge'

    # USER ACTION
    CONFIRM_ALL = 'confirm_all'
    MERGED_REGION = 'merged_region'
    JOIN_TOGETHER = 'join_together'
    REMOVE = 'remove'
    STRONG_REMOVE = 'strong_remove'


class Log:
    def __init__(self):
        self.data_ = []

    def add(self, category, name, data):
        self.data_.append([category, name, time.time(), data])
