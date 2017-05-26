__author__ = 'flipajs'

import time
from core.settings import Settings as S_


class LogCategories:
    GRAPH_EDIT = 0
    USER_ACTION = 1
    OTHERS = 2
    DEBUG_INFO = 3

    @staticmethod
    def get_name(val):
        if val == LogCategories.GRAPH_EDIT:
            return 'graph edit'
        elif val == LogCategories.USER_ACTION:
            return 'user action'
        elif val == LogCategories.OTHERS:
            return 'others'
        elif val == LogCategories.DEBUG_INFO:
            return 'debug info'

class ActionNames:
    # GRAPH_EDIT
    ADD_NODE = 'add_node'
    REMOVE_NODE = 'remove_node'
    ADD_EDGE = 'add_edge'
    REMOVE_EDGE = 'remove_edge'
    DISASSEMBLE_CHUNK = 'disassemble_chunk'
    ASSEMBLE_CHUNK = 'assemble_chunk'
    MERGE_CHUNKS = 'merge_chunks'
    CHUNK_APPEND_LEFT = 'chunk_append_left'
    CHUNK_APPEND_RIGHT = 'chunk_append_right'
    CHUNK_POP_FIRST = 'chunk_pop_first'
    CHUNK_POP_LAST = 'chunk_pop_last'
    CHUNK_ADD_TO_REDUCED = 'chunk_add_to_reduced'
    CHUNK_REMOVE_FROM_REDUCED = 'chunk_remove_from_reduced'
    CHUNK_SET_START = 'chunk_set_start'
    CHUNK_SET_END = 'chunk_set_end'
    IGNORE_NODE = 'ignore_node'

    # USER ACTION
    IGNORE_NODES = 'ignore_nodes'
    CONFIRM_ALL = 'confirm_all'
    CONFIRM = 'confirm'
    MERGED_REGION = 'merged_region'
    JOIN_REGIONS = 'join_regions'
    REMOVE = 'remove'
    STRONG_REMOVE = 'strong_remove'
    NEW_REGION = 'new_region'
    MERGED_SELECTED = 'merged_selected'
    JOIN_CHUNKS = 'join_chunks'
    MARK_JOIN = 'mark_join'
    MARK_SPLIT = 'mark_split'
    MARK_JOIN_AND_SPLIT = 'mark_join_and_split'

class LogEntry:
    def __init__(self, category, action_name, data=None):
        self.category = category
        self.action_name = action_name
        self.time = time.time()
        self.data = data

    def __str__(self):
        s = "Log entry t: \t" + str(self.time)
        s += "\n\tcategory: " + LogCategories.get_name(self.category)
        s += "\n\tname: " + self.action_name + "\n\t data:" + str(self.data)

        return s

class Log:
    def __init__(self):
        self.data_ = []

    def add(self, category, action_name, data=None):
        if category == LogCategories.GRAPH_EDIT and not S_.general.log_graph_edits:
            return

        if S_.general.print_log:
            print category, action_name, data

        self.data_.append(LogEntry(category, action_name, data))

    def pop_last_user_action(self):
        actions = []
        while self.data_:
            a = self.data_.pop()
            if a.category == LogCategories.USER_ACTION:
                return actions
            if a.category == LogCategories.GRAPH_EDIT:
                actions.append(a)

        return actions