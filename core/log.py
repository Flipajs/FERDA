__author__ = 'flipajs'

import time
import sqlite3 as sql
import cPickle as pickle
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


class LogEntry:
    def __init__(self, category, action_name, time=time.time(), data=None):
        self.category = category
        self.action_name = action_name
        self.time = time
        self.data = data

    def __str__(self):
        s = "Log entry t: \t" + str(self.time)
        s += "\n\tcategory: " + LogCategories.get_name(self.category)
        s += "\n\tname: " + self.action_name + "\n\t data:" + str(self.data)

        return s


class Log:
    def __init__(self, path):
        self.db_path = path+"/log.db"
        print "Initializing db at %s " % self.db_path
        self.cur = sql.connect(self.db_path, isolation_level=None).cursor()
        self.cur.execute("CREATE TABLE IF NOT EXISTS log(\
            id INTEGER PRIMARY KEY AUTOINCREMENT, \
            time TIMESTAMP DEFAULT (DATETIME('now')), \
            category TINYINT, \
            action STRING, \
            data BLOB, \
            active BOOLEAN);")
        self.cur.execute("CREATE INDEX IF NOT EXISTS log_index ON log(id, time, category, action);")
        print "Done!"

    def add(self, category, action_name, data=None):
        t = time.time()
        if category == LogCategories.GRAPH_EDIT and not S_.general.log_graph_edits:
            return

        if S_.general.print_log:
            print "c: %s, a: %s, d: %s" % (category, action_name, data)

        if data == None:
            data = ""
        else:
            data = pickle.dumps(data, -1)

        cmd = "INSERT INTO log (category, action, data, active) VALUES (?, ?, ?, 1);"

        try:
            self.cur.execute(cmd, (category, action_name, sql.Binary(data)))
        except sql.ProgrammingError:
            self.cur = sql.connect(self.db_path, isolation_level=None).cursor()
            self.cur.execute(cmd, (category, action_name, sql.Binary(data)))
        print time.time() - t


    def add_many(self, iter):
        t = time.time()
        cmd = "INSERT INTO log (category, action, data, active) VALUES (?, ?, ?, 1);"

        try:
            self.cur.executemany(cmd, iter)
        except sql.ProgrammingError:
            self.cur = sql.connect(self.db_path, isolation_level=None).cursor()
            self.cur.executemany(cmd, iter)
        print time.time() - t


    def pop_last_user_action(self):
        # TODO: change flag active to 0, fix udno function (gui/tracker/tracker_widget.py)
        # get_last_uset_action_cmd = "SELECT id FROM log WHERE category = 1 ORDER BY id DESC LIMIT 1"
        get_undo = "SELECT \
            (SELECT id FROM log WHERE category = 1 AND active = 1 ORDER BY id DESC LIMIT 1) as last,\
            * FROM log WHERE id >= last AND active = 1;"
        try:
            self.cur.execute(get_undo)
            rows = self.cur.fetchall()
        except sql.ProgrammingError:
            self.cur = sql.connect(self.db_path, isolation_level=None).cursor()
            self.cur.execute(get_undo)
            rows = self.cur.fetchall()

        actions = []
        for row in rows:
            # l_id    id      time    cat     act     data
            # row[0], row[1], row[2], row[3], row[4], row[5]
            data = str(row[5])

            print "Rollback data - category %s, action %s, time %s, data %s" % (row[3], row[4], row[2], pickle.loads(data))
            actions.append(LogEntry(row[3], row[4], data=pickle.loads(data), time=row[2]))

        cmd = "UPDATE log SET active = 0 WHERE id >= %s" % row[0]
        self.cur.execute(cmd)

        return actions