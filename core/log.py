from PyQt4.QtCore import QThread

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
        self.con = sql.connect(self.db_path)
        self.cur = self.con.cursor()
        self.cur.execute("CREATE TABLE IF NOT EXISTS log(\
            id INTEGER PRIMARY KEY AUTOINCREMENT, \
            time INT, \
            category TINYINT, \
            action STRING, \
            data BLOB, \
            active BOOLEAN);")
        self.cur.execute("CREATE INDEX IF NOT EXISTS log_index ON log(id, time, category, action);")
        self.time = 0
        print "Current thread: %s" % int(QThread.currentThreadId())
        print "Done!"

    def add(self, category, action_name, data=None):
        # print "Current thread: %s" % int(QThread.currentThreadId())
        if category == LogCategories.GRAPH_EDIT and not S_.general.log_graph_edits:
            return

        if S_.general.print_log:
            print "c: %s, a: %s, d: %s" % (category, action_name, data)

        if data == None:
            data = ""
        else:
            data = pickle.dumps(data, -1)

        cmd = "INSERT INTO log (time, category, action, data, active) VALUES (?, ?, ?, ?, 1);"

        self.cur.execute(cmd, (int(time.time()), category, action_name, sql.Binary(data)))
        """
        try:
            self.cur.execute(cmd, (int(time.time()), category, action_name, sql.Binary(data)))
        except sql.ProgrammingError:
            print "new con"
            self.cur = sql.connect(self.db_path).cursor()
            self.begin()
            self.cur.execute(cmd, (int(time.time()), category, action_name, sql.Binary(data)))
        print time.time() - t
        """


    def add_many(self, iter):
        # print "Current thread: %s" % int(QThread.currentThreadId())
        cmd = "INSERT INTO log (time, category, action, data, active) VALUES (?, ?, ?, ?, 1);"

        self.cur.executemany(cmd, iter)
        """
        try:
            self.cur.executemany(cmd, iter)
        except sql.ProgrammingError:
            print "new con (many)"
            self.cur = sql.connect(self.db_path).cursor()
            self.begin()
            self.cur.executemany(cmd, iter)
        print time.time() - t
        """


    def begin(self):
        print "Current thread: %s" % int(QThread.currentThreadId())
        self.time = time.time()
        print "Begining transaction in thread %s" % int(QThread.currentThreadId())
        self.cur.execute("BEGIN TRANSACTION")
        """
        try:
            self.cur.execute("BEGIN TRANSACTION")
        except sql.ProgrammingError:
            print "Creating new SQLite object"
            self.cur = sql.connect(self.db_path).cursor()
            self.cur.execute("BEGIN TRANSACTION")
        """

    def end(self):
        print "Ending transaction in thread %s (%ss total)" % (int(QThread.currentThreadId()), time.time() - self.time)
        self.con.commit()

        """
        try:
            self.cur.execute("END TRANSACTION")
        except sql.ProgrammingError:
            print "Creating new SQLite object"
            self.cur = sql.connect(self.db_path).cursor()
            self.cur.execute("END TRANSACTION")
        """


    def pop_last_user_action(self):
        # TODO: when project is closed without saving and re-opened, undo can recieve invalid data (nothing got undo-ed
        # TODO:   while testing this, but it might behave unexpectedly in some cases).
        get_undo = "SELECT \
            (SELECT id FROM log WHERE category = 1 AND active = 1 ORDER BY id DESC LIMIT 1) as last,\
            * FROM log WHERE id >= last AND active = 1 ORDER BY id DESC;"
        try:
            self.cur.execute(get_undo)
            rows = self.cur.fetchall()
        except sql.ProgrammingError:
            self.cur = sql.connect(self.db_path).cursor()
            self.cur.execute(get_undo)
            rows = self.cur.fetchall()

        actions = []
        for row in rows:
            # l_id    id      time    cat     act     data
            # row[0], row[1], row[2], row[3], row[4], row[5]
            data = str(row[5])

            print "Rollback data - category %s, action %s, time %s, data %s" % (row[3], row[4], row[2], pickle.loads(data))
            actions.append(LogEntry(row[3], row[4], data=pickle.loads(data), time=row[2]))

        if len(rows) > 0:
            cmd = "UPDATE log SET active = 0 WHERE id >= %s" % row[0]
            self.cur.execute(cmd)
            print "Undo successful"
        else:
            print "Sorry, can't undo any more"

        return actions