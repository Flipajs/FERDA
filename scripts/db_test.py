import sqlite3 as sql
import json
import time
import numpy as np


class DB_Test():
    def __init__(self, db_path, db_path_json, pts_size=10, reg_size=1000):
        self.db_path = db_path
        self.db_path_json = db_path_json
        self.pts_size = pts_size
        self.reg_size = reg_size
        try:
            self.cur = sql.connect(self.db_path).cursor()
            self.cur_json = sql.connect(self.db_path_json).cursor()
        except:
            pass

    def test(self):
        self.create_db()
        self.select()
        self.create_db_json()
        self.select_json()

    def create_db(self):
        print "Creating new db..."
        self.cur.execute("DROP TABLE IF EXISTS Regions")
        self.cur.execute("DROP TABLE IF EXISTS Points")
        self.cur.execute("CREATE TABLE Regions(Id INT, Area INT)")
        self.cur.execute("CREATE TABLE Points(RegionId INT, X INT, Y INT, PRIMARY KEY(RegionId, X, Y))")
        self.cur.execute("CREATE INDEX RegionIndex ON Points(RegionId);")
        reg_values = np.random.randint(0, 100, self.reg_size)

        t = time.time()
        self.cur.executemany("INSERT INTO Regions VALUES((?), (?));", get_regions_data(reg_values))
        self.cur.executemany("INSERT INTO Points VALUES((?),(?),(?));", get_points_data(reg_values, self.pts_size))
        print "Done (%ss)" % (time.time() - t)

    def create_db_json(self):
        print "Creating new db with json data..."
        self.cur_json.execute("DROP TABLE IF EXISTS Regions")
        self.cur_json.execute("CREATE TABLE Regions(id INT, area INT, data STRING, PRIMARY KEY (id))")
        self.cur_json.execute("CREATE INDEX RegionIndex ON Regions(id);")
        reg_values = np.random.randint(0, 100, self.reg_size)
        pts_values = np.random.randint(0, 100, self.pts_size*2)

        t = time.time()
        self.cur_json.executemany("INSERT INTO Regions VALUES((?), (?), (?));", get_regions_data_json(reg_values, pts_values))
        print "Done (%ss)" % (time.time() - t)

    def select(self):
        t = time.time()
        for i in range(0, 200):
            self.cur.execute("SELECT Regions.*, Points.* FROM Regions LEFT JOIN Points WHERE Regions.Id = %s AND Points.RegionId=Regions.Id;" % (3*i))
            rows = self.cur.fetchall()
            for row in rows:
                #print "Region id: %s, Region area: %s, Point region: %s, X: %s, Y: %s" % (row[0], row[1], row[2], row[3], row[4])
                pass
        print "Time taken to SELECT from db: %s" % ((time.time() - t))

    def select_json(self):
        t = time.time()
        for i in range(0, 200):
            str = "SELECT * FROM Regions WHERE id LIKE %s;" % (i*3)
            self.cur_json.execute(str)

            rows = self.cur_json.fetchall()
            for row in rows:
                #print "Id %s, area: %s, json: %s" % (row[0], row[1], row[2])
                pass
        print "Time taken to SELECT from db_json: %s)" % (time.time() - t)


def get_regions_data_json(area_array, pts_array):
    # id, area, [{x:?, y:?}{x:?, y:?}{...}...]
    for i in range(0, len(area_array)):
        id = i
        area = area_array[i]
        str = "["
        for j in range(0, len(pts_array), 2):
            str += "{x:%d, y:%d}" % (pts_array[j], pts_array[j+1])
        str += "]"
        if i%10000 == 0:
            print i
        yield(id, area, json.dumps(str))

def get_regions_data(array):
    for i in range(0, len(array)):
        yield (i, array[i])

def get_points_data(array, limit):
    k = 0
    for i in range(0, len(array), 2):
        #print i/2
        if(i/2 % limit == 0):
            k += 1
        yield (k, i, array[i+1])

tester = DB_Test("/home/dita/PycharmProjects/db1.db", "/home/dita/PycharmProjects/db2.db")
tester.test()