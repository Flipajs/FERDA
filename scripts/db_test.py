import sqlite3 as lite
import json
import random
import numpy
import string
import sys


def points(array, limit):
    k = 0
    for i in range(0, len(array), 2):
        if(k % limit == 0):
            k = 0
        #print array[i]
        yield (i/2, k, array[i], array[i+1])
        k += 1

def regions(array):
    for i in range(0, len(array)):
        yield (i, array[i])

con = lite.connect('/home/dita/PycharmProjects/regions.db')

with con:

    cur = con.cursor()
    cur.execute('SELECT SQLITE_VERSION()')

    data = cur.fetchone()

    print "SQLite version: %s" % data

    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS Regions")
    cur.execute("DROP TABLE IF EXISTS Points")
    cur.execute("CREATE TABLE Regions(Id INT, Area INT)")
    cur.execute("CREATE TABLE Points(Id INT, Region INT, X INT, Y INT)")

    regions_size = 100000
    points_size = 100

    reg_values = numpy.random.randint(0, 100, regions_size)

    import time
    t = time.time()
    cur.executemany("INSERT INTO Regions VALUES((?), (?));", regions(reg_values))

    str = "INSERT INTO Points VALUES((?),(?),(?),(?));"
    cur.executemany(str, points(reg_values, points_size))
    print time.time() - t

    #cur.execute("SELECT Region, Regions.Area, X, Y FROM Points LRegions WHERE Points.Region=Regions.Id AND Regions.Id LIKE 890;")
    cur.execute("SELECT Regions.Id, Regions.Area, Points.X, Points.Y FROM Regions LEFT JOIN Points WHERE Regions.Id < 890 AND Points.Region=Regions.Id;")

    rows = cur.fetchall()


    print "Points: "
    for row in rows:
        pass
        #print "Region %s (Area: %s): [%s,%s]" % (row[0], row[1], row[2], row[3])
        #print "Region %s Area: %s, %s, %s" % (row[0], row[1], row[2], row[3])



    """
    cur.execute("CREATE TABLE Cars(Id INT, Name TEXT, Price INT)")
    cur.execute("INSERT INTO Cars VALUES(1,'Audi',52642)")
    cur.execute("INSERT INTO Cars VALUES(2,'Mercedes',57127)")
    cur.execute("INSERT INTO Cars VALUES(3,'Skoda',9000)")
    cur.execute("INSERT INTO Cars VALUES(4,'Volvo',29000)")
    cur.execute("INSERT INTO Cars VALUES(5,'Bentley',350000)")
    cur.execute("INSERT INTO Cars VALUES(6,'Citroen',21000)")
    cur.execute("INSERT INTO Cars VALUES(7,'Hummer',41400)")
    cur.execute("INSERT INTO Cars VALUES(8,'Volkswagen',21600)")

    cur.execute("SELECT Name, Price FROM Cars WHERE Price > 30000")

    rows = cur.fetchall()

    print "Cars more expensive than 30000:"
    for row in rows:
        print "%s cost's %s" % (row[0], row[1])

    print json.dump("Hello", "world")
    """