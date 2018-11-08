from __future__ import print_function
from core.region.region import Region, encode_RLE
import sqlite3 as sql
import cPickle as pickle
from core.region.region import encode_RLE
from core.region import region_manager as RM
import numpy as np

dbDir = '/home/casillas/Documents/analysisColonies/data/F1C51min/'
#dbDir ='/home/casillas/Documents/COLONIESWD/S1T1/'
dbName = 'rm.sqlite3'

outputDBName = 'IRF'+dbName

#This starts the regionManager
myRM = RM.RegionManager(db_wd=dbDir,db_name=dbName)
np.set_printoptions(threshold=np.nan)  #This is to tell numpy to print the whole array with out any sort of ellipsis.



#create database
con = sql.connect(dbDir+"/"+outputDBName)
cur = con.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS regions(\
		id INTEGER PRIMARY KEY, frame INTEGER,\
		regionX TEXT, regionY TEXT);")
cur.execute("CREATE INDEX IF NOT EXISTS regions_index ON regions(id);")
cur.execute("CREATE INDEX frameIdx ON regions(frame);")  #Creates second index. This enables fast querying of the database to retrieve all regions alive in a given frame

#for every region, save it's id, frame and points as two text fields, one for X one for Y coordinate
for Rid in range(1,len(myRM)):
	reg = myRM[Rid];
	pointsX = reg.pts()[:,0]
	pointsY = reg.pts()[:,1]
	Rfr = reg.frame_;
	Rx = np.array2string(pointsX,max_line_width=np.Inf)[1:-1]
	Ry = np.array2string(pointsY,max_line_width=np.Inf)[1:-1]
	cur.execute("INSERT INTO regions VALUES (?,?,?,?);", (Rid,Rfr,Rx,Ry))

	if (Rid % 100) == 0:
		con.commit()

con.commit()

print("converted a total of "+str(len(myRM))+" regions")
