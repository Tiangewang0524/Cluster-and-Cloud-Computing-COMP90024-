#This file is developed by Team 18 of COMP90024 of The University of Melbourne, under Apache Licence(see LICENCE).
#Researched Cities: Victoria, AU
#Team member - id:
#Yixiong Ding  671499     
#Yijie Mei     861351
#Tiange Wuang  903588
#Wuang Shen    716090
#Ruifeng Luo   686141

import couchdb

class dbclient:
	'''
	Initialize the database client
	Params:
		url: the url of the couchdb server
		dbname: the name of the db
	'''
	def __init__(self,url,dbname):
		self.couch = couchdb.Server(url)
		self.db=self.couch[dbname]


	'''
	Put a json file in db. Will not throw a conflict exception
	Params:
		data: the dict that holds the data
	'''
	def put(self,data):
		try:
			self.db.save(data)
		except couchdb.http.ResourceConflict:
			pass

	'''
	Put a json file in db. Will throw a conflict exception
	Params:
		data: the dict that holds the data
	'''
	def save(self,data):
		self.db.save(data)
