#This file is developed by Team 18 of COMP90024 of The University of Melbourne, under Apache Licence(see LICENCE).
#Researched Cities: Victoria, AU
#Team member - id:
#Yixiong Ding  671499     
#Yijie Mei     861351
#Tiange Wuang  903588
#Wuang Shen    716090
#Ruifeng Luo   686141

import json
from TwitterAPI import TwitterAPI
import TwitterSearch
import TwitterStream
import db
import sys

#load the configuration file
data = json.load(open('config.json'))

#load variables from json
consumer_key = data["consumer_key"]
consumer_secret = data["consumer_secret"]
access_token_key = data["access_token_key"]
access_token_secret = data["access_token_secret"]
keywords = data["keywords"]

locations = data['locations']
geocode = data['geocode']

#create db connection
db=db.dbclient(data['db_url'],data['db_name'])

#create TwitterAPI object
api = TwitterAPI(consumer_key,
                 consumer_secret,
                 access_token_key,
                 access_token_secret)
if sys.argv[1]=='s':
	TwitterSearch.search(api,db,keywords,geocode)
elif sys.argv[1]=='f':
	TwitterStream.search(api,db,keywords,locations)
else:
	print "Command not recognized"
