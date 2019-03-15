#This file is developed by Team 18 of COMP90024 of The University of Melbourne, under Apache Licence(see LICENCE).
#Researched Cities: Victoria, AU
#Team member - id:
#Yixiong Ding  671499     
#Yijie Mei     861351
#Tiange Wuang  903588
#Wuang Shen    716090
#Ruifeng Luo   686141

import pickle
class Area:
	def __init__(self,name,totalCount,posCount,negCount):
		self.name = name
		self.totalCount = totalCount
		self.posCount = posCount
		self.negCount = negCount
with open('locationDict.pkl', 'rb') as f:
	output = pickle.load(f)
for key,value in output.items():
	if key!='__progress__':
		print(key,value.totalCount,value.negCount,value.posCount)