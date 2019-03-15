#This file is developed by Team 18 of COMP90024 of The University of Melbourne, under Apache Licence(see LICENCE).
#Researched Cities: Victoria, AU
#Team member - id:
#Yixiong Ding  671499     
#Yijie Mei     861351
#Tiange Wuang  903588
#Wuang Shen    716090
#Ruifeng Luo   686141

import pickle
import couchdb_map
import os.path
from nltk.sentiment.vader import SentimentIntensityAnalyzer
class Area:
    def __init__(self,name,totalCount,posCount,negCount):
        self.name = name
        self.totalCount = totalCount
        self.posCount = posCount
        self.negCount = negCount
        # self.languageDict = languageDict
# with open('output.pkl', 'rb') as f:
#   output = pickle.load(f)
# locationDict = {}
# for item in output[:1000]:
#   try:
#       location = item['location']
#       if not location in locationDict.keys():
#           locationDict[location] = Area(name = location, totalCount = 0, posCount = 0, negCount = 0)
#       locationDict[location].totalCount += 1
#       sentence = item['post_text']
#       if sid.polarity_scores(sentence)['neg'] - sid.polarity_scores(sentence)['pos'] > 0:
#           locationDict[location].negCount += 1
#       elif sid.polarity_scores(sentence)['neg'] - sid.polarity_scores(sentence)['pos'] < 0:
#           locationDict[location].posCount += 1
#   except:
#       pass

locationDict= {}



def testfunc(item,batch_id):
    global locationDict
    sid = SentimentIntensityAnalyzer()
    if True:
        location = item['location']
        if not location in locationDict.keys():
            locationDict[location] = Area(name = location, totalCount = 0, posCount = 0, negCount = 0)
        locationDict[location].totalCount += 1
        sentence = item['post_text']
        if sid.polarity_scores(sentence)['neg'] - sid.polarity_scores(sentence)['pos'] > 0:
            locationDict[location].negCount += 1
        elif sid.polarity_scores(sentence)['neg'] - sid.polarity_scores(sentence)['pos'] < 0:
            locationDict[location].posCount += 1
    #except:
    else:
        pass

def batch_done(batch_id):
    global locationDict
    output = open('locationDict.pkl','wb')
    locationDict['__progress__']=batch_id
    pickle.dump(locationDict,output,protocol=0)
    output.close()



mapfunc='''
function(doc){
    if(doc.location!=undefined){
        emit(doc._id)
    }
}
'''

if os.path.isfile("locationDict.pkl"):
    with open('locationDict.pkl', 'rb') as f:
        locationDict = pickle.load(f)
        prog = locationDict['__progress__']
        couchdb_map.map2(testfunc,'http://admin:admin@localhost:5984/',"tweets","locations","view",mapfunc,recompute=False,batch_done=batch_done,batch_start=prog+1,batch_size=200)
else:
    couchdb_map.map2(testfunc,'http://admin:admin@localhost:5984/',"tweets","locations","view",mapfunc,recompute=True,batch_done=batch_done,batch_size=200)


