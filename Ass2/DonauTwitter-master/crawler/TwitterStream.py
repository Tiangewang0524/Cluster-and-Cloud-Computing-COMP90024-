#This file is developed by Team 18 of COMP90024 of The University of Melbourne, under Apache Licence(see LICENCE).
#Researched Cities: Victoria, AU
#Team member - id:
#Yixiong Ding  671499     
#Yijie Mei     861351
#Tiange Wuang  903588
#Wuang Shen    716090
#Ruifeng Luo   686141

import TwitterAPI
from util import *

'''
filter the twitter stream
Param:
    api: the TwitterAPI object
    db: the couchdb wrapper object
    keywords: the list of keywords
    location: the string of location coordinates boxes
'''
def search(api,db,keywords,locations):
    count=0
    query=",".join(keywords)
    
    #construct the request to twitter
    request_map={'locations': locations}
    if query=="":
        print "empty keywords"
    else:
        #if the keywords are not empty, set it in the request
        request_map['track']=query
    while True:
        try:
            r = api.request('statuses/filter', request_map)
            for item in r:
                #for each tweets in the stream, construct the dict data for our db
                info=get_dict_object_from_tweet(item)
                if not info:
                    print "Error parsing the tweet, ignore it"
                    continue
                
                db.put(info)
                count+=1
                #print item["text"]
                if count % 200 ==0:
                    print count
        except TwitterAPI.TwitterError.TwitterRequestError as e:
            if e.status_code < 500:
                # something needs to be fixed before re-connecting
                raise
            else:
                # temporary interruption, re-try request
                pass
        except TwitterAPI.TwitterError.TwitterConnectionError:
            # temporary interruption, re-try request
            pass



