#This file is developed by Team 18 of COMP90024 of The University of Melbourne, under Apache Licence(see LICENCE).
#Researched Cities: Victoria, AU
#Team member - id:
#Yixiong Ding  671499     
#Yijie Mei     861351
#Tiange Wuang  903588
#Wuang Shen    716090
#Ruifeng Luo   686141

import TwitterAPI
from TwitterAPI import TwitterPager
#from TwitterPager2 import TwitterPager2
import json
import progress
from time import sleep
from util import *
import db


def try_get_iterator(pager):
    while True:
        try:
            ret=pager.get_iterator()
            return ret
        except TwitterAPI.TwitterError.TwitterRequestError,e:
            if e.status_code==429:
                print ("Too Many Requests, now sleeping...")
                sleep(60)
            else:
                raise e

'''
Get a page of tweets using search API of twitter. Will return when finish fetching the current page.
Return the count of tweets added from this page.
Params:
    api: the TwitterAPI object
    db: the couchdb wrapper object
    keyword_query: the key word for the parameter 'q' in the request
    geocode: the string that specifies the location we are interested
    from_id: from which tweet-id we want to fetch (max_id) - can be -1, and twitter will pick some id for us
    to_id: the smallest id we want to fetch (since_id)
    next_id: the from_id for the next page
'''
def do_search(api,db,keyword_query,geocode,from_id,to_id,next_id):
    #r = api.request('statuses/filter', {'locations': '112.5,-37.5,154.1,-12.8'})
    next_id=-1
    cur_id=-1
    if from_id==-1:
        from_id=None
    if to_id==-1:
        to_id=0
    count=0
    pager = TwitterPager(api, 'search/tweets', {'q': keyword_query, 'geocode': geocode,  'count': '100', 'max_id': str(from_id), 'since_id' : str(to_id)})
    while True:
        try:
            for item in pager.get_iterator():
                #print(item)
                if 'text' in item:
                    #try:
                    if True:
                        #print item["id"]
                        cur_id=int(item["id"])
                        #if next_id != -1, we run in re-start mode, don't reset next_id
                        #else we need to update next_id when the first item arrives in this iteration
                        #and next iteration's to_id will be set to next_id of this iteration 
                        if next_id==-1:
                            next_id=cur_id
                        if cur_id<=to_id:
                            break
                        info=get_dict_object_from_tweet(item)
                        if not info:
                            print "Error parsing the tweet, ignore it"
                            continue
                        #put the data in the db
                        db.put(info)
                        count+=1
                        if count % 1000 == 0:
                            print count
                        #print item["id"],"ok"
                        #print(info["post_text"])
                    #persist the progress to ensure we can resume the harvester from here
                    progress.update(cur_id,to_id,next_id)
                elif 'message' in item:
                    # something needs to be fixed before re-connecting
                    raise Exception(item['message'])
            return count
        except TwitterAPI.TwitterError.TwitterRequestError,e:
            if e.status_code==429:
                print ("Too Many Requests, now sleeping...")
                sleep(60)
            else:
                raise e

'''
Get tweets using search API of twitter. Will fetch pages by pages, from old ones to new ones.
Within each page, it will first fetch new tweets, then old tweets.
Params:
    api: the TwitterAPI object
    db: the couchdb wrapper object
    keywords: list of the key words
    geocode: the string that specifies the location we are interested
'''
def search(api,db,keywords,geocode):
    #join the keywords for the query
    keyword_query = " OR ".join(keywords)
    #continue the unfinished progress from from_id to to_id
    from_id,to_id,next_id=progress.get()
    c=do_search(api,db,keyword_query,geocode,from_id,to_id,next_id)
    print "iteration done. added", c ,"tweets"
    while True:
        from_id,to_id,next_id=progress.get()
        #the previous page is fetched, we now start a new page with to_id "next_id"
        c=do_search(api,db,keyword_query,geocode,None,next_id,-1)
        print "iteration done. added", c ,"tweets"
        if c==0:
            #if no tweets is fetch, we sleep for a while to wait for new tweets
            print("No tweets found, sleeping...")
            sleep(60)
        print "F",from_id,"T",to_id,"N",next_id

