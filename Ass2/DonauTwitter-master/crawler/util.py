#This file is developed by Team 18 of COMP90024 of The University of Melbourne, under Apache Licence(see LICENCE).
#Researched Cities: Victoria, AU
#Team member - id:
#Yixiong Ding  671499     
#Yijie Mei     861351
#Tiange Wuang  903588
#Wuang Shen    716090
#Ruifeng Luo   686141

'''
get the leaf node using the path
"item" is a tree of dict. "path" is an array of names
e.g. get_data({1:{2:{3:{"DATA"}}}},[1,2,3]) will return "DATA"
If the path is not valid, return None
'''
def get_data(item,path):
    itr=item
    for idx in path:
        if itr and itr.has_key(idx):
            itr=itr[idx]
        else:
            return None
    return itr

'''
set info[key]=data if data!=None
'''
def set_if_not_none(info,key,data):
    if data:
        info[key]=data

'''
build a dict for output from the raw data got from twitter
returned dict will contain: _id, user_id, post_text, location, location_fullname, coodinates
'''
def get_dict_object_from_tweet(item):
    info = dict()
    
    if 'id' in item:
        info["_id"] = str(item["id"])
    else:
        return None
    d=get_data(item,["user","id"])
    if d:
        info["user_id"] = d
    else:
        return None

    if 'text' in item:
        info["post_text"] = item['text']
    else:
        return None
    set_if_not_none(info,"location", get_data(item,["place","name"]) )
    set_if_not_none(info,"location_fullname", get_data(item,["place","full_name"]) )
    c=get_data(item,["coordinates","coordinates"])
    if c:
        info["coodinates"]=c
        #print c
    set_if_not_none(info,"time", get_data(item,["created_at"]))
    return info