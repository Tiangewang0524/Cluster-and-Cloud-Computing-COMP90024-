#This file is developed by Team 18 of COMP90024 of The University of Melbourne, under Apache Licence(see LICENCE).
#Researched Cities: Victoria, AU
#Team member - id:
#Yixiong Ding  671499     
#Yijie Mei     861351
#Tiange Wuang  903588
#Wuang Shen    716090
#Ruifeng Luo   686141

import couchdb
'''
map all documents in the db of the url with the map function
Params:
    func: the function to process the documents, the parameter of the func is a dict of a document. The func must
        return True if it wants to update the document
    url: the url of the db
    dbname: the name of the db
    batch_size: the size of one batch of iteration
'''
def map(func,url,dbname,batch_size=5000,batch_done=None):
    server = couchdb.Server(url=url)
    db = server[dbname]
    count = 0

    doc_count = db.info()["doc_count"]
    batches = doc_count/batch_size
    if doc_count % batch_size!=0 :
        batches+=1

    for i in range(batches):
        for row in db.view("_all_docs", skip=i*batch_size,limit=batch_size):
            data = db.get(row.id)
            func(data,i)
        if batch_done:
            batch_done(i)
        print "Batch", i ,"done"
        

'''
create a view in couchdb using the JS map function, and iterate the view with a python function "func"
Params:
    func: the function to process the documents, the parameter1 of the func is a dict of a document. the parameter 2
        of the func is the batch_id
    url: the url of the db
    dbname: the name of the db
    design_doc_name: design document name, used for id:_design/{design_doc_name}
    view_name: the view name
    mapfunc: a string of JavaScript function for the map function in couchdb
    recompute: whether to delete the old design document. If false, the parameter "mapfunc" is ignored
    batch_size: the size of one batch of iteration
    batch_start: start fetching the documents from which batch?
    batch_done: will be called when a batch is done if not None
'''
def map2(func,url,dbname,design_doc_name,view_name,mapfunc,recompute=False,batch_size=5000,batch_start=0,batch_done=None):
    server = couchdb.Server(url=url)
    db = server[dbname]
    if recompute:
        #fetch and delete old design doc
        olddoc=db.get("_design/"+design_doc_name)
        if olddoc:
            db.delete(olddoc)
        #make and save the new doc
        design_doc={'_id': "_design/"+design_doc_name, 'views': {view_name: {'map': mapfunc, 'reduce': '_count'}}}
        db.save(design_doc)
    #get the number of rows in total
    while True:
        try:
            countrow=db.view(design_doc_name+"/"+view_name)
            doc_count = 0
            for r in countrow:
                doc_count = r.value
            break
        except couchdb.http.ServerError as e:
            #if is time out exception, ignore and try again
            if e.args[0][0]!=500:
                print e.args[0]
                raise e
            print "Time out, but we can wait...."
    print "In total",doc_count,"Documents"

    batches = doc_count/batch_size
    if doc_count % batch_size!=0 :
        batches+=1

#    for i in range(batch_start,batches):
#        for row in db.view(design_doc_name+"/"+view_name,reduce=False,include_docs=True, skip=i*batch_size,limit=batch_size):
#            data = row.doc
#            func(data,i)
#        if batch_done:
#            batch_done(i)
#        print "Batch", i ,"done"
    cnt=0
    i=batch_start
    for row in db.iterview(design_doc_name+"/"+view_name,batch_size,reduce=False,include_docs=True, skip=batch_start*batch_size):
        data = row.doc
        func(data,i)
        cnt+=1
        if cnt==batch_size:
            if batch_done:
                batch_done(i)
            print "Batch", i ,"done"
            i+=1
            cnt=0
    if cnt!=0:
        batch_done(i)  


'''
map all documents in the db of the url with the map function, and write to another db
Params:
    func: the function to process the documents, the parameter of the func is a dict of a document. The func should
     return a dict to insert a document or return None
    url: the url of the db
    dbname: the name of the source db
    outdbname: the output db name
    batch_size: the size of one batch of iteration
'''
# def map_new(func,url,dbname,outdbname,batch_size=5000):
#     count = 0
#     server = couchdb.Server(url=url)
#     db = server[dbname]
#     outdb= server[outdbname]

#     doc_count = db.info()["doc_count"]
#     batches = doc_count/batch_size
#     if doc_count % batch_size!=0 :
#         batches+=1

#     for i in range(batches):
#         for row in db.view("_all_docs", skip=i*batch_size,limit=batch_size):
#             data = db.get(row.id)
#             newdata = func(data)
#             outdb.save(newdata)
#             count += 1
#             if count >= 10:
#                 exit(0)
#         print "Batch", i ,"done"
#     