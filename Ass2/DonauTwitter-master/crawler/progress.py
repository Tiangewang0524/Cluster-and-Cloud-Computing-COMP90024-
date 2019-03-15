#This file is developed by Team 18 of COMP90024 of The University of Melbourne, under Apache Licence(see LICENCE).
#Researched Cities: Victoria, AU
#Team member - id:
#Yixiong Ding  671499     
#Yijie Mei     861351
#Tiange Wuang  903588
#Wuang Shen    716090
#Ruifeng Luo   686141

import os.path

f=None
count=0

def flush():
    f.flush()

def update(from_id,to_id,next_id):
    f.seek(0)
    s="{}\n{}\n{}".format(from_id,to_id,next_id)
    f.write(s)
    f.truncate(len(s))
    global count
    count+=1
    if count>=10:
        count=0
        flush()
    
def get():
    f.seek(0)
    s=f.read().split("\n")
    frm=int(s[0])
    to=int(s[1])
    nxt=int(s[2])
    return (frm,to,nxt)



def __init__():
    global f
    if os.path.isfile("progress.txt"):
        f=open("progress.txt","r+")
    else:
        f=open("progress.txt","w+")
        update(-1,-1,-1)

__init__()
