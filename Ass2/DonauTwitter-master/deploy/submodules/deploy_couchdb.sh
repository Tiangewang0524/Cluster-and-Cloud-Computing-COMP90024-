#This file is developed by Team 18 of COMP90024 of The University of Melbourne, under Apache Licence(see LICENCE).
#Researched Cities: Victoria, AU
#Team member - id:
#Yixiong Ding  671499     
#Yijie Mei     861351
#Tiange Wuang  903588
#Wuang Shen    716090
#Ruifeng Luo   686141

node=$1
docker rm -f $(docker ps -a -q)

export size=1
export user=admin
export pass=admin

mkdir /extra/data
mkdir /extra/etc
mkdir /extra/etc/local.d
rm /extra/etc/local.d/*
touch /extra/etc/local.d/1.ini
docker create --net=host --volume /extra/data:/opt/couchdb/data --volume /extra/etc/local.d:/opt/couchdb/etc/local.d  couchdb:2.1.1
sleep 1

cont=$(docker ps --all | grep couchdb | cut -f1 -d' ')

docker start ${cont}
sleep 1


docker exec ${cont} \
      bash -c "echo \"-setcookie couchdb_cluster\" >> /opt/couchdb/etc/vm.args"
docker exec ${cont} \
      bash -c "echo \"-name couchdb@${node}\" >> /opt/couchdb/etc/vm.args"

docker restart ${cont}
sleep 10
 
curl -XPUT "http://localhost:5984/_node/_local/_config/admins/${user}" --data "\"${pass}\""    
curl -XPUT "http://${user}:${pass}@localhost:5984/_node/couchdb@${node}/_config/chttpd/bind_address" --data '"0.0.0.0"'

rev=`curl -XGET "http://localhost:5986/_nodes/nonode@nohost" --user "${user}:${pass}" | sed -e 's/[{}"]//g' | cut -f3 -d:`
curl -X DELETE "http://localhost:5986/_nodes/nonode@nohost?rev=${rev}"  --user "${user}:${pass}"
