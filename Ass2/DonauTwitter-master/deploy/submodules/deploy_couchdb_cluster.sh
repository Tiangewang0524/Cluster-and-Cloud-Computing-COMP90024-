#This file is developed by Team 18 of COMP90024 of The University of Melbourne, under Apache Licence(see LICENCE).
#Researched Cities: Victoria, AU
#Team member - id:
#Yixiong Ding  671499     
#Yijie Mei     861351
#Tiange Wuang  903588
#Wuang Shen    716090
#Ruifeng Luo   686141

nodes=( "$@" )
export masternode=`echo ${nodes} | cut -f1 -d' '`
export othernodes=`echo ${nodes[@]} | sed s/${masternode}//`
export size=${#nodes[@]}
export user=admin
export pass=admin

for node in "${nodes[@]}"; do     
    curl -XPOST "http://${user}:${pass}@${masternode}:5984/_cluster_setup" \
      --header "Content-Type: application/json" \
      --data "{\"action\": \"enable_cluster\", \"bind_address\":\"0.0.0.0\", \
        \"username\": \"${user}\", \"password\":\"${pass}\", \"port\": \"5984\", \
        \"remote_node\": \"${node}\", \
        \"remote_current_user\":\"${user}\", \"remote_current_password\":\"${pass}\"}"
done
for node in "${nodes[@]}"; do     
    curl -XPOST "http://${user}:${pass}@${masternode}:5984/_cluster_setup" \
      --header "Content-Type: application/json" \
      --data "{\"action\": \"add_node\", \"host\":\"${node}\", \
        \"port\": \"5984\", \"username\": \"${user}\", \"password\":\"${pass}\"}"
done

curl -XPOST "http://${user}:${pass}@${masternode}:5984/_cluster_setup" \
    --header "Content-Type: application/json" --data "{\"action\": \"finish_cluster\"}" 

curl -X PUT "http://${user}:${pass}@${masternode}:5984/tweets"