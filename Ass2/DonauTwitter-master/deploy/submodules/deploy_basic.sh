#This file is developed by Team 18 of COMP90024 of The University of Melbourne, under Apache Licence(see LICENCE).
#Researched Cities: Victoria, AU
#Team member - id:
#Yixiong Ding  671499     
#Yijie Mei     861351
#Tiange Wuang  903588
#Wuang Shen    716090
#Ruifeng Luo   686141

sudo apt-get -y install vim
echo -e "cqmygysdss\ncqmygysdss" | sudo passwd ubuntu
sudo mkdir /extra
sudo mkfs.ext4 /dev/vdc

echo -e "sudo mount /dev/vdc /extra -t auto\nsudo chown ubuntu /extra" > mount.sh
chmod 777 mount.sh
./mount.sh

echo "/dev/vdc        /extra  auto    defaults        0       2" | sudo tee --append /etc/fstab > /dev/null