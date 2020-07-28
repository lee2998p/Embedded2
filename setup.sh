sudo apt-get update -y
sudo apt-get install -y python3-pip  python3-flask python3-scipy python3-matplotlib python3-paramiko

#Fan drivers
git clone https://github.com/Pyrestone/jetson-fan-ctl.git
cd jetson-fan-ctl
./install.sh
cd ../

#PyTorch 1.4.0
whl="torch-1.4.0-cp36-cp36m-linux_aarch64.whl"
wget https://nvidia.box.com/shared/static/c3d7vm4gcs9m728j6o5vjay2jdedqb55.whl -O ${whl}
sudo apt-get install libopenblas-base libopenpi-dev
pip3 install future ${whl} 
rm ${whl}

#Torchvision v0.5.0
sudo apt-get install libjpeg-dev zlib1g-dev
git clone --branch "v0.5.0" https://github.com/pytorch/vision torchvision
cd torchvision
sudo python3 setup.py install
cd ../
pip3 install 'pillow<7'

#pip dependencies
pip3 install -r jetson-requirements.txt

#mysql
sudo apt-get install mysql-server
pip3 install mysql-connector
