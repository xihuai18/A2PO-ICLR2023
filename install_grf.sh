sudo apt update
sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev \
libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
libdirectfb-dev libst-dev mesa-utils xvfb x11vnc python3-pip -y

python3 -m pip install --upgrade pip setuptools psutil wheel

cd onpolicy/envs/grf

pip uninstall gfootball -y
pip install -U gfootball
echo "install wheel grf env"