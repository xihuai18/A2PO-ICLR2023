sudo apt update
sudo apt-get install libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev libglew-dev patchelf gcc -y
mkdir ~/.mujoco

# download mujoco

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/your/path/to/mujoco/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/your/path/to/mujoco/bin" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia" >> ~/.bashrc

pip install gym==0.21.0 mujoco_py

sudo dpkg -i libs/libffi7_3.3-4_amd64.deb

sh ./tests/test_mujoco.sh