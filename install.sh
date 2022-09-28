# conda activate co-marl
# conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install psycopg2-binary setproctitle absl-py pysc2 gym tensorboardX prettytable pyyaml
# sudo apt install build-essential -y
pip install Cython
pip install sacred aim
# install on-policy package
pip install -e .