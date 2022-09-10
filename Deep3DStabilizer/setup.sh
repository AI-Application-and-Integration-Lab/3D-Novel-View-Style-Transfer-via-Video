apt-get update -y
apt-get upgrade -y
apt-get install -y vim python-pip python3-pip libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg
apt install libgl1-mesa-glx

# python2 for PWC-Net
pip install --upgrade "pip < 21.0"
apt-get -qq -y install curl bzip2 && \
curl -sSL https://repo.continuum.io/miniconda/Miniconda2-4.6.14-Linux-x86_64.sh -o /tmp/miniconda.sh && \
bash /tmp/miniconda.sh -bfp /usr/local && \
rm -rf /tmp/miniconda.sh && \
conda install -y python=2 && \
conda update conda && \
apt-get -qq -y autoremove && \
apt-get autoclean && \
rm -rf /var/lib/apt/lists/* /var/log/dpkg.log && \
conda clean --all --yes && \
conda install --all --yes pytorch=0.2.0 cuda80 -c soumith 
python2 -m pip install torchvision==0.2.2 opencv-python==4.1.0.25 cffi==1.12.2 tqdm==4.19.9 scipy
cd ./PWC/correlation-pytorch-master
./make_cuda.sh
cd ../..

# python3 for main program
apt-get update -y
apt-get upgrade -y
pip3 install --upgrade "pip < 21.0"
pip3 install torch==1.4.0 torchvision==0.5.0 opencv-python scipy tqdm path imageio scikit-image imageio scikit-image ffmpeg pypng
