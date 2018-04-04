# Install software in GPU instance
# GPU instances docs https://cloud.google.com/compute/docs/gpus/add-gpus#create-new-gpu-instance
cd $HOME

sudo apt-get update
sudo apt-get -y upgrade
sudo apt-get -y install bzip2 git htop

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

mkdir dev && cd dev
git clone https://github.com/edublancas/dotfiles
python3 dotfiles/setup/make_symlinks

git clone https://github.com/edublancas/trojan-defender
cd trojan-defender
pip install -r requirements.txt

cd pkg
pip install --upgrade pip
pip install -e .

pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0-cp36-cp36m-linux_x86_64.whl


# gcloud compute scp ./libcudnn6_6.0.21-1+cuda8.0_amd64.deb \
#     gpu-instance-1:~ --zone us-east1-c

sudo dpkg -i libcudnn6_6.0.21-1+cuda8.0_amd64.deb
