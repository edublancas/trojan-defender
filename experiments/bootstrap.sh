# GPU instances docs https://cloud.google.com/compute/docs/gpus/add-gpus#create-new-gpu-instance
# install software
cd $HOME

sudo apt-get update
sudo apt-get -y upgrade
sudo apt-get -y install bzip2 git

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

mkdir dev
cd dev
git clone https://github.com/edublancas/dotfiles
./dotfiles/setup/make_symlinks

git clone https://github.com/edublancas/trojan-defender
cd trojan-defender/pkg
pip install --upgrade pip
pip install -e .