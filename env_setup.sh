# Goto main
cd /home/seelmath/Desktop/spaceship_titanic/code

# Remove previous environment
conda deactivate
conda remove --name spaceship --all -y

# Create new environment
conda create --name spaceship python=3.9 -y
conda activate spaceship

# Upgrade pip
python -m pip install --upgrade pip==22.2.2

# Install packages
pip install -r requirements.txt
