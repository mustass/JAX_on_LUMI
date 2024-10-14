python3 -m venv /scratch/project_465001020/hackathon/venv/ --system-site-packages
source /scratch/project_465001020/hackathon/venv/bin/activate
python3 -m pip install -r requirements.txt --ignore-installed
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu