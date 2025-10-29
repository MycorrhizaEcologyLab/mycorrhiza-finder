echo "Preparing environment for building AMFinder executable..."

python -m venv venv
venv\Scripts\activate
python -m pip install -r ../requirements.txt
python -m pip install pyinstaller
