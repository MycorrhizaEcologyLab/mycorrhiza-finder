echo "Building AMFinder executable..."

@REM You need to build react app and then copy static files over as below
cd ../../../amfbrowser-react
npm install
npm run build

cd ..\amf\amf_code\build
rm -Recurse ../_internal/static/js
rm -Recurse ../_internal/static/media
rm -Recurse ../_internal/static/css
rm ../_internal/static/amfbrowser.ico
rm ../_internal/templates/index.html

cp -r ../../../amfbrowser-react/build/static/* ../_internal/static
cp -r ../../../amfbrowser-react/build/amfbrowser.ico ../_internal/static
cp -r ../../../amfbrowser-react/build/index.html ../_internal/templates

pyinstaller ../main.py --name=amf --noconfirm -i amfbrowser.ico --add-data "../trained_networks/*;./trained_networks" --add-data "../_internal/static/css/*;_internal/static/css" --add-data "../_internal/static/js/*;_internal/static/js" --add-data "../_internal/static/amfbrowser.ico;_internal/static" --add-data "../_internal/static/media/*;_internal/static/media" --add-data "../_internal/templates/index.html;_internal/templates" --add-data "../scripts/*;./scripts" --add-data "../database.ini;." --add-data "../config/*;./config"

@REM For mac, need to brew install python@3.11, brew install postgresql before building on EC2 mac instance, then run above with : instead of ; in the pyinstaller command. Sim for Linux.
