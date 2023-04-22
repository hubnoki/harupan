
Param([String]$Arg1 = "dist")
pyinstaller harupan.py --distpath pyinstaller/$Arg1 --workpath pyinstaller/build --specpath pyinstaller --add-data "..\data\harupan_svm_220412.dat;data" --add-data "..\data\templates2021.json;data" --noconsole --onefile --icon ..\icon\harupan_icon4.png
