# ISP (Integrated Seismic Program)

Software for making analysis of seismic waves. 

### For Developers:

Please install [pyqt5](https://pypi.org/project/PyQt5/) in your python env. Make sure you 
also have installed qt designer. 

* Rebuilding resources.
 
After any change in the resource file *resources/resources.qrc* run the following command to 
regenerate the file *resources_rc.py*:
    
    pyrcc5 -o resources_rc.py resources.qrc

* Compiling C code. 

Make sure you have the right python venv activated:

    source path/venv/bin/activate 

Then just go to the project folder and run: 

    python setup.py build_ext --inplace
