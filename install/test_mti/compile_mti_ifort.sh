#https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-fortran-compiler/top/get-started-on-macos.html
#https://www.intel.com/content/www/us/en/developer/articles/tool/oneapi-standalone-components.html#fortran
source /opt/intel/oneapi/setvars.sh
ifort -c elemse.for
ifort -c gr_xyz.for
ifort elemse.o -o elemse
ifort gr_xyz.o -o gr_xyz