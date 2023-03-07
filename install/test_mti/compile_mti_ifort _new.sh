#https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-fortran-compiler/top/get-started-on-macos.html
#https://www.intel.com/content/www/us/en/developer/articles/tool/oneapi-standalone-components.html#fortran
source /opt/intel/oneapi/setvars.sh
ifort -c elemse1024Intel.for
ifort -c gr_xyz_Intel.for
ifort elemse_1024Intel.o -o elemse
ifort gr_xyz_Intel.o -o gr_xyz