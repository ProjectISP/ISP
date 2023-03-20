gfortran -O3 -c elemse.for
gfortran -O3 -c gr_xyz.for
gfortran -O3 elemse.o -o elemse
gfortran -O3 gr_xyz.o -o gr_xyz
