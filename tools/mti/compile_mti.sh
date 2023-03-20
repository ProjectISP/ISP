gfortran -O3 -c elemse.for
gfortran -O3 -c gr_xyz.for
gfortran -O3 elemse.o -o elemse
gfortran -O3 gr_xyz.o -o gr_xyz
gfortran -O3 -c curfor.for 
gfortran -O3 curfor.o -o curfor
mkdir -p bin
mv elemse gr_xyz curfor bin/