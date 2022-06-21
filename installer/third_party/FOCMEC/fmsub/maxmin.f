C+
	SUBROUTINE MAXMIN(DATA,NPTS,RMIN,RMAX,ITYPE,NU1,NU2)
C 
C     Returns (real) minimum and maximum values for an array DATA
C	Number of points to be included is NPTS
C	DATA can be INTEGER*2, INTEGER*4, REAL or DOUBLE PRECISION
C		If ITYPE = 'SP' or 'R4', single precision
C		           'DP' or 'R8', double precision
C		           'I2', integer*2
C		           'I4', integer*4
C	If NU1 > 0, prints RMAX and RMIN on unit NU1.  Same for NU2.
C		Arthur Snoke:  February 1983
C	6 October 1988:  If NU1 or NU2 are 5, writes to *  (VAX)
C	7 July 1991:  sun version:  calls minmax
C-
	INTEGER*2 DATA(*)
	character*2 itype
	NSTART = 1
	call minmax(data,nstart,npts,rmin,rmax,itype,nu1,nu2)
	RETURN
	END 
