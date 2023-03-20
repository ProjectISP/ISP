C+
	SUBROUTINE MINMAX(DATA,NSTART,NPTS,RMIN,RMAX,ITYPE,NU1,NU2)
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
C	7 July 1991:  unix/sun version
C-
	CHARACTER*2 ICHECK,ITYPE,SP,R4,DP,R8,I2,I4 
	INTEGER*2 DATA(*),IDAT2(4)
	INTEGER*4 IDAT4,MIN,MAX
	real*8 rdat8
	EQUIVALENCE (RDAT4,IDAT4,IDAT2(1),rdat8)
	DATA SP,R4,DP,R8,I2,I4/'SP','R4','DP','R8','I2','I4'/
	RMAX = -1.0E30 
	RMIN = 1.0E30
	ICHECK = ITYPE
	IF (ICHECK .EQ. SP) ICHECK = R4
	IF (ICHECK .EQ. DP) ICHECK = R8
	IF (ICHECK .EQ. I2) NST = NSTART
        IF (ICHECK .EQ. I4 .OR. ICHECK .EQ. R4) NST = 2*NSTART - 1
        IF (ICHECK .EQ. R8) NST = 4*NSTART - 3
        IF (ICHECK .EQ. I2) NTOT = NST - 1 + NPTS
        IF (ICHECK .EQ. R4 .OR. ICHECK .EQ. I4) NTOT = NST - 1 + 2*NPTS
        IF (ICHECK .EQ. R8) NTOT = NST - 1 + 4*NPTS
        IF (ICHECK .EQ. I2) INTRVL = 1
        IF (ICHECK .EQ. R4 .OR. ICHECK .EQ. I4) INTRVL = 2
        IF (ICHECK .EQ. R8) INTRVL = 4
        DO 100 J=NST,NTOT,INTRVL
	IDAT2(1) = DATA(J)
	IDAT2(2) = DATA(J+1)
	IF (ICHECK .EQ. I2) RDAT = IDAT2(1)
	IF (ICHECK .EQ. I4) RDAT = IDAT4
	IF (ICHECK .EQ. R4) RDAT = RDAT4
	if (ICHECK .EQ. R8) then
	  IDAT2(3) = DATA(J+2)
	  IDAT2(4) = DATA(J+3)
	  RDAT = RDAT8
	END IF
	IF (RDAT .LT. RMIN) RMIN = RDAT
100	IF (RDAT .GT. RMAX) RMAX = RDAT
	IF (NU1 .LE. 0) RETURN
	IF (ICHECK .EQ. I4 .OR. ICHECK .EQ. I2) GO TO 200
	HALF = 0.5*ABS(RMIN-RMAX)
	IF (NU1 .NE. 5) THEN
	  WRITE (NU1,1) RMIN,RMAX,HALF
	ELSE
	  WRITE (*,1) RMIN,RMAX,HALF
	END IF
1	FORMAT(' Minimum is ',1PG10.3,'   Maximum is ',G10.3,
     .    '  Half Range is ',G10.3)
	IF (NU2 .GT. 0) THEN
	  IF (NU2 .NE. 5) THEN
	    WRITE (NU2,1) RMIN,RMAX,HALF
	  ELSE
	    WRITE (*,1) RMIN,RMAX,HALF
	  END IF
	END IF	
	RETURN
200	MIN = RMIN
	MAX = RMAX
	IF (NU1 .NE. 5) THEN
	  WRITE (NU1,2) MIN,MAX
	ELSE
	  WRITE (*,2) MIN,MAX
	END IF
2	FORMAT(' Minumum is ',I10,'   Maximum is ',I10)
	IF (NU2 .GT. 0) THEN
	  IF (NU2 .NE. 5) THEN
	    WRITE (NU2,2) MIN,MAX
	  ELSE
	    WRITE (*,2) MIN,MAX
	  END IF
	END IF
	RETURN
	END 
