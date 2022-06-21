C+
C	ratplt.inc
C
C	INCLUDE file for Ratplt
C-
	PARAMETER (MAX = 500)
	PARAMETER (MAX2 = 2*MAX)
C
C	Input Data
C
	REAL*4 XYZ(9,MAX2)	!  R, Theta, and Phi station coordinates
	REAL*4 XYZden(6,MAX2)	!  station coordinates for ratio denominator
	REAL*4 RSTATN(MAX)	!  Stations with amplitude-ratio data
        real*4 rtoa1(MAX)       !  toang1
        real*4 rtoa2(MAX)       !  toang2
        real*4 az(MAX)          !  azin
        character*3 flag(MAX)
C				     X = north, Y east, Z down
	REAL*4 LOGRAT(MAX)	!  Log10 of (SV/P), (SH/P), or (SV/SH)
C				   including free surface correction, etc.
	real*4 diff(max)        !  observ-calc log10 ratio
        INTEGER NRAT		!  Number of ratios read in
	INTEGER KEYRAT(MAX)	!  Keys ratios data relative to input
	CHARACTER*1 SVSH(2,MAX)	! First element V for SV or H for SH,
C				  Second element polarity: F or B for SV
C				  L or R for SH (back to station) for
C	                          numerator in SV/P, SH/P, or SV/SH ratio
C
C	  Processing things
C
	REAL*4 ERRRAT		!  Maximum allowed Log10 ratio error
	REAL*4 CALRAT(MAX)	!  Calculated Log10 of amplitude ratio
	INTEGER WTRAT(MAX)	!  1 if that ratio used, 0 otherwise
	REAL*4 VPVS3		!  Cube of P/S velocity ratio
	REAL*4 CUTP		!  cutoff limit for  P radiation factor
C				   in amplitude ratios
	REAL*4 CUTS		!  cutoff limit for  S radiation factor
C				   in amplitude ratios
C
	COMMON /FOCMEC/ XYZ,XYZden,RD,LOGRAT,CUTP,CUTS,rstatn,diff,svsh,
     1	  NRAT,ERRRAT,CALRAT,WTRAT,VPVS3,KEYRAT,rtoa1,rtoa2,az,flag 
