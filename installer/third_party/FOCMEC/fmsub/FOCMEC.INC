C+
C	FOCMEC.INC
C
C	INCLUDE file for FOCMEC
C	27 August 1991 for the sun now includes SV
C	  polarities and SV/SH ratios
C	2 August 1993: changed ratio evaluations near nodal planes,
C	     so left out RLOW* and RHI* and replaced CUT with CUTP
C	     and CUTS.
c       December 2013: When I added SV/SH I did not take into account the 
C               fact that the denominator take-off angle is not a P.
C               Added XYNden array defined in focinp.f and used in lratio.f
C	NOTE! For some compilers, the ! starting an on-line comment may cause
C		an error.  Simply delete the ends of line starting at the !
C		if that is the case.
C-
	PARAMETER (MAX = 500)
	PARAMETER (MAX2 = 2*MAX)
C
C	Input Data
C
	REAL*4 XYZ(9,MAX2)	!  R, Theta, and Phi station coordinates
	REAL*4 XYZden(6,MAX2)	!  station coordinates for ratio denominator 
	REAL*4 PSTATN(MAX)	!  Stations with polarity data
	REAL*4 RSTATN(MAX)	!  Stations with amplitude-ratio data
C				     X = north, Y east, Z down
	INTEGER POLRTY(MAX)	!  1 for C, F, or <;  -1 for D, B, or >
	INTEGER NPOL		!  Number of Polarities read in
	INTEGER KEYPOL(MAX)	!  Keys polarity data relative to input
	REAL*4 LOGRAT(MAX)	!  Log10 of (SV/P), (SH/P), or (SV/SH)
C				   including free surface correction, etc.
	INTEGER NRAT		!  Number of ratios read in
	INTEGER KEYRAT(MAX)	!  Keys ratios data relative to input
	CHARACTER*1 SVSH(2,MAX)	! First element V for SV or H for SH,
C				  Second element polarity: F or B for SV
C				  L or R for SH (back to station) for
C	                          numerator in SV/P, SH/P, or SV/SH ratio
C
C	  Processing things
C
	INTEGER MAXSOL	!  Exit after this many accept. aols.
	REAL*4 ERR		!  Total allowed number of errors
	REAL*4 ERRP		!  Allowed number of P polarity errors
	REAL*4 ERRSV		!  Allowed number of SV polarity errors
	REAL*4 ERRSH		!  Allowed number of SH polarity errors
	REAL*4 THRESH		!  Polarity weighting threshold
C				  (1.0 if no weighting)
	REAL*4 BADPP(MAX)	!  Stations with a P polarity error
	REAL*4 BADSVP(MAX)	!  Stations with a SV polarity error
	REAL*4 BADSHP(MAX)	!  Stations with a SH polarity error
	REAL*4 WBADP(MAX)	!  Weishts of P polarity errors
	REAL*4 WBADSV(MAX)	!  Weights of SV polarity errors
	REAL*4 WBADSH(MAX)	!  Weights of SH polarity errors
	REAL*4 BADP		!  Total weight of P polarity errors
	REAL*4 BADSV		!  Total weight of SV polarity errors
	REAL*4 BADSH		!  Total weight of SH polarity errors
	INTEGER NERRR		!  Allowed number of ratio errors
	REAL*4 ERRRAT		!  Maximum allowed Log10 ratio error
	REAL*4 CALRAT(MAX)	!  Theoretical Log10 of amplitude ratio
	INTEGER WTRAT(MAX)	!  1 if that ratio used, 0 otherwise
	REAL*4 VPVS3		!  Cube of P/S velocity ratio 
	REAL*4 BTMIN,BTDEL,BTMAX	!  Search range in B axis trend
	REAL*4 BPMIN,BPDEL,BPMAX	!  Search range in B axis plunge
	REAL*4 AAMIN,AADEL,AAMAX	!  Search range in A axis angle
C					   measured from B trend
	REAL*4 CUTP		!  cutoff limit for  P radiation factor
C				   in amplitude ratios
	REAL*4 CUTS		!  cutoff limit for  S radiation factor
C				   in amplitude ratios
C
	COMMON /FOCMEC/ PSTATN,RSTATN,XYZ,XYZden,RD,POLRTY,NPOL,LOGRAT,
     1	  NRAT,MAXSOL,ERRP,ERRSV,NERRR,ERRRAT,CALRAT,WTRAT,VPVS3,
     2	  BTMIN,BTDEL,BTMAX,BPMIN,BPDEL,BPMAX,AAMIN,AADEL,AAMAX,
     3	  BADPP,KEYRAT,KEYPOL,ERRSH,BADSHP,ERR,THRESH,WBADP,WBADSH,
     4    WBADSV,BADSVP,CUTP,CUTS,BADP,BADSV,BADSH
	COMMON /FMCHAR/ SVSH
