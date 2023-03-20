C+
C	PROGRAM FOCPLT
C
C	Plots polarity and/or SV/P or SH/P amplitude ratio data on the
C	  projection of the upper or lower hemisphere of the focal sphere.
C	Also can plot focam mechanism solutions entered individually
C	  or from a file, such as that produced by FOCMEC.
C	  The options here are to plot either fault planes
C	  or P, T, and B axis projections.
C	The symbols can have different colors and the focal plane
C	  lines can be dashed.
C
C	Library routines used are NSTRNG, PRINTX, VALUE, IYESNO, 
C	  TRUTH, PRINT, TIMDAT, TSEND, CLRGRF, LINEWIDTH, VCURSR,SYMBOL, MINMAX
C	Digital F77 routines include: SECNDS, OPEN, ASSIGN, CLOSE,
C	  TIME, DATE, ENCODE
C
C	Written by Arthur Snoke and Jeff Munsey
C	31 JULY 1985 CHANGED PRPLOT (ADDED SH POLARITIES)
C	  INCLUDED MOMENT TENSOR INFO IN FMREPS
C	7 September 1985:  Added upper hemisphere option
C	4 October 1985:  Added plot options of SV and SH nodal surfaces
C	  (Code lifted from Bruce Julian via Paul Silver of DTM)
C	6 February 1986:  Made changes to SOLPLT and PTBPLT to
C	  allow plotting individual P,T,B axes and to use different
C	  LINEWIDTH values
C	12 September 1986:  Corrected errors in plots of nodal planes
C	  and surfaces for dip=rake=0
C	16 October 1986:  Tidied:  Made conventions more compatible with F77
C	30 August 1991  sun version.  Includes SV polarities, SV/SH ratios
C       7 September 2017:  Took out ratio option because better done in
C        program RATPLT.
C-	
	character*24 TD
	LOGICAL TRUTH,DATPLT,UPPER
	DATA RADIUS,R2/2.3,5.6/
	open(2,file='focplt.lst',status='unknown')
	CALL TIMDAT(2,'Focplt')
	BACKSPACE 2
	READ(2,'(1X,A24)') TD
	UPPER = TRUTH('Upper hemisphere projection?..[N]')
	DATPLT = TRUTH('Plot polarity data?')
	IF(DATPLT) CALL POLPLOT(TD,RADIUS,UPPER)
	IF (TRUTH('Plot focal mechanism solutions?..[Y]'))
     .	  CALL SOLPLT(RADIUS,DATPLT,TD,UPPER)
 	IF(TRUTH('Add a plot label?')) CALL PLTLAB(R2,R2)
	CALL PLOT(0.0,0.0,999)
	STOP
	END
