C+
	LOGICAL FUNCTION OKRAT(DSR,NBADR,ESUM,EMAX,DIFF,SPOL,FLAG)
C
C	Called by OKSOL, which is called by FOCMEC
C	Compares observed log10 amplitude ratios with trial mechanism
C	Diff = observed - calculated ratio for the NRAT stations
C	ESUM = sum of the square of the DIFF for "good" ratios.
C	ESUMAL = sum of the square of the DIFF for all ratios.
C	  designated by A and N
C	8 July 1990:  Put in limits on ratios to allow for nodal surfaces.
C	  The limit for the numerator and denominator as used in LSPRAT is
C	  CUT.  Allowing for the other part of the ratio being non-unity is
C	  taken care of through FACTOR.  My choices are CUT=0.05, FACTOR=2.0.
C	  These have been added to the .INC file and are prompted for in FOCINP
C	29 August 1991:  sun version.  Now allows SV/SH ratios
C	3 August 1993:  Changed rules for nodal surfaces.  No more FACTOR,
C	  Now have CUTP and CUTS.  FLAG labels nodal surface cases
c       December 2013: Added separate station coordinates for ratio denominator
C	July 2016: Now the number of ratio errors is relative to the input
C	  total number rather than that number minus stations for which N & D 
C	  are goth near nodal surfaces.
C       September 2017: Now I will output the maximum abs ratio, not rms all
C         ESUMALL is replaced by EMAX in the calling arguments
C-
	INCLUDE 'FOCMEC.INC'
	character*3 FLAG(MAX)
	INTEGER NBADR
	REAL*4 SPOL(MAX),DSR(3),DIFF(MAX)
	OKRAT = .FALSE.
	NBADR = 0
	IF (NRAT .LE. 0) RETURN
	ESUM = 0.0
	ESUMAL = 0.0
	EMAX = 0.0
	DO K=1,NRAT
	  KKK = KEYRAT(K)
	  IF (KKK-1000 .LT. 0) THEN
	    KK = KKK
	    JR = 1
	  ELSE IF (KKK-2000 .LT. 0) THEN
	    KK = KKK - 1000
	    JR = 2
	  ELSE
	    KK = KKK - 2000
	    JR = 3
	  END IF
	  CALL LRATIO(JR,DSR(1),DSR(2),DSR(3),XYZ(1,KK),XYZden(1,KK),
     .	       VPVS3,RAT,SP,CUTP,CUTS,FLAG(K))
	  SPOL(K) = SP
	  CALRAT(K) = RAT
	  IF (FLAG(K) .eq. 'N&D') THEN
c	Both the numberator and the denominator are < cutoff values.
	    diff(k) = 0.0
	    NBADR = NBADR + 1
	    IF (NBADR .GT. NERRR) RETURN
	  ELSE
	    DIFF(K) = LOGRAT(K) - CALRAT(K)
	    ESUMAL = ESUMAL + DIFF(K)**2
	    IF (ABS(DIFF(K)) .GT. ERRRAT) THEN
	      WTRAT(K) = 0
	      NBADR = NBADR + 1
	      IF (NBADR .GT. NERRR) RETURN
	    ELSE
	      WTRAT(K) = 1
	      ESUM = ESUM + DIFF(K)**2
	      if (abs(diff(k)) .gt. emax) emax = abs(diff(k))
          endif
        ENDIF
	END DO
	OKRAT = .TRUE.
	RETURN
	END
