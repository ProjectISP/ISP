C+
	SUBROUTINE SRCHFM
C
C	Subroutine called by FOCMEC
C
C	Using parameters established in FOCINP it searches the
C	focal sphere for acceptable solutions based on polarity
C	and/or amplitude ratios.  The logic of the search
C	is as follows:  One chooses an orientation for the null
C	or B axis (the trend and plunge).  To cover the focal
C	sphere, the trend would vary from 0 to 360 degrees and
C	the plunge from 0 to 90.  All possible focal mechanisms
C	will then be included if the A axis (Herrmann's X axis)
C	varies from 0 to 180 degrees.  However, solutions in the
C	second quadrant are the same as ones in the first except
C	for the sign of the slip direction.  So only the range
C	from 0 to 90 degrees need be calculated with two 
C	possibilities for slip direction considered for each
C	orientation.  (For the ratios the two solutions would
C	be identical.)  The procedure to calculate the actual
C	solution is described in FLTSOL and the comparison for
C	polarities in OKPOL.
C
C	Arthur Snoke  Virginia Tech  July 1984
C
C	2 September 1986:  made sampling truly equal area on focal sphere
C	8 January 2000:  Added do ... 200 instead of a second do ... 300
C	June 2009:  ANGLE may be with respect to the A axis or the N axis.
C		It depends on JSLIP
C       June 2014:  Took out JSLIP loop because search for A 0-180 now
C               ANGLE now unambiguously defined.  Passed TREND and PLUNGE
C               to oksol.f
C-
	INCLUDE 'FOCMEC.INC'
	REAL*4 A(3), N(3), BMATRX(3,3)
	LOGICAL OKSOL
	NPLUNG = 1 + NINT((BPMAX-BPMIN)/BPDEL)
	if (nplung .eq. 1) bpmax = bpmin
	NAANG = 1 + NINT((AAMAX-AAMIN)/AADEL)
	if (naang .eq. 1) aamax = aamin
	NSOL = 0
	DO 500 JP = 1,NPLUNG
	  IF (JP .LT. NPLUNG) THEN
	    PLUNGE = (BPMIN + (JP-1)*BPDEL)/RD
	  ELSE
	    PLUNGE = BPMAX/RD
	  ENDIF
	  IF (JP .EQ. NPLUNG .AND. BPMAX .GE. 90.0) THEN
	    BTDELN = 0.0
	    NTREND = 1
	  ELSE
	    BTDELN = BTDEL/COS(PLUNGE)
	    NTREND = NINT((BTMAX+BTDEL-BTMIN)/BTDELN)
	    BTDELN = (BTMAX+BTDEL-BTMIN)/NTREND
	  ENDIF
	  DO 400 JT = 1,NTREND
	    TREND = AMIN1(BTMIN + (JT-1)*BTDELN,BTMAX)/RD
	    IF (PLUNGE .EQ. 0.0 .AND. TREND .GE. 180.0/RD) GO TO 400
	    DO 300 JA = 1,NAANG
	      IF (JA .LT. NAANG) THEN
	        ANGLE = (AAMIN + (JA-1)*AADEL)/RD
	      ELSE
	        ANGLE = AAMAX/RD
	      ENDIF
	      CALL FLTSOL(A,N,BMATRX,PLUNGE,TREND,ANGLE,JA)
	        IF (.NOT.OKSOL(A,N,TREND*RD,PLUNGE*RD,ANGLE*RD)) 
     1              GO TO 300
	        NSOL = NSOL + 1
	        IF (NSOL .EQ. MAXSOL) THEN
	          WRITE(2,3) MAXSOL
	          WRITE(*,3) MAXSOL
3	  FORMAT(' Reached chosen maximum of',I5,' solutions')
	          GO TO 600
	        END IF
300	    continue
400	  continue
500	CONTINUE
600	IF (NSOL .GT. 0) THEN
	  WRITE(*,1) NSOL
	  WRITE(2,1) NSOL
1	FORMAT(/' There are',I4,' acceptable solutions')
	ELSE
	  WRITE(2,2)
	  WRITE(*,2)
2	FORMAT(/' There are no acceptable solutions')
	ENDIF
	RETURN
	END
