C+
	REAL FUNCTION RVALUE(MSG,RDEF)
C
C PURPOSE:
C		THIS FUNCTION ACCEPTS A MESSAGE (ASKING FOR A VALUE)
C		AND RETURNS THE VALUE ENTERED AT THE TERMINAL
C ROUTINES CALLED:
C		PRINTX
C
C USE:
C		ANS=RVALUE('ENTER AN REAL NUMBER',RDEF)
C	If enter a cariage return, rvalue is set to RDEF
C
C AUTHOR:
C			ALAN LINDE ... AUGUST 1980 (for VALUE)
C
C EXTENSIONS:
C	30 JULY 1989:  CAN HANDLE ENTRY FOLLOWED BY A BLANK OR TAB
C       27 July 1993: Did input read through cstring so can have 
C         comment lines
C	19 July 2002: PCs had a problem with single-digit integers, so ...
C-
	CHARACTER*1 E/'E'/,BLANK/' '/
	CHARACTER*30 STUFF
	CHARACTER*(*) MSG
C
100	CALL PRINTX(MSG)
	call cstring(stuff,nin)
	IF (NIN .GT. 0) THEN
	  NBLANK = INDEX(STUFF(1:NIN),BLANK)
	  IF (NBLANK .GT. 0) NIN = NBLANK - 1
	END IF
	IF (NIN .EQ. 0) THEN
	  RVALUE = RDEF
	ELSE
	  if (nin .eq. 1) then
	    stuff(2:2) = stuff(1:1)
	    stuff(1:1) = '0'
	    nin = 2
	  end if
	  READ(STUFF(1:NIN),*,ERR=100) RVALUE
	END IF
	RETURN
	END
