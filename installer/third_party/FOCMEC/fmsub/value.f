C+
	FUNCTION VALUE(MSG)
C
C PURPOSE:
C		THIS FUNCTION ACCEPTS A MESSAGE (ASKING FOR A VALUE)
C		AND RETURNS THE VALUE ENTERED AT THE TERMINAL
C ROUTINES CALLED:
C		PRINTX, CSTRING
C
C USE:
C		ANS=VALUE('ENTER A REAL VALUE')
C	AND
C		IANS=VALUE('ENTER AN INTEGER')
C
C AUTHOR:
C			ALAN LINDE ... AUGUST 1980
C
C EXTENSIONS:
C
C	  One can now enter exponential format in VALUE or RVALUE
C	30 JULY 1989:  CAN HANDLE ENTRY FOLLOWED BY A BLANK OR TAB
C	27 July 1993: Did input read through cstring so can have
C	  comment lines
C
C	2001.05.24	Khalil Hayek, GSC
C	Changed READ(,F20,) to READ(,*,) to be able to handle
C	integers.
C	Not needed on my compilers, but ok.  Commented out E20 option  (jas)
C	19 July 2002: PCs had a problem with single-digit integer, so ...
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
	  VALUE = 0.0
	ELSE
	  if (nin .eq. 1) then
	    stuff(2:2) = stuff(1:1)
	    stuff(1:1) = '0'
	    nin = 2
	  end if
	  READ(STUFF(1:NIN),*,ERR=100) VALUE
	END IF
	RETURN
	END
