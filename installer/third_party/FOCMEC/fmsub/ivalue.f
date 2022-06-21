C+
	INTEGER FUNCTION IVALUE(MSG,IDEF)
C
C PURPOSE:
C		THIS FUNCTION ACCEPTS A MESSAGE (ASKING FOR A VALUE)
C		AND RETURNS THE VALUE ENTERED AT THE TERMINAL
C ROUTINES CALLED:
C		PRINTX
C
C USE:
C		IANS=IVALUE('ENTER AN INTEGER',IDEF)
C	If enter a carriage return, IVALUE is set to IDEF.
C
C AUTHOR:
C			ALAN LINDE ... AUGUST 1980 (for VALUE)
C
C EXTENSIONS:
C	30 JULY 1989:  CAN HANDLE ENTRY FOLLOWED BY A BLANK OR TAB
C       27 July 1993: Did input read through cstring so can have 
C         comment lines
C	19 July 2002: PCs had a problem with single-digit integer, so ...
C	14 July 2008:  Some compilers had problems with an input like 20.
C	  so I now read it in as a floating point and use nint
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
	  IVALUE = IDEF
	ELSE
	  if (nin .eq. 1) then
	    stuff(2:2) = stuff(1:1)
	    stuff(1:1) = '0'
	    nin = 2
	  end if
	  READ(STUFF(1:NIN),*,ERR=100) test
	  ivalue = nint(test)
	END IF
	RETURN
	END
