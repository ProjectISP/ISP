	SUBROUTINE PRINTX(LINE)
C+
c	SUBROUTINE PRINTX(LINE)
C  OUTPUTS A MESSAGE TO THE TERMINAL
C  PRINTX STARTS WITH A LINE FEED BUT DOES NOT END WITH A CARRIAGE RETURN
C  THE PRINT HEAD REMAINS AT THE END OF THE MESSAGE
C
C  IF THE MESSAGE LENGTH IS LESS THAN 40,
C	DOTS ARE INSERTED UP TO COL. 39
C	AND A COLON IS PUT IN COL. 40.
C
C  USE FOR CONVERSATIONAL INTERACTION
C			Alan Linde ... April 1980.
C	10 Sugust 1985:  Corrected a minor error for  strings > 40 bytes
C	20 June 1986:  Made it compatible with Fortran 77
C	24 September 2001: On some platforms there are problems when one
C		writes into column 1.  So the write sstatement now has a 
C		1x to start out.
C-
	character*(*) line
	CHARACTER*60 BUF
	CHARACTER*2 COLON
	CHARACTER*1 DOT,DELIM
	DATA DELIM/'$'/,DOT/'.'/,COLON/': '/
	KK = lenc(LINE)	!  length minus right-hand blanks
	  IF (LINE(KK:KK) .EQ. DELIM) KK = KK - 1
	  IF (KK .GT. 58) KK = 59
	BUF(1:KK) = LINE(1:KK)
	IF (KK .LT. 49) THEN
	  DO J=KK+1,49
	    BUF(J:J) = DOT
	  END DO
	  KK = 49
	END IF
	BUF(KK:KK+1) = COLON
	KK = KK + 1
	WRITE(*,'(1x,A,$)') BUF(1:KK)
	RETURN
	END
