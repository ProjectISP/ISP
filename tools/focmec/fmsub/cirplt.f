C+
	SUBROUTINE CIRPLT(RADIUS,TITLE,ASK)
C
C	 This subroutine draws a circle, marks the center (+) and
C	 plots an 'N' for north.  The radius is the inputted RADIUS.
C	It then prompts for a title (to be printed below).
C	It calls PLOTS.  If a disk file is desired, the unit is 3.
C	If ASK is .TRUE., one is prompted for a title,
C	  Otherwise one is passed in.
C	  (Note that the title cannot be
C	  more than 40 characters.)
C	If the first character in a passed in title is a blank,
C	  no title is printed.
C	5 July 2002: Some compilers do not like the way I had used
C	  CVALUE -- CVALUE(1:NFILE) with NFILE defined on the right-hand side.
C	June 2009 spcsmb call instead of symbol to get rid of warning
C-
	LOGICAL TRUTH,ASK
	CHARACTER*40 TITLE,CVALUE,DUMMY
	RD = 45.0/ATAN(1.0)
	R2 = 2.0*RADIUS
	CALL PLOTS(3,15,3)	!  Open plot file (disk and/or terminal)
	IPEN = IVALUE('linewidth for circle outline..[2]',2)
	CALL linewidth(IPEN)
	CALL PLOT(2.5+RADIUS,0.8+RADIUS,-3)
	CALL PLOT(RADIUS,0.0,3)
	DO 100 J=1,361
	  TH = FLOAT(J)/RD
	CALL PLOT(RADIUS*COS(TH),RADIUS*SIN(TH),2)
100	CONTINUE
	CALL SPCSMB(0.0,0.0,0.1,3,0.0,-1)
	CALL PLOT(0.0,RADIUS-0.2,3)
	CALL PLOT(0.0,RADIUS,2)
	CALL SYMBOL(-0.07,RADIUS+0.1,0.25,'N',0.0,1)
	CALL PLOT(0.0,-RADIUS+0.2,3)
	CALL PLOT(0.0,-RADIUS,2)
	CALL PLOT(RADIUS-0.2,0.0,3)
	CALL PLOT(RADIUS,0.0,2)
	CALL PLOT(-RADIUS+0.2,0.0,3)
	CALL PLOT(-RADIUS,0.0,2)
	CALL TSEND
	IF (ASK) THEN
	  IF (.NOT.TRUTH('Add a title?..[Y]')) RETURN
	  TITLE = 
     1       CVALUE('Enter title - up to 40 characters',DUMMY,NTITLE)
	ELSE
	  IF (TITLE(1:1) .EQ. ' ') RETURN
	  NTITLE = LENC(TITLE)
	END IF
	IPEN = IVALUE('linewidth for title..[2]',2)
	CALL linewidth(IPEN)
	CALL MYLABL(-RADIUS,-RADIUS-0.6,2*RADIUS,0.2,0.,TITLE,ntitle)
	CALL TSEND
	RETURN
	END
