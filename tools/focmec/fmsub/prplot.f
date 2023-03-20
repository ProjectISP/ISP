C+
	SUBROUTINE PRPLOT(TD,RADIUS,UPPER)
C
C	Reads in polarity and/or ratio data from a file, plots it
C	  on the lower hemisphere projection of the focal sphere
C	Argurments are the RADIUS of the circle and TD the time
C	  and date for the run in a 19 character string.
C	If UPPER is .TRUE., an upper hemisphere plot is given
C       Virginia Tech Symbol conventions:
C         C = compression (plotted as a hexagon symbol number NS = 1)
C         U = same as C
C         D = dilatation (plotted as a triangle NS=2)
C         + = emergent compression  (NS=3)
C         - = emergent dilatation
C         e = emergent P arrival
C         < = SH first motion to left (back to event) impulsive
C         > = SH first motion to right (back to event) impulsive
C         L = same as <
C         R = same as > (Note: earlier versions used R for SV/P ratio)
C         F = SV first motion away from event impulsive
C         B = SV first motion towards event impulsive
C       SH and SV polarities are plotted as arrows (NS=6)
C         l = SH first motion to left (back to event) emergent
C         r = SH first motion to right (back to event) emergent
C         u = emergent SH arrival
C         V = Log10(SV/P)  (NS=4) (plotted as an X)
C         H = Log10(SH/P)  (NS=4) (plotted as an X)
C         S = Log10(SV/SH) (NS=4) (plotted as an X)
C	If one wants to have errors marked, follow the polarity
C	  or ratio with a duplicate line with a E in the
C	  symbol position.  The error will then be flagged
C	  as a square superimposed on the polarity or ratio.
C	The log10 S/P ratios are plotted as X's (NS=4) and the size
C	  varies continuously.  If the log ratio is less than -0.6,
C	  it will be plotted with a constant, minimum size of 1/7
C	  the inputted scaling parameter.  A log ratio of unity
C	  has the size of the scaling paramater.
C	2 August 1985:  SH polarity options added
C	11 August 1985:  Added multiple files
C	10 September:  Added upper hemisphere option
C	6 October 1985:  INFO is now 27 characters to include the
C	  S takeoff angle
C	1 JULY 1990: S TAKE OFF IS SEPARATED FROM INFO
C	31 August 1991:  sun version.  Includes SV
C	10 July 1993  Moved time/date/fiespec to left
C	5 April 2000 Slight modifications to symbols (R no longer a ratio)
C       June 2009: call to spcsmb rather than symbol gets rid of warning
C       July 2014: Changed scaling for ratios
C-
	CHARACTER*1 SVSHP,CODE
      character*24 td
	CHARACTER*80 FILENA, COM, CVALUE, DUMMY
	CHARACTER*40 TITLE
	LOGICAL TRUTH,PIMP,PEMERG,PGUESS,SHIMP,SHGUES,SHEMRG,RATSYM
	LOGICAL ERRORS, FIRST, UPPER,SVIMP
	CHARACTER*40 INFO
	RD = 45.0/ATAN(1.0)
	SQ2 = SQRT(2.0)
	FIRST = .TRUE.
100	FILENA = CVALUE('Enter input file name',DUMMY,NFILE)
	CLOSE(UNIT=1)
	OPEN(UNIT=1,FILE=FILENA(1:NFILE),STATUS='OLD',ERR=100)
	READ(1,'(A)') COM
	ncom = lenc(com)
	WRITE(*,'(1X,A)') COM(1:NCOM)
	IF (.NOT.TRUTH('Desired file?..[Y]')) GO TO 100
	WRITE(2,'('' Input from file '',A)') FILENA(1:NFILE)
	WRITE(2,4) COM(1:NCOM)
C
	PIMP = .FALSE.
	PEMERG = .FALSE.
	PGUESS = .FALSE.
	SHIMP = .FALSE.
	SHGUES = .FALSE.
	SHEMRG = .FALSE.
	RATSYM = .FALSE.
	ERRORS = .FALSE.
	svimp = .false.
200	READ(1,6,END=300) STA,AZIM,TOA,CODE
	  if (code .eq. 'U') code = 'C'
	  IF (CODE .EQ. 'C' .OR. CODE .EQ. 'D') PIMP = .TRUE.
	  IF (CODE .EQ. '+' .OR. CODE .EQ. '-') PGUESS = .TRUE.
	  IF (CODE .EQ. 'e') PEMERG = .TRUE.
	  if (code .eq. 'R') code = '>'
	  if (code .eq. 'L') code = '<'
	  IF (CODE .EQ. '<' .OR. CODE .EQ. '>') SHIMP = .TRUE.
	  IF (CODE .EQ. 'l' .OR. CODE .EQ. 'r') SHGUES = .TRUE.
	  IF (CODE .EQ. 'u') SHEMRG = .TRUE.
          IF (CODE .EQ. 'F' .OR. CODE .EQ. 'B') SVIMP = .TRUE.
	  IF (CODE .EQ. 'H' .OR. CODE .EQ. 'V' 
     .	    .OR. CODE .eq. 'S') RATSYM = .TRUE.
	  IF (CODE .EQ. 'E') ERRORS = .TRUE.
	  GO TO 200
C
300	IF (PIMP) THEN
	  PIMP = TRUTH('Plot impulsive P polarities?..[Y]')
	  IF (PIMP) THEN
	    CDSIZE = VALUE
     +	      ('Enter size for impulsive P polarities [0.25]')
	      IF(CDSIZE.LE.0) CDSIZE=0.25
	    ICDPEN = VALUE('linewidth for impulsive P polarities..[1]')
	      IF (ICDPEN .LE. 0) ICDPEN = 1
	  END IF
	END IF
	IF (PGUESS) THEN
   	  PGUESS=TRUTH('Plot emergent P polarities?...[N]')
	  IF(PGUESS) THEN
	    EMGPSZ=VALUE
     +        ('Enter size for emergent P polarities [0.15]')
	      IF(EMGPSZ.LE.0.0) EMGPSZ=0.15
	    IEMGPI = VALUE('linewidth for emergent P polarities..[1]')
	      IF (IEMGPI .LE. 0) IEMGPI = 1
	  END IF
	END IF
	IF (PEMERG) THEN
	  PEMERG=TRUTH('Plot emergent P arrivals?..[N]')
	  IF (PEMERG) THEN
	    EMGASZ=VALUE
     +      ('Enter size for emergent P arrivals [0.15]')
	    IF(EMGASZ.EQ.0) EMGASZ=0.15
	    IEMGP = VALUE('linewidth for emergent P arrivals..[1]')
	      IF (IEMGP .LE. 0) IEMGP = 1
	  END IF
	END IF
	IF (SHIMP) THEN
	  SHIMP= TRUTH('Plot impulsive SH polarities?..[Y]')
	  IF (SHIMP) THEN
	    SHSIZE = VALUE
     +	      ('Enter size for impulsive SH polarities [0.25]')
	      IF(SHSIZE.LE.0) SHSIZE=0.25
	    ISH = VALUE('linewidth for impulsive SH polarities..[1]')
	      IF (ISH .LE. 0) ISH = 1
	  END IF
	END IF
	IF (SHGUES) THEN
   	  SHGUES=TRUTH('Plot emergent SH polarities?...[N]')
	  IF(SHGUES) THEN
	    EMGSHG=VALUE
     +        ('Enter size for emergent SH polarities [0.15]')
	      IF(EMGSHG.LE.0.0) EMGSHG=0.15
	    IEMSHI = VALUE('linewidth for emergent SH polarities..[1]')
	      IF (IEMSHI .LE. 0) IEMSHI = 1
	  END IF
	END IF
	IF (SHEMRG) THEN
	  SHEMRG=TRUTH('Plot emergent SH arrivals?..[N]')
	  IF (SHEMRG) THEN
	    EMGSH=VALUE
     +        ('Enter size for emergent SH arrivals [0.15]')
	        IF(EMGSH.EQ.0) EMGSH=0.15
	    IEMGSH = VALUE('linewidth for emergent SH arrivals..[1]')
	      IF (IEMGSH .LE. 0) IEMGSH = 1
	  END IF
	END IF
	IF (SVIMP) THEN
	  SVIMP= TRUTH('Plot impulsive SV polarities?..[Y]')
	  IF (SVIMP) THEN
	    SVSIZE = VALUE
     +	      ('Enter size for impulsive SV polarities [0.25]')
	      IF(SVSIZE.LE.0) SVSIZE=0.25
	    ISV = VALUE('linewidth for impulsive SV polarities..[1]')
	      IF (ISV .LE. 0) ISV = 1
	  END IF
	END IF
	IF (RATSYM) THEN
	  RATSYM=TRUTH('Plot ratios?..[Y]')
	  IF (RATSYM) THEN
	    RTSCLE = VALUE
     +        ('Size for ratio scaling factor..[0.20]')
	      IF(RTSCLE.EQ.0) RTSCLE=0.20
	    IRTSNP = VALUE('linewidth for ratios..[1]')
	      IF (IRTSNP .LE. 0) IRTSNP = 1
	  END IF
	END IF
	IF (ERRORS) THEN
	  ERRORS = TRUTH('Plot error box encircling symbol?..[Y]')
	  IF (ERRORS) THEN
	    IERPEN = VALUE('linewidth for error box..[1]')
	      IF (IERPEN .LE. 0) IERPEN = 1
	  END IF
	END IF
C
	IF (FIRST) THEN
	  CALL CIRPLT(RADIUS,TITLE,.TRUE.)
	  IF (TRUTH('Include time & file name?..[Y]')) THEN
	    IPEN = VALUE('Enter linewidth value..[1]..')
	      IF (IPEN .LE. 0) IPEN = 1
	    CALL linewidth(IPEN)
	    CALL SYMBOL(-1.5*RADIUS,RADIUS+0.5,0.15,FILENA,0.,NFILE)
	    CALL SYMBOL(-1.5*RADIUS,RADIUS+0.75,0.15,td,0.,24)
	    CALL TSEND
	  ENDIF
	  FIRST = .FALSE.
	END IF
	REWIND 1
	READ(1,'(1X)')
	WRITE(2,5)
400	READ(1,6,END=600) STA,AZIM,TOA,CODE,XLGRAT,SVSHP,STOANG,INFO
	IF ((CODE .EQ. 'V' .OR. CODE .EQ. 'H' 
     .	  	.OR. CODE .eq. 'S') .AND. RATSYM) then
	  WRITE(2,7) STA,AZIM,TOA,CODE,XLGRAT,CODE,SVSHP,STOANG,
     1		INFO(1:lenc(info))
	  CODE = 'X'
	  RATIO = 10**AMAX1(0.1,XLGRAT + 0.7)
	  RTSIZE = 0.1*RTSCLE*RATIO
	  GO TO 500
	END IF
	  IF((.NOT.RATSYM) .AND. (CODE .EQ.'V'
     1	 .OR.CODE.EQ.'H' .OR.CODE.eq.'S')) GO TO 400
	  if (code .eq. 'U') code = 'C'
	  IF(.NOT.PIMP.AND.(CODE.EQ.'C'.OR.CODE .EQ. 'D')) GO TO 400
	  IF(.NOT.PGUESS.AND.(CODE.EQ.'+'.OR.CODE.EQ.'-')) GO TO 400
	  IF(.NOT.PEMERG.AND.CODE.EQ.'e') GO TO 400
	  if (code .eq. 'R') code = '>'
	  if (code .eq. 'L') code = '<'
	  IF(.NOT.SHIMP.AND.(CODE.EQ.'<'.OR.CODE .EQ. '>')) GO TO 400
	  IF(.NOT.SHGUES.AND.(CODE.EQ.'l'.OR.CODE.EQ.'r')) GO TO 400
	  IF(.NOT.SHEMRG.AND.CODE.EQ.'u') GO TO 400
	  IF(.NOT.SVIMP.AND.(CODE.EQ.'F'.OR.CODE .EQ. 'B')) GO TO 400
	  WRITE(2,7) STA,AZIM,TOA,CODE
  500	IF (TOA .GT. 90.0) THEN
	  TOA = 180.0 - TOA
	  AZIM = AZIM + 180.0
	END IF
	IF (UPPER) AZIM = AZIM + 180.0
	IF (AZIM .GT. 360.0) AZIM = AZIM - 360.0
	AZR = AZIM/RD
  	R = RADIUS*SQ2*SIN(TOA/(2.0*RD))
	X = R*SIN(AZR)
	Y = R*COS(AZR)
	IF(CODE.EQ.'e') THEN
	  SIZE=EMGASZ
	  CALL linewidth(IEMGP)
	  CALL symbol(X-0.45*SIZE,Y-0.35*SIZE,SIZE,'e',0.0,1)
	  GO TO 400
	END IF
	IF (CODE .EQ.'-') THEN
	  SIZE=EMGPSZ
	  CALL linewidth(IEMGPI)
	  CALL symbol(X-0.65*SIZE,Y-0.375*SIZE,SIZE,'-',0.0,1)
	  GO TO 400
	END IF
	IF(CODE.EQ.'u') THEN
	  SIZE=EMGSH
	  CALL linewidth(IEMGSH)
	  CALL symbol(X-0.45*SIZE,Y-0.35*SIZE,SIZE,'u',0.0,1)
	  GO TO 400
	END IF
	IF (CODE .EQ.'r') THEN
	  SIZE=EMGSHG
	  CALL linewidth(IEMSHI)
	  CALL symbol(X-0.65*SIZE,Y-0.375*SIZE,SIZE,'r',0.0,1)
	  GO TO 400
	END IF
	IF (CODE .EQ.'l') THEN
	  SIZE=EMGSHG
	  CALL linewidth(IEMSHI)
	  CALL symbol(X-0.65*SIZE,Y-0.375*SIZE,SIZE,'l',0.0,1)
	  GO TO 400
	END IF
	ANGLE = 0.0
	IF (CODE .EQ. 'C' .OR. CODE .EQ. 'D') THEN
	  NS = 1
	  SIZE = CDSIZE
	  CALL linewidth(ICDPEN)
	  IF (CODE .EQ.'D') NS=2
	ELSE IF (CODE .EQ.'+') THEN
	  NS=3
	  SIZE=EMGPSZ
	  CALL linewidth(IEMGPI)
	ELSE IF (CODE .EQ.'X') THEN
	  NS=4
	  SIZE=RTSIZE
	  CALL linewidth(IRTSNP)
	ELSE IF (CODE .EQ. '<' .OR. CODE .EQ. '>') THEN
	  SIZE = SHSIZE
	  CALL linewidth(ISH)
	  NS = 6
	  IF (CODE .EQ. '<') ANGLE = 450.0 - AZIM 
	  IF (CODE .EQ. '>') ANGLE = 270.0 - AZIM 
	ELSE IF (CODE .EQ. 'F' .OR. CODE .EQ. 'B') THEN
	  SIZE = SVSIZE
	  CALL linewidth(ISV)
	  NS = 6
	  IF (CODE .EQ. 'F') ANGLE = 360.0 - AZIM 
	  IF (CODE .EQ. 'B') ANGLE = 180.0 - AZIM 
	ELSE IF (CODE .EQ. 'E') THEN
	  CALL linewidth(IERROR)
	  NS = 0
	ELSE 
	  GO TO 400
	END IF
	CALL SPCSMB(X,Y,SIZE,NS,ANGLE,-1)
	GO TO 400  
  600	CALL TSEND
	IF (TRUTH('Add more data to plot?..[N]')) GO TO 100
	RETURN
C
4	FORMAT(5X,A1,A)
5	FORMAT(' Statn',T9,'Azimuth',T18,'TOAngl',T26,
     1    'Key',T31,'Log10 Ratio',T45,
     2    '  Pol',T52,'TOAng2',T60,'Comment')
6	FORMAT(A4,2F8.2,A1,F8.4,1X,A1,1X,F6.2,1X,A)
7	FORMAT(2X,A4,T10,F5.1,T19,F5.1,T27,A1,T32,F8.4,T41,'S',A1,T47,
     .    A1,T53,F6.2,T60,A)
	END
