C+
C	PROGRAM Ratplt
C
C	Plots amplitude ratio observed minus calculated.
C	May 2017  changed allowed range from a cross to a star  
C
C   Read in the ratios from a Focmec input file
        call ratinp
C   Input a DSR and calculate difference  between observed and calculaed ratio
        call ratdif
C   Plot the o-c ratios and the allowed ranges
        CALL OMCPLT
	STOP
	END
C+
	SUBROUTINE ratinp
C
C	Input routine for Ratplt
C
C       Slimmed down focinp.f
C       Asks or a focmec.inp file and selects ratio lines. Stores in 
C         ratplt.inc
C-
	INCLUDE 'ratplt.inc'
	CHARACTER*1 SENSE,SNUMPOL
	CHARACTER*80 FILENA,CVALUE,DUMMY,commnt
	LOGICAL TRUTH
	CHARACTER*40 INFO
	RD = 45.0/ATAN(1.0)
        open(unit=2,file='ratplt.lst',status='unknown')
        call timdat(2,'Ratplt')
        COMMNT =
     1    CVALUE('Comment - up to 80 characters',DUMMY,NCOM)
	WRITE(2,'(1X,A)') COMMNT(1:NCOM)
100     FILENA = CVALUE('Focmec input filespec',DUMMY,NFILE)
	OPEN(UNIT=1,FILE=FILENA(1:NFILE),STATUS='OLD',ERR=100)
	READ(1,'(A)') COMMNT
	ncom = lenc(commnt)
	WRITE(*,'(1X,A)') COMMNT(1:NCOM)
	IF (.NOT.TRUTH('Correct file?...[Y]')) GO TO 100
	WRITE(2,3) FILENA(1:NFILE)
	WRITE(2,'(1X,A)') COMMNT(1:NCOM)
	WRITE(2,4)
	J = 0
200	read(1,'(a)',end=300) commnt
	ncom = lenc(commnt)
	if (ncom.lt.22 .or. commnt(24:29).eq.'      ') then
c   toang1 hers is the take-off angle for the P or S ray with the polarity
	  read(commnt(1:21),'(A4,2F8.2,A1)') STA,AZIN,TOANG1,SENSE
        if (ncom .gt. 21) then
            read(commnt(40:ncom),'(a)') info
c            write(*,'(i3, '' '',a)') lenc(info),info(1:lenc(info))
        else
	      info = ' '
        endif
	else
C   toang1 here is the take-off angle for the numerator ray
C   toang2 is the take-off angle for the denominator ray in a ratio
	  READ(commnt,5) STA,AZIN,TOANG1,SENSE,
     .	    RATLOG,SNUMPOL,TOANG2,VPVS
        info = ' '
        if (ncom .gt. 45) read(commnt(46:ncom),'(a)') info
      endif
        IF (SENSE.EQ.'V' .OR. SENSE.EQ.'S' .OR. SENSE.EQ.'H') THEN
	    J = J + 1
	    IF (J .GT. MAX2) GO TO 300
	    NRAT = J
	    LOGRAT(NRAT) = RATLOG
	    SVSH(2,NRAT) = SNUMPOL
	    RSTATN(NRAT) = STA
            rtoa1(nrat) = toang1
            rtoa2(nrat) = toang2
            az(nrat) = azin
	    NADD = 0
	    SVSH(1,NRAT) = SENSE
	    IF (SENSE .EQ. 'H') NADD = 1000
	    IF (SENSE .EQ. 'S') NADD = 2000
	    WRITE(2,6) RSTATN(NRAT),AZIN,TOANG1,SENSE,LOGRAT(NRAT),
     .	      SVSH(1,NRAT),SVSH(2,NRAT),TOANG2,INFO(1:lenc(info))
	    KEYRAT(NRAT) = J + NADD
        else
            goto 200
        endif
        TREND = AZIN/RD
        toa = TOANG1/RD 
        COST = COS(TREND)
        SINT = SIN(TREND)
        sinP = sin(toa)
        cosP = cos(toa)
        XYZ(1,J) = COST*sinp
        XYZ(2,J) = SINT*sinP
c   Positive z is down
        XYZ(3,J) = cosP
C  Next two vectors reversed in sign from A&R convention because
C   of my convention for SV and SH (down and left, facing the station)
        XYZ(4,J) = -COST*cosP
        XYZ(5,J) = -SINT*cosP
        XYZ(6,J) = sinp 
        XYZ(7,J) = SINT
        XYZ(8,J) = -COST
        XYZ(9,J) = 0.0
        IF (SENSE.EQ.'V' .OR. SENSE.EQ.'S' .OR. SENSE.EQ.'H') THEN
            toa = toang2/rd
            sinP = sin(toa)
            cosP = cos(toa)
            XYZden(1,J) = COST*sinp
            XYZden(2,J) = SINT*sinP
	    XYZden(3,J) = cosP
            XYZden(4,J) = -COST*cosP
	    XYZden(5,J) = -SINT*cosP
	    XYZden(6,J) = sinp
        endif
        GO TO 200
300	CLOSE(UNIT=1)
	IF (NRAT .LE. 0) THEN
	  WRITE(2,12)
          stop
	ELSE
	  ERRRAT = RVALUE('Reference ratio log10 |o-c|..[0.1]',0.1)
	  CUTP = RVALUE('CUTP: lower-limit P cutoff...[0.1]',0.1)
	  CUTS = RVALUE('CUTS: lower-limit S cutoff...[0.1]',0.1)
	  WRITE(2,13) NRAT,ERRRAT,VPVS
	  WRITE(2,'(a,2(f5.3,a))') 'For ratios,  ', CUTP,
     1     ' = P radiation cutoff  ',CUTS,' = S radiation cutoff'
	  VPVS3 = VPVS**3
	ENDIF
	RETURN
C
3	FORMAT(1X,'Input from a file ',A)
4	FORMAT(/' Statn',T9,'Azimuth',T18,'TOAng',T26,
     1    'Key',T31,'Log10 Ratio',T44,
     2    'NumPol',T52,'DenTOAng',T62,'Comment')
5	FORMAT(A4,2F8.2,A1,F8.4,1X,A1,1X,F6.2,1X,f6.4)
6	FORMAT(2X,A4,T10,F5.1,T19,F5.1,T27,A1,T32,F8.4,T41,'S',A1,T47,
     .    A1,T53,F6.2,T60,A)
7	FORMAT(' Input:',I4,' ratios')
10	FORMAT(2X,A4,T10,F5.1,T19,F5.1,T27,A1,T60,A)
12	  FORMAT(' There are no amplitude ratio data')
13        FORMAT(' Input',I3,' ratios',' Reference |o-c| of',
     2      F7.4,'  VP/VS =',F6.3)
	END
C+
        subroutine ratdif
C
C   Prompts for DSR.  Calls okrat to find DIFF for each ratio.Prints
C     ratio stuff as in oksol
C-
        INCLUDE 'ratplt.inc'
	CHARACTER*1 POL,MARK(3)
	REAL*4 DSR(3), DSRD(3), SPOL(MAX), A(3), N(3)
	DATA MARK/'*',' ','#'/
        CALL PRINTX('Enter Dip, Strike and Rake (degrees)')
        READ(*,*) (DSRD(J),J=1,3)
        WRITE (2,1) (DSRD(I),I=1,3)
1	FORMAT(5X,'Dip  Strike  Rake ',3F9.2)
        do j=1,3
          dsr(j) = dsrd(j)/rd
        enddo
        WRITE(2,7)
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
	    DIFF(K) = 0.0
	  ELSE
	    DIFF(K) = LOGRAT(K) - CALRAT(K)
	  END IF
          IF (ABS(DIFF(K)) .GT. ERRRAT) THEN
	    WTRAT(K) = 0
	  ELSE
	    WTRAT(K) = 1
	  ENDIF
	  IF (SVSH(1,K) .EQ. 'V' .AND. SPOL(K) .GT. 0.0) POL = 'F'
	  IF (SVSH(1,K) .EQ. 'V' .AND. SPOL(K) .LT. 0.0) POL = 'B'
	  IF (SVSH(1,K) .EQ. 'S' .AND. SPOL(K) .GT. 0.0) POL = 'F'
	  IF (SVSH(1,K) .EQ. 'S' .AND. SPOL(K) .LT. 0.0) POL = 'B'
	  IF (SVSH(1,K) .EQ. 'H' .AND. SPOL(K) .GT. 0.0) POL = 'L'
	  IF (SVSH(1,K) .EQ. 'H' .AND. SPOL(K) .LT. 0.0) POL = 'R'
	  if (FLAG(k) .ne. 'N&D') then
	    WRITE(2,8) LOGRAT(K),CALRAT(K),DIFF(K),MARK(WTRAT(K)+1),
     .	      RSTATN(K),SVSH(1,K),SVSH(2,K),POL,FLAG(K)
	  else
	    WRITE(2,21) LOGRAT(K),MARK(3),
     .	      RSTATN(K),SVSH(1,K),SVSH(2,K),POL,FLAG(K)
	  endif
	END DO
	RETURN
C
7	FORMAT(/T11,'Log10(Ratio)',T53,'Ratio',
     .    T63,'S Polarity'/T6,'Observed',
     1    T16,'Calculated',T30,
     2    'Difference',T42,'Station',T53,' Type',T63,'Obs.',
     3    T69,'Calc.',T75,'Flag')
8	FORMAT(4X,F8.4,T17,F8.4,T30,F8.4,T43,A1,A4,T55,'S',A1,T64,A1,
     .    T71,A1,T76,A3)
21	FORMAT(4X,F8.4,T43,A1,A4,T55,'S',A1,T64,A1,
     .    T71,A1,T76,A3)
	END
C+
        Subroutine omcplt
C
C       downsized prplot to just work with ratios.
C       diff is log10 (observed - calc) ratio  Plots diff and allowed.
c	June 2016:  Allows for title (.true. to cirplt)
C-
        INCLUDE 'ratplt.inc'
	CHARACTER*80 COM, CVALUE, DUMMY
	CHARACTER*40 TITLE
	LOGICAL TRUTH, UPPER,ASK/.true./
	DATA RADIUS,R2/2.3,5.6/
C
	SQ2 = SQRT(2.0)
	UPPER = TRUTH('Upper hemisphere projection?..[N]') 
        RTSCLE = RVALUE('Plotting scale factor..[0.20]',0.20)
	write(2,'(f4.2,a)')rtscle,' is plotting scale factor' 
C
        CALL CIRPLT(RADIUS,TITLE,ASK)
        ersize = RTSCLE*10**errrat
        CALL linewidth(1)
        do j=1,nrat
	  if (flag(j) .ne. 'N&D') then
            RTSIZE = RTSCLE*10**abs(diff(j))
            toa = rtoa1(j)
            azim = az(j)
            IF (TOA .GT. 90.0) THEN
	      TOA = 180.0 - TOA
	      AZIM = AZIM + 180.0
	    END IF
	    IF (UPPER) AZIM = AZIM + 180.0
	    IF (AZIM .GT. 360.0) AZIM = AZIM - 360.0
	    AZR = AZIM/RD
  	    R = RADIUS*SQ2*SIN(TOA/(2.0*RD))
            X = R*SIN(AZR)
	    Y = R*COS(AZR)	
            ANGLE = 0.0
	    if (svsh(1,j) .eq. 'V') then 
              NS=0
	    elseif (svsh(1,j) .eq. 'H') then
              NS = 5
            else
              NS = 1
            endif
            CALL SPCSMB(X,Y,RTSIZE,NS,ANGLE,-1)
	    NS=12               	
            CALL SPCSMB(X,Y,ERSIZE,NS,ANGLE,-1)
          endif
	enddo  
  	CALL TSEND
 	IF(TRUTH('Add a plot label?')) CALL PLTLAB(R2,R2)
	CALL PLOT(0.0,0.0,999)
        RETURN
	END
