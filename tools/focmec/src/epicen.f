C+
	program epicen
C
C	Input an earthquakes epicenter and a set of station locations.
C	Output is the epicentral distance, azimuth and back azimuth.
C	For the km distance option, use Julian's approximation of local radius.
C	9 January 2004: Uses the SAC method to find epicentral dist. in km.
C       8 June 2011:  Allowed for 5 character stations.  Changed stations.f
C	July 2016:  Increaed numer of decimal places for epicenter & 
C	  distances use my disaz that may have higher precision
C-
	character*80 com,filename
	CHARACTER*4 DEGKM(2)
        character*5 stanam(15),sta,blank
	DATA DEGKM/' Deg',' Km '/,deg2km/111.195/
	LOGICAL TRUTH,DEG,FIRST
	DATA BLANK/'     '/,FIRST/.TRUE./
        character*40 arcdatdir,ARCDAT/'ARCDAT'/
	real*8 elatin/0.0d0/,elongin/0.0d0/
C
	rd = 45.0/atan(1.0)
	open(unit=2,file='epicen.lst',status='unknown')
        call getenv(ARCDAT,arcdatdir)
        if (arcdatdir(1:1) .eq. '/') then
          filename = arcdatdir(1:lenc(arcdatdir))//'/stations.loc'
        else
          filename = 'stations.loc'
        end if
	open(unit=3,file=filename,status='old')
	DEG = TRUTH('Epicentral distances in degrees?..[Y]')
	if (deg) id = 1
	if (.not.deg) id = 2
80	CALL PRINT('Comment line - up to 80 characters')
	READ(*,'(A)') com
90	CALL PRINTX('Earthquake latitude and longitude')
	READ(*,*) ELATIN,ELONGIN
	elat = sngl(elatin)
	elong = sngl(elongin)
	elatr = elat/rd
	elongr = elong/rd
	IF (.NOT.FIRST) FIRST=.NOT.TRUTH('Same station set?..[Y]')
	IF (FIRST) THEN
	  NSUM = VALUE('Enter number of stations - up to 15 (15A5)')
	  WRITE(*,5) NSUM,NSUM
	  READ(*,16) (STANAM(JJ),JJ=1,NSUM)
	  write(*,16) (STANAM(JJ),JJ=1,NSUM)
	END IF
	CALL TIMDAT(2,'epicen')
	write(2,'(1X,a)') com(1:lenc(com))
	write(2,'(a,f10.4,a,f10.4)')
     1      'Latitude:',elat,' Longitude:',elong
C
150	DO 600 N=1,NSUM
	  call uppercase(stanam(N))
	  CALL STATIONS(3,STANAM(N),SLAT,SLONG)
	  IF (STANAM(N) .EQ. BLANK) GO TO 600
	  STA = STANAM(N)
	  slatr = slat/rd
	  slongr = slong/rd
c	    CALL DISTAZ_sac(ELAT,ELONG,SLAT,SLONG,dist,AZ,BAZ,gcp)
	  CALL DISAZ(ELATr,ELONGr,SLATr,SLONGr,AZr,BAZr,gcp)
c	  if (N.eq.7) write(2,*) ELATr,ELONGr,SLATr,SLONGr,AZr,gcp
	  az = azr*rd
	  baz = bazr*rd
	  EDIST = gcp*rd
	  IF (.NOT.DEG) EDIST = deg2km*edist
	  WRITE(2,'(5X,A4,A,F9.3,A4,A,F7.3,A4,A,F7.3,A4)')
	1     STA,' Dist. = ',EDIST,degkm(ID),'  Az. = ',AZ,
	2    degkm(1),'  Back Az. = ',BAZ,degkm(1)
	  WRITE(*,'(5X,A4,A,F10.4,A4,A,F7.3,A4,A,F7.3,A4)')
	1     STA,' Dist. = ',EDIST,degkm(ID),'  Az. = ',AZ,
	2    degkm(1),'  Back Az. = ',BAZ,degkm(1)
600	CONTINUE
	CALL IYESNO('Run some more?',IANS)
	IF(IANS.EQ.0) GO TO 1000
	FIRST = .FALSE.
	CALL IYESNO('Different comment?',IANS)
	IF(IANS.EQ.1) GO TO 80
	GO TO 90
 1000 WRITE(2,'('' '')')
    5 FORMAT(' Enter ',I2,' station names in ',I2,'A5 format - LEFT',
     :       ' justified')
c   11 FORMAT(' Earthquake latitude is ',F9.5,' longitude is ',F10.5)
16	FORMAT(15A5)
	STOP
	   END
