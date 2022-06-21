C+
	SUBROUTINE SOLPLT(R,DATPLT,TD,UPPER)
C
C	Prompts one for type of focal mechanism solution to be plotted
C	(either P, T and B axes or P, SV, or SH nodal surfaces).
C	 Asks if the solutions are to be entered from a file or
C	 interactively. Reads in solutions in the form of Dip, Strike
C	and Rake for one of the two possible fault planes, and calls
C	 the solution transformation and plotting routines.
C	SV and SH nodal surfaces cam originally (I think) from Bruce
C	  Julian (usgs/Menlo)
C	The projection is equal area (Schmidt net).
C
C	9 September 1985:  Added upper hemisphere plot option
C	1 October 1985:  Added SH and SV nodal surface options
C	6 February 1986:  Added choices and linewidth for P,T,B plot
C	9 July 1990: Added flexibility for reading from a file.  Don't
C	  need all same number of header lines.
C	15 February 1992  sun version
C	10 July 1993:  Moved time/date/filename display to right.
C	2 April 2009: Added wording to clarify possibility of adding
C	  more solutions or putting different representations of solutions.
C	June 2009: Previously if adding solutions and reading from a file,
C	  did not check to see if the file is correct.  Now it does.  Prompt
C	  for adding file name to the plot is still only doen if file is
C	  called first.
C-
	REAL*4 MOMTEN(6)
	DIMENSION PTTP(4),ANGS(3),ANGS2(3),ANBTP(6)
	INTEGER IPENS(3)
	LOGICAL CHOICE(3),FIRST,TRUTH,DATPLT,PT,DSR,UPPER
	LOGICAL PTB,FILE,PNODE,SHNODE,SVNODE,MORE,mt
	character*4 aline
	character*24 TD
	CHARACTER*40 TITLE
	CHARACTER*80 COM, FILENA, CVALUE, DUMMY, LINE
	INTEGER LUN1, LUN2, SOLPEN
	DATA PT,mt,DSR/2*.FALSE.,.TRUE./
C
	PI = 4.0*ATAN(1.0)
	FIRST = .TRUE.
	IF (TRUTH('Print/display solution summaries?..[N]')) THEN
	  LUN1 = 2
	  LUN2 = 5
	ELSE
	  LUN1 = 0
	  LUN2 = 0
	END IF
	WRITE(*,*) ' Can plot P, SV & SH nodal surfaces or P, T & B axes'
	WRITE(*,*) ' If you want to superimpose both, answer yes to more'
100	PTB=TRUTH('Plot P, T and B axes?  [Y]')
	IF(PTB) THEN 
	  PTBSIZ=VALUE(
     +      'Enter size for P, T and B axes symbols [0.15]')
	  IF(PTBSIZ.EQ.0) PTBSIZ = 0.15
	  IF (TRUTH('Plot all three?..[Y]')) THEN
	    DO 200 J=1,3
	      CHOICE(J) = .TRUE.
200	    CONTINUE
	  ELSE
	    CHOICE(1) = TRUTH('Plot P axes?..[Y]')
	    CHOICE(2) = TRUTH('Plot T axes?..[Y]')
	    CHOICE(3) = TRUTH('Plot B axes?..[Y]')
	  END IF
	  IF (CHOICE(1)) IPENS(1) = IVALUE('IPEN for P..[1]',1)
	  IF (CHOICE(2)) IPENS(2) = IVALUE('IPEN for T..[1]',1)
	  IF (CHOICE(3)) IPENS(3) = IVALUE('IPEN for B..[1]',1)
	ELSE
	  PNODE = .FALSE.
	  SHNODE = .FALSE.
	  SVNODE = .FALSE.
	  IF (TRUTH('P nodal planes..[Y]')) THEN
	    PNODE = .TRUE.
	    write(2,*) 'Plotting P nodal planes'
	  ELSE IF (TRUTH('SH nodal surfaces..[Y]')) THEN
	    SHNODE = .TRUE.
	    write(2,*) 'Plotting SH nodal surfaces'
	  ELSE
	    SVNODE = .TRUE.
	    write(2,*) 'Plotting SV nodal surfaces'
	  END IF
	  IF (.NOT.PNODE) CALL TA2XYI(R,UPPER)
	  ALINE = 'LINE'
	  IF (TRUTH('Dashed line?..[N]')) THEN
	    ALINE = 'DASH'
	    DASHL = VALUE('Enter length of dash  [0.1]')
	      IF (DASHL .LE. 0.0) DASHL = 0.1
	    SIZE = VALUE('Spacing between dashes [0.1]')
	      IF (SIZE .LE. 0.0) SIZE = 0.1
	    SIZE = SIZE + DASHL
	  END IF
	  SOLPEN = VALUE('Enter linewidth for solutions..[1]')
	  IF (SOLPEN .LE. 0) SOLPEN = 1
	END IF
	IF (FIRST) THEN
	  IF(DATPLT) THEN
	    IF(.NOT.TRUTH('Solutions on same plot as data?..[Y]')) THEN 
	      CALL PLOT(0.0,0.0,999)
     	      CALL CIRPLT(R,TITLE,.TRUE.)
	    END IF
	  ELSE
	    CALL CIRPLT(R,TITLE,.TRUE.)
	  END IF
	END IF
	WRITE(2,'('' '')')
	FILE=TRUTH('Input solutions from a file?..[Y]')
	IF(FILE) THEN
300	  FILENA = CVALUE('Input file name',DUMMY,NFILE)
	  CLOSE(UNIT=1)
	  OPEN(UNIT=1,file=FILENA(1:NFILE),status='OLD',ERR=300)
	  READ(1,'(A)') com
	  READ(1,'(A)') COM
	  ncom = lenc(com)
	  WRITE(*,'(1X,A)') COM(1:NCOM)
	  IF (.NOT.TRUTH('Correct file?  [Y]')) GO TO 300
	  BACKSPACE 1
	  BACKSPACE 1
	  IF (FIRST) THEN
	    IF (TRUTH('Plot time & file name?..[Y]')) THEN
	      IPEN = VALUE('Enter linewidth..[1]')
	      IF (IPEN .LE. 0) IPEN = 1
	      CALL linewidth(IPEN)
    	      CALL LENGTH(XLEN,0.15,FILENA,NFILE)
    	      CALL LENGTH(XLEN2,0.15,td,24)
	      CALL SYMBOL(1.5*R-XLEN,R+0.50,0.15,FILENA,0.,NFILE)
	      CALL SPCSMB(1.5*R-XLEN2,R+0.75,0.15,td,0.,24)
	      CALL TSEND
	    END IF
	  ENDIF
	  WRITE(2,*) ' Solutions input from file ', FILENA(1:NFILE)
	  MORE = .TRUE.
	  DO WHILE (MORE)
	    READ(1,'(A)') LINE
	    IF (LINE(1:7) .EQ. '    Dip') MORE = .FALSE.
	    IF (MORE) WRITE(2,*) LINE(1:LENC(LINE))
	  END DO
	END IF
	IF (LUN1 .NE. 0) WRITE(2,'('' '')')
	IF (.NOT.PTB) CALL linewidth(SOLPEN)
C
C	Start loop
C
400	IF(FILE) THEN    
	  READ(1,*,END=500) (ANGS(I),I=1,3)
	ELSE
	  CALL PRINTX('Enter dip, strike, rake for fault plane (deg.)') 
	  READ(*,*) (ANGS(I),I=1,3)
	END IF
	CALL FMREPS(ANBTP,ANGS,PTTP,ANGS2,PT,DSR,mt,MOMTEN,LUN1,LUN2)
	IF (PTB) THEN
	  CALL PTBPLT(R,PTBSIZ,PTTP,ANBTP(5),UPPER,CHOICE,IPENS)
	ELSE
	  IF (SHNODE) THEN
	    CALL SHNSRF(PI,MOMTEN,R,ALINE,DASHL,SIZE,UPPER)
	  ELSE IF (SVNODE) THEN
	    CALL SVNSRF(PI,MOMTEN,R,ALINE,DASHL,SIZE,UPPER)
	  ELSE
	     CALL PLNPLT(R,ANGS,ANGS2,ALINE,DASHL,SIZE,UPPER)
	  END IF
	END IF
	IF (FILE) GO TO 400
500	IF (FILE) CLOSE(UNIT=1)
	FIRST = .FALSE.
	write(*,*)' Can superimpose P,T,B, nodal planes, etc.' 
	write(*,*)'   E.g., a dashed lines for a second solution' 
	IF (TRUTH('Add more solutions?..[N]')) GO TO 100
	RETURN
	END
