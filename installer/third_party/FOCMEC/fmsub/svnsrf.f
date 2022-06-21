C+
	SUBROUTINE SVNSRF(PI,MT,R,ALINE,DASHL,SIZE,UPPER)
C
C	Calculates and plots the projections of the SV nodal surfaces.
C	Based on Bruce Julian's code.
C-
	character*4 aline
	LOGICAL XRTMAX,YTPMAX
	REAL X(181), Y(181)
	INTEGER SEGST(10), NTOT(10)
C  moment tensor
	REAL MT(6)
	LOGICAL SVNODE, SVNNXT
	LOGICAL RCLIP
	LOGICAL PENUP
C  loop index
	INTEGER I
C  focal-sphere coordinates along contour
	REAL CI1, CI2, CZ1, CZ2
C k = data, j = node (but j is not used)
	INTEGER K
	DATA XRTMAX,YTPMAX/2*.TRUE./
	DO 100 I=1,10
	  NTOT(I) = 0
	  SEGST(I) = 0
100	CONTINUE
	IF ((MT(4) .EQ. 0.0) .AND. (MT(5) .EQ. 0)) THEN
	  IF ((MT(2)+MT(3)-2.0*MT(1)) .EQ. 0.0) THEN
	    NCURVE = 2
	    AZ = 0.5*ATAN2(MT(2)-MT(3),-2.0*MT(6))
	    SAZ = SIN(AZ)
	    CAZ = COS(AZ)
	    SEGST(1) = 1
	    SEGST(2) = 3
	    NCURVE = 2
	    NTOT(1) = 2
	    NTOT(2) = 2
	    X(1) = R*SAZ
	    Y(1) = R*CAZ
	    X(2) = -X(1)
	    Y(2) = -Y(1)
	    X(3) = Y(1)
	    Y(3) = X(2)
	    X(4) = -X(3)
	    Y(4) = -Y(3)
	  END IF
	ELSEIF ((MT(1) .EQ. 0.0) .AND. (MT(2) .EQ. 0.0) .AND.
     .	    (MT(3) .EQ. 0.0) .AND. (MT(6) .EQ. 0.0)) THEN
	  AZ = ATAN2(MT(4),MT(5))
	  CAZ = COS(AZ)
	  SAZ = SIN(AZ)
	  NCURVE = 2
	  SEGST(1) = 1
	  NTOT(1) = 2
	  X(1) = R*SIN(AZ)
	  Y(1) = R*COS(AZ)
	  X(2) = -X(1)
	  Y(2) = -Y(1)
	  SEGST(2) = 3
	  NTOT(2) = 91
	  DR4 = 4.0*ATAN(1.0)/45.0
	  R45 = SQRT(2.0)*R*SIN(22.5*DR4/4.0)
	  DO 200 JJ=1,91
	    ANG = DR4*(JJ-1)
	    X(JJ+2) = R45*COS(ANG)
	    Y(JJ+2) = R45*SIN(ANG)
200	  CONTINUE
	ELSE
	  NCURVE = 0
	  K = 0
	  DO 400 I = 1, 4
	    IF (.NOT.(SVNODE(MT, 0.05*(0.5*PI), CI1, CZ1))) GOTO 400
	    PENUP = .TRUE.
	    IF (I .GT. 2) CI1 = CI1 + 0.5*PI
	    IF ((I .EQ. 2) .OR. (I .EQ. 4)) THEN
	      CI1 = PI - CI1
	      CZ1 = CZ1 + PI
	    END IF
300	    IF (.NOT.SVNNXT(CI2, CZ2)) GO TO 400
	    IF (I .GT. 2) CI2 = CI2 + 0.5*PI
	    IF ((I .EQ. 2) .OR. (I .EQ. 4)) THEN
	      CI2 = PI - CI2
	      CZ2 = CZ2 + PI
	    END IF
	    IF (RCLIP(CI1, CZ1, CI2, CZ2, (0.5*PI))) THEN
	      IF (PENUP) THEN
	        K = K + 1
	        NCURVE = NCURVE + 1
	        SEGST(NCURVE) = K
	        NTOT(NCURVE) = 1
	        CALL TA2XY(CI1, CZ1, X(K), Y(K))
	        PENUP = .FALSE.
	      END IF
	      K = K + 1
	      NTOT(NCURVE) = NTOT(NCURVE) + 1
	      CALL TA2XY(CI2, CZ2, X(K), Y(K))
	    ELSE IF (.NOT.PENUP) THEN
	      PENUP = .TRUE.
	    END IF
	    CI1 = CI2
	    CZ1 = CZ2
	    GO TO 300
400	  CONTINUE
	END IF
	IF (NCURVE .EQ. 0) RETURN
	DO 500 I=1,NCURVE
	  NN = NTOT(I)
	  NSTART = SEGST(I)
	  CALL PLTDAT(X(NSTART),0.0,-R,1.0,-R,R,
     .	    Y(NSTART),-R,-R,R,1.0,NN,ALINE,DASHL,SIZE,
     .	    NSYMBL,1,'XLIN','YLIN',XRTMAX,YTPMAX)
500	CONTINUE
	CALL TSEND
	RETURN
	END
