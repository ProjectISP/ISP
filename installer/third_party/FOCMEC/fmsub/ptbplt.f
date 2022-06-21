C+
	SUBROUTINE PTBPLT(RADIUS,PTBSIZ,PTTP,BTP,UPPER,CHOICE,IPEN)
C
C     Reads and plots P, T, and B axes from focal mechanism solutions.
C
C	29 Septmeber 1985:  Added upper hemisphere option
C	6 February 1986:  Added CHOICE which allows for P, T, or B
C	  individually, and IPEN which allows different linewidths
C	17 February 1992:  Added RADIUS as input veriable
C-
	DIMENSION TREND(3),PLUNGE(3),PTTP(4),BTP(2)
	character*1 axis(3)
	INTEGER IPEN(3)
	LOGICAL UPPER, CHOICE(3)
	DATA RD/57.29578/
	SQ2=SQRT(2.0)
	TREND(1)=PTTP(1)
	PLUNGE(1)=PTTP(2)
	AXIS(1)='P'
	TREND(2)=PTTP(3)
	PLUNGE(2)=PTTP(4)
	AXIS(2)='T'
	TREND(3)=BTP(1)
	PLUNGE(3)=BTP(2)
	AXIS(3)='B'
	IF (UPPER) THEN
	  DO 100 I=1,3
	    TREND(I) = TREND(I) + 180.0
100	  CONTINUE
	END IF
	DO 200 I=1,3
	  IF (.NOT.CHOICE(I)) GO TO 200
	  PLUNGE(I) = 90.0 - PLUNGE(I)
  	  R = RADIUS*SQ2*SIN(PLUNGE(I)/(2.0*RD))
	  IP = IPEN(I)
	  CALL linewidth(IP)
	  CALL SYMBOL(R*SIN(TREND(I)/RD),R*COS(TREND(I)/RD),PTBSIZ,
     1	    AXIS(I),0.0,1)
200	CONTINUE
  	CALL TSEND
	RETURN
	END
