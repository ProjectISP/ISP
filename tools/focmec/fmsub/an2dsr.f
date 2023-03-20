C+
C	SUBROUTINE AN2DSR(A,N,ANGS,PI)
C
C	Calculates dip, strike and rake (ANGS) - A&R convention,
C		from A and N.
C	12 January 2000:  Fixed a divide by zero when angs(1) .eq. 0
C	1 October 2001: When porting to the PC, there were roundoff
C		errors when acos was near its limits.
C-
	SUBROUTINE AN2DSR(A,N,ANGS,PI)
	REAL N(3),A(3),ANGS(3)
	if (N(3) .eq. -1.0) then
	  angs(2) = atan2(a(2),a(1))
	  angs(1) = 0.0
	else
	  ANGS(2) = ATAN2(-N(1),N(2))
	  if (N(3) .eq. 0.0) then
	    angs(1) = 0.5*PI
	  else IF (ABS(SIN(ANGS(2))) .ge. 0.1) then
	    ANGS(1) = ATAN2(-N(1)/SIN(ANGS(2)),-N(3))
	  else
	    ANGS(1) = ATAN2(N(2)/COS(ANGS(2)),-N(3))
	  end if
	end if
	A1 = A(1)*COS(ANGS(2)) + A(2)*SIN(ANGS(2))
	if (abs(a1) .lt. 0.0001) a1 = 0.0
	if (a(3) .ne. 0.0) then
	  if (angs(1) .ne. 0.0) then
	    ANGS(3) = ATAN2(-A(3)/SIN(ANGS(1)),A1)
	  else
	    ANGS(3) = atan2(-1000000.0*A(3),A1)
	  end if
	else
	  a2 = a(1)*sin(angs(2)) - a(2)*cos(angs(2))
	  if (abs(a2) .lt. 0.0001) a2 = 0.0
	  if (abs(sin(2*angs(2))) .ge. 0.0001) then
	    angs(3) = atan2(a2/sin(2*angs(2)),a1)
	  else if (abs(sin(angs(2))) .ge. 0.0001) then
	    acosarg = amin1(1.0,amax1(-1.0,a(2)/sin(angs(2))))
	    angs(3) = acos(acosarg)
	  else
	    acosarg = amin1(1.0,amax1(-1.0,a1))
	    angs(3) = acos(a1)
	  end if
	end if
	IF (ANGS(1) .lt. 0.0) then
	  ANGS(1) = ANGS(1) + PI
	  ANGS(3) = PI - ANGS(3)
	  IF (ANGS(3) .GT. PI) ANGS(3) = ANGS(3) - 2*PI
	end if
	IF(ANGS(1) .gt. 0.5*PI) then
	  ANGS(1)=PI-ANGS(1)
	  ANGS(2)=ANGS(2)+PI
	  ANGS(3)=-ANGS(3)
	  IF (ANGS(2) .GE. 2*PI) ANGS(2) = ANGS(2) - 2*PI
	end if
	IF (ANGS(2) .LT. 0.0) ANGS(2) = ANGS(2) + 2.0*PI
	RETURN
	END
