C+
	SUBROUTINE V2TRPL(XYZ,TRPL,PI)
C
C	Transforms from XYZ components of a unit vector to
C	  the trend and plunge for the vector.
C	Trend is the azimuth (clockwise from north looking down)
C	Plunge is the downward dip measured from the horizontal.
C	All angles in radians
C	X is north, Y is east, Z is down
C	If the component of Z is negative (up), the plunge,TRPL(2),
C	  is replaced by its negative and the trend, TRPL(1),
C	  Is changed by PI.
C	The trend is returned between 0 and 2*PI, the plunge
C	  between 0 and PI/2.
C	12 January 2000: If xyz(3) = -1.0, make the trend PI.  Made
C	  consistency in the roundoff -- all are now 0.0001
C-
	DIMENSION XYZ(3),TRPL(2)
	do j=1,3
	  if (abs(xyz(j)) .le. 0.0001) xyz(j) = 0.0
	  IF (ABS(ABS(XYZ(j))-1.0).LT.0.0001) xyz(j)=xyz(j)/abs(xyz(j))
	end do
	IF (ABS(XYZ(3)) .eq. 1.0) THEN 
C
C	plunge is 90 degrees
C
	  if (xyz(3) .lt. 0.0) then
	    trpl(1) = PI
	  else
	    TRPL(1) = 0.0
	  end if
	  TRPL(2) = 0.5*PI
	  RETURN
	END IF
	IF (ABS(XYZ(1)) .LT. 0.0001) THEN
	  IF (XYZ(2) .GT. 0.0) THEN
	    TRPL(1) = PI/2.
	  ELSE IF (XYZ(2) .LT. 0.0) THEN
	    TRPL(1) = 3.0*PI/2.0
	  ELSE
	    TRPL(1) = 0.0
	  END IF
	ELSE
	  TRPL(1) = ATAN2(XYZ(2),XYZ(1))
	END IF
	C = COS(TRPL(1))
	S = SIN(TRPL(1))
	IF (ABS(C) .GE. 0.1) TRPL(2) = ATAN2(XYZ(3),XYZ(1)/C)
	IF (ABS(C) .LT. 0.1) TRPL(2) = ATAN2(XYZ(3),XYZ(2)/S)
	IF (TRPL(2) .LT. 0.0) THEN
	  TRPL(2) = -TRPL(2)
	  TRPL(1) = TRPL(1) - PI
	  END IF
	IF (TRPL(1) .LT. 0.0) TRPL(1) = TRPL(1) + 2.0*PI
	RETURN
	END
