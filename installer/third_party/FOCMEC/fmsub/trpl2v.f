C+
C	SUBROUTINE TRPL2V(TRPL,XYZ)
C
C	Transforms to XYZ components of a unit vector from
C		the trend and plunge for the vector.
C	Trend is the azimuth (clockwise from north looking down)
C	Plunge is the downward dip measured from the horizontal.
C	All angles in radians
C	X is north, Y is east, Z is down
C-
	SUBROUTINE TRPL2V(TRPL,XYZ)
	DIMENSION XYZ(3),TRPL(2)
	XYZ(1) = COS(TRPL(1))*COS(TRPL(2))
	XYZ(2) = SIN(TRPL(1))*COS(TRPL(2))
	XYZ(3) = SIN(TRPL(2))
	do j=1,3
	  if (abs(xyz(j)) .lt. 0.0001) xyz(j) = 0.0
	  if (abs(abs(xyz(j))-1.0).lt.0.0001) xyz(j)=xyz(j)/abs(xyz(j))
	end do
	RETURN
	END
