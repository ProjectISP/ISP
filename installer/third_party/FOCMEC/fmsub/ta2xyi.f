C+
	SUBROUTINE TA2XYI(RADIUS,UPPER)
C
C	Initializing part of TA2XY, which converts TOA (takeoff
C	  angle in radians) and AZimuth to X and Y for an equal
C	  area projection.  RADIUS is the radius of the circle and
C	  UPPER is .TRUE. if it is an upper hemisphere projection.
C
C	Arthur Snoke   VTSO   3 October 1985
C-
	LOGICAL UPPER
	REAL FACTOR
	SAVE FACTOR
	FACTOR = RADIUS*SQRT(2.0)
	IF (UPPER) FACTOR = -FACTOR
	RETURN
C
	ENTRY TA2XY(TOA,AZ,X,Y)
	F = FACTOR*SIN(TOA/2)
	X = F*SIN(AZ)
	Y = F*COS(AZ)
	RETURN
	END
