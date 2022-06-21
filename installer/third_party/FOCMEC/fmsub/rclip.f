C+
	LOGICAL FUNCTION RCLIP(R1, T1, R2, T2, RMAX)
C rclip - remove portion of line segment outside of circle
C	RCLIP is .TRUE. if (at least) one of the two points is
C	  in the circle.  If one point is outside, it is changed 
C	  to be on the border.
C-
C Polar coordinates of ends of segment
	REAL R1, R2, T1, T2
C Radius of circle
	REAL RMAX
	LOGICAL XCHNG
	REAL CT1, CT2, ST1, ST2
	REAL R0, T0
	REAL TEMP
	XCHNG = R1 .GT. R2
	IF (XCHNG) THEN
	  TEMP = R1
	  R1 = R2
	  R2 = TEMP
	  TEMP = T1
	  T1 = T2
	  T2 = TEMP
	END IF
	RCLIP = R1 .LT. RMAX
	IF (RCLIP .AND. (R2 .GT. RMAX)) THEN
	  CT1 = COS(T1)
	  ST1 = SIN(T1)
	  CT2 = COS(T2)
	  ST2 = SIN(T2)
	  T0 = ATAN((R2*CT2 - R1*CT1)/(R1*ST1 - R2*ST2))
	  R0 = R1*COS(T1 - T0)
	  IF (R0 .LT. 0.0) THEN
	    R0 =  - R0
	    T0 = T0 + 3.14159265
	  END IF
	  T2 = T0 + SIGN(ACOS(R0/RMAX), SIN(T2 - T0))
	  R2 = RMAX
	END IF
	IF (XCHNG) THEN
	  TEMP = R1
	  R1 = R2
	  R2 = TEMP
	  TEMP = T1
	  T1 = T2
	  T2 = TEMP
	END IF
	RETURN
	END
