C+
	SUBROUTINE AN2MOM(A,N,MOMTEN)
C
C	Starting with the A and N axis, calculates the elements
C	  of the moment tensor with unit scalar moment.
C	  Coordinate system:  X = North, Y = East, Z = Down
C	  Convention used is that of Dziewonski & Woodhouse
C	  (JGR 88, 3247-3271, 1983) and Aki & Richards (p 118)
C	24 September 1985: If an element is < 0.000001 (ABS), set to zero
C-
	REAL*4 A(3), N(3), MOMTEN(6)
C	      Moment tensor components:  M(I,j) = A(I)*N(J)+A(J)*N(I)
	MOMTEN(1) = 2.0*A(3)*N(3)	!  MRR = M(3,3)
	MOMTEN(2) = 2.0*A(1)*N(1)	!  MTT = M(1,1)
	MOMTEN(3) = 2.0*A(2)*N(2)	!  MPP = M(2,2)
	MOMTEN(4) = A(1)*N(3)+A(3)*N(1)	!  MRT = M(1,3)
	MOMTEN(5) = -A(2)*N(3)-A(3)*N(2)!  MRP = -M(2,3)
	MOMTEN(6) = -A(2)*N(1)-A(1)*N(2)!  MTP = -M(2,1)
	DO 100 J=1,6
	  IF (ABS(MOMTEN(J)) .LT. 0.000001) MOMTEN(J) = 0.0
100	CONTINUE
	RETURN
	END
