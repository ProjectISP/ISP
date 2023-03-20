C+
	SUBROUTINE DSRIN (ANGS,ANBTP,ANGS2,PTTP,MOMTEN,PI)
C
C	Calculates other representations of fault planes with
C		dip, strike and rake (A&R/RBH convinention) input.
C		All angles are in radians.
C-
	REAL N(3), MOMTEN(6)
	DIMENSION PTTP(4),ANGS(3),ANGS2(3),ANBTP(6),P(3),T(3),A(3),B(3)
	DATA SR2/0.707107/
	RAKE=ANGS(3)
	STR = ANGS(2)
	DIP = ANGS(1)
	A(1) = COS(RAKE)*COS(STR) + SIN(RAKE)*COS(DIP)*SIN(STR)
	A(2) = COS(RAKE)*SIN(STR) - SIN(RAKE)*COS(DIP)*COS(STR)
	A(3) = -SIN(RAKE)*SIN(DIP)
	N(1) = -SIN(STR)*SIN(DIP)
	N(2) = COS(STR)*SIN(DIP)
	N(3) = -COS(DIP)
	B(1) = COS(STR)*SIN(RAKE) - COS(RAKE)*COS(DIP)*SIN(STR)
	B(2) = COS(RAKE)*COS(STR)*COS(DIP) + SIN(RAKE)*SIN(STR)
	B(3) = COS(RAKE)*SIN(DIP)
C
	do j=1,3
          if (abs(A(j)) .le. 0.0001) A(j) = 0.0
          IF ((ABS(A(j))-1.0) .gt. 0.0) A(j)=A(j)/abs(A(j))
          if (abs(N(j)) .le. 0.0001) N(j) = 0.0
          IF ((ABS(N(j))-1.0) .gt. 0.0) N(j)=N(j)/abs(N(j))
          if (abs(B(j)) .le. 0.0001) B(j) = 0.0
          IF ((ABS(B(j))-1.0) .gt. 0.0) B(j)=B(j)/abs(B(j))
        end do
	CALL V2TRPL(A,ANBTP(1),PI)
	CALL V2TRPL(N,ANBTP(3),PI)
	CALL V2TRPL(B,ANBTP(5),PI)
	DO J=1,3
            T(J) = SR2*(A(J) + N(J))
            P(J) = SR2*(A(J) - N(J))
        enddo 
	CALL V2TRPL(P,PTTP(1),PI)
	CALL V2TRPL(T,PTTP(3),PI)
	CALL AN2DSR(N,A,ANGS2,PI)
	CALL AN2MOM(A,N,MOMTEN)
	RETURN
	END
