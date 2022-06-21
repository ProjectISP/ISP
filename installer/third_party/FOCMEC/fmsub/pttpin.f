C+
	SUBROUTINE PTTPIN (PTTP,ANGS,ANGS2,ANBTP,MOMTEN,PI)
C
C	Calculates other representations of fault planes with
C		trend and plunge of P and T as input.  All
C		angles are in radians.
C	22 July 1985:  Added moment tensor output
C-
	REAL N(3),MOMTEN(6)
	DIMENSION PTTP(4),ANGS(3),ANGS2(3),ANBTP(6),P(3),T(3),A(3),B(3)
	DATA SR2/0.707107/
	CALL TRPL2V(PTTP(1),P)
	CALL TRPL2V(PTTP(3),T)
	DO J=1,3
	  A(J) = SR2*(P(J) + T(J))
	  N(J) = SR2*(T(J) - P(J))
	enddo
        CALL AN2DSR(A,N,ANGS,PI)
        call dsrin(ANGS,ANBTP,ANGS2,PTTP,MOMTEN,PI)
	RETURN
	END
