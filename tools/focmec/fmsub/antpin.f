C+
	SUBROUTINE ANTPIN (ANBTP,ANGS,ANGS2,PTTP,MOMTEN,PI)
C
C	Calculates other representations of fault planes with
C		trend and plunge of A and N as input.  All
C		angles are in radians.
C	22 July 1985:  Added moment tensor output.
C       June 2014:  Because of projections of vectors into the lower
C           hemisphere, can get wrong sign of rake unless make N(3)<0
C-
	REAL N(3), MOMTEN(6)
	DIMENSION PTTP(4),ANGS(3),ANGS2(3),ANBTP(6),P(3),T(3),A(3),B(3)
	DATA SR2/0.707107/
	CALL TRPL2V(ANBTP(1),A)
	CALL TRPL2V(ANBTP(3),N)
        if (N(3) .gt. 0) then
          do j=1,3
            N(j) = -N(j)
          enddo
        endif
        CALL AN2DSR(A,N,ANGS,PI)
        call dsrin(ANGS,ANBTP,ANGS2,PTTP,MOMTEN,PI)
	RETURN
	END
