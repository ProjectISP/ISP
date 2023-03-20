C+
C     SUBROUTINE GMPRD(A,B,R,N,M,L)
C
C     MULTIPLIES A N BY M MATRIX A TIMES A M BY L MATRIX B GIVING A N
C         BY L MATRIX R
C-
      SUBROUTINE GMPRD(A,B,R,N,M,L)
      DIMENSION A(1),B(1),R(1)
      IR=0
      IK=-M
      DO 10 K=1,L
      IK=IK+M
      DO 10 J=1,N
      IR=IR+1
      JI=J-N
      IB=IK
      R(IR)=0
      DO 10 I=1,M
      JI=JI+N
      IB=IB+1
   10 R(IR)=R(IR)+A(JI)*B(IB)
      RETURN
      END
