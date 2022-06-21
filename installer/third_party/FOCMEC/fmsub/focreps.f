C+
      SUBROUTINE focreps(A,N,angle,ANGS,TREND,PLUNGE,PI,lunit1)
C
C      Input: A and N vectors, and dip/strike/slip 
C           plus B trned and plunge in radians 
C       Prints angles for two planes plus trnd and plunge
C           for A, N, B, P, T
C       Replaces fmreps.f within Focmec (June 2014)
C-
      real*4 btrpl(2),ANGS(3),ANGS2(3),antp(4)
      real*4 t(3),p(3),pttp(4),a(3),n(3)
      rd = 180.0/pi
        sr2 = sqrt(2.0)
        call v2trpl(a,antp(1),pi)
        call v2trpl(n,antp(3),pi)
      DO J=1,3
        T(J) = SR2*(A(J) + N(J))
        P(J) = SR2*(A(J) - N(J))
        enddo
        call v2trpl(p,pttp(1),pi)
        call v2trpl(t,pttp(3),pi)
      CALL AN2DSR(N,A,ANGS2,PI)
      DO I=1,3
        ANGS(I) = ANGS(I)*rd
        ANGS2(I) = ANGS2(I)*rd
        enddo
        do j=1,4
          antp(j) = rd*antp(j)
        PTTP(j) = PTTP(j)*rd
        enddo
      IF (LUNIT1 .GT. 0) THEN
        WRITE (LUNIT1,1) (ANGS(I),I=1,3)
        WRITE(LUNIT1,2)(ANGS2(I),I=1,3),'   Auxiliary Plane'
        WRITE (LUNIT1,3) ANTP
        WRITE(LUNIT1,5) PTTP
          write(lunit1,7) trend,plunge,angle
      END IF
      RETURN
C
1      FORMAT(5X,'Dip,Strike,Rake ',3F9.2)
2      FORMAT(5X,'Dip,Strike,Rake ',3F9.2,A)
3      FORMAT(5X,'Lower Hem. Trend, Plunge of A,N ',4F9.2)
5      FORMAT(5X,'Lower Hem. Trend, Plunge of P,T ',4F9.2)
7      FORMAT(5x,'B trend, B plunge, Angle: ',3F7.2)
      END
