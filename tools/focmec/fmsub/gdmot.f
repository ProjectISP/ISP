C+
      SUBROUTINE GDMOT(RD,IPS,VSVP,AIN,AR,PHR,AV)
C
C  GDMOT GIVES GROUND DISPLACEMENT FOR INCIDENT P (IPS=1) OR S (IPS=2)
C  BULLEN (1963) P. 129 FOR INCIDENT P (BUT AIN=90-E)
C  AIN IS EMERGENCE ANGLE IN DEGREES RELATIVE TO VERTICAL
C  VSVP=VS/VP AT THE SURFACE
C  AV AND AR ARE THE VERTICAL AND RADIAL GROUND DISPLACEMENTS
C  FOR SV INC. BEYOND CRITICAL, PHV=PHR+90.
C     (AR MAY BE NEGATIVE IN THIS CASE - NOT REALLY AN AMPLITUDE)
C-
      COTAN(B) = 1.0/TAN(B)
      COTCOT(B)=1.-COTAN(B)**2
      PHR=0.0
      IF (AIN .EQ. 0.0) THEN
        AR=2.*(IPS-1)
	AV = 2*(2 - IPS)
      ELSE
        IF (IPS .EQ. 1) THEN
          A = AIN/RD
          B=ASIN(SIN(A)*VSVP)
          DEN=2.*COS(A)/(SIN(B)**2*(4.*COTAN(A)*COTAN(B)
     *      +COTCOT(B)**2))
          AR=2.*COTAN(B)*DEN
          AV=-COTCOT(B)*DEN
        ELSE IF (IPS .EQ. 2) then
          B=AIN/RD
          SA=SIN(B)/VSVP
          IF(SA .LE. 1.0) THEN
            A = ASIN(SA)
            DEN=2.*COTAN(B)/(SIN(B)*(4.*COTAN(A)*COTAN(B)
     *        +COTCOT(B)**2))
            AV = -2*COTAN(A)*DEN
            AR=-COTCOT(B)*DEN
          ELSE
            CTA=SQRT(1.-1./(SA*SA))
            DEN=2.*COTAN(B)/(SIN(B)*SQRT((4.*CTA*COTAN(B))**2
     *        +COTCOT(B)**4))
            AV=-2.*CTA*DEN
            AR=0.
            PHR = 90.0
            IF(COTCOT(B).EQ.0.0) RETURN
            AR=-COTCOT(B)*DEN
            PHR=ATAN2(4.*COTAN(B)*CTA,-COTCOT(B)**2)*RD
          END IF
        END IF
      END IF
      RETURN
      END
