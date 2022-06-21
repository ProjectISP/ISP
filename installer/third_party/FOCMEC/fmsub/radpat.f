C+
	SUBROUTINE RADPAT(D,ST,SL,AZ,APS,PR,SVR,SHR)
C
C     Input the dip, strike and slip - in the Aki & Richards convention
C        plus the azimuth and the P & S takeoff angles from the focus.
C        APS(1) P toa, APS(2) S TOA (relative to down)
c        All angles are in radians
C     Output radiation-pattern terms for P, SV, SH normalized to unity
C-
	REAL N(3)
	DIMENSION A(3),APS(2),R(2,3),TH(2,3),PH(3)
	A(1)=COS(SL)*COS(ST)+SIN(SL)*COS(D)*SIN(ST)
	A(2)=COS(SL)*SIN(ST)-SIN(SL)*COS(D)*COS(ST)
	A(3)=-SIN(SL)*SIN(D)
	N(1)=-SIN(ST)*SIN(D)
	N(2)=COS(ST)*SIN(D)
	N(3)=-COS(D)
	DO I=1,2
	  AT=APS(I)
	  R(I,1)=SIN(AT)*COS(AZ)
	  R(I,2)=SIN(AT)*SIN(AZ)
	  R(I,3)=COS(AT)
C  THETA AND PHI ARE REVERSED FROM THE USUAL CONVENTION
	  TH(I,1)=-COS(AT)*COS(AZ)
	  TH(I,2)=-COS(AT)*SIN(AZ)
C   Positive z is down
	  TH(I,3)=SIN(AT)
        ENDDO
	PH(1)=SIN(AZ)
	PH(2)=-COS(AZ)
        PH(3) = 0.0
	RPA=0.0
	RPN=0.0
	RSA=0.0
	RSN=0.0
	PHA=0.0
	PHN=0.0
	THA=0.0
	THN=0.0
	DO J=1,3
	  RPA=RPA+R(1,J)*A(J)
	  RPN=RPN+R(1,J)*N(J)
	  RSA=RSA+R(2,J)*A(J)
	  RSN=RSN+R(2,J)*N(J)
	  PHA=PHA+PH(J)*A(J)
	  PHN=PHN+PH(J)*N(J)
	  THA=THA+TH(2,J)*A(J)
	  THN=THN+TH(2,J)*N(J)
	END DO
C  P IS + ALONG RAY, SV + DOWN AND SH TO THE LEFT LOOKING FROM EQ
	PR = 2.*RPA*RPN
	SVR = RSN*THA+RSA*THN
	SHR = RSN*PHA+RSA*PHN
	RETURN
	END
