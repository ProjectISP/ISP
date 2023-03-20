C+
	LOGICAL FUNCTION SHNODE(MT, INCRMT, TOA, AZ)
C	Called by SHNSRF.  Calculates next pont on nodal surface for SH
C	Bruce Julian's (USGS) routine.  Perturbed by Arthur Snoke
C	  October 1985
C-
C  moment tensor - not Julian's convention (D & W)
	REAL MT(6)
C  in azimuth
	REAL INCRMT
C  take-off angle, azimuth
	REAL TOA, AZ
	REAL MRT, MRP, MTP
	SAVE MRT, MRP, MTP
	REAL MTTPP2
	SAVE MTTPP2
C  starting, final azimuth
	REAL Z1, Z2
	SAVE Z1, Z2
C  current azimuth, incrmt
	REAL Z, DZ
	SAVE Z, DZ
	INTEGER J
	SAVE J
C  numerator, denominator
	REAL NUM, DEN
C  sin, cos of z, 2*z
	REAL CZ, SZ, C2Z, S2Z
	SAVE PI
C entry point
	LOGICAL SHNNXT
C  initialize, return first point
	MRT = MT(4)		! Julian: MT(2)
	MRP = MT(5)		! Julian: MT(4)
	MTP = MT(6)		! Julian: MT(5)
	MTTPP2 = (MT(2) - MT(3))/2.0	! Julian: 3-6
	PI = 4.0*ATAN(1.0)
	IF ((MRT .EQ. 0.0) .AND. (MRP .EQ. 0.0) .AND. (MTP .EQ. 0.0) 
     *   .AND. (MTTPP2 .EQ. 0.0)) THEN
C  no sh waves
	  SHNODE = (.FALSE.)
	  RETURN
	END IF
	Z1 = ATAN2( - MRP, MRT) + 4.0*5.96046448E - 08*PI
	IF ((-MRT*COS(Z1)+MRP*SIN(Z1) .LT. 0.0)) Z1 = Z1 + PI
C  so toa>0 for z1<az<z2
	Z2 = Z1 + PI - 0.00001
	DZ = INCRMT
	Z = Z1
	J = -1
C  fall through shnnxt entry point
C  calculate next point on nodal curve
	ENTRY SHNNXT(TOA, AZ)
	IF (Z .GE. Z2) THEN 
C  done
	  SHNODE = (.FALSE.)
	  RETURN
	END IF
	J = J + 1
	Z = MIN(Z1 + J*DZ, Z2)
	AZ = Z
	SZ = SIN(Z)
	CZ = COS(Z)
	S2Z = 2.0*SZ*CZ
	C2Z = 2.0*CZ**2 - 1.0
	NUM =  - MRT*SZ - MRP*CZ
	DEN = MTP*C2Z + MTTPP2*S2Z
	IF ((NUM .EQ. 0.0) .AND. (DEN .EQ. 0.0)) THEN
C  degenerate:  use l'hospital's rule
	  NUM = MRP*SZ - MRT*CZ
	  DEN = 2.0*(MTTPP2*C2Z - MTP*S2Z)
	END IF
	TOA = ABS(ATAN2(NUM, DEN))
	SHNODE = (.TRUE.)
	RETURN
	END
