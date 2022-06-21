C+
	LOGICAL FUNCTION SVNODE(MT, INCRMT, TOA, AZ)
C	Called by SVNSRF.  Calculates next pont on nodal surface for SV
C	Bruce Julian's (USGS) routine.  Perturbed by Arthur Snoke
C	  October 1985
C-
C  moment tensor - not Julian's convention (D & W)
	REAL MT(6)
C  in azimuth
	REAL INCRMT
C  take-off angle, azimuth
	REAL TOA, AZ
	REAL DC0Z, DC2Z, DS2Z
	SAVE DC0Z, DC2Z, DS2Z
	REAL MRT, MRP
	SAVE MRT, MRP
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
	LOGICAL SVNNXT
C  initialize, return first point
	PI = 4.0*ATAN(1.0)
	DC0Z = 0.5*(MT(1) - 0.5*(MT(2) + MT(3)))
	DC2Z = 0.25*(MT(3) - MT(2))
	DS2Z = 0.5*MT(6)
	MRT = MT(4)
	MRP = MT(5)
	IF ((MRT .EQ. 0.0) .AND. (MRP .EQ. 0.0) .AND. (DC0Z .EQ. 0.0
     *) .AND. (DC2Z .EQ. 0.0) .AND. (DS2Z .EQ. 0.0)) THEN
C  no sv waves
	  SVNODE = (.FALSE.)
	  RETURN
	END IF
	Z1 = ATAN2(MRT, MRP) + 4.0*5.96046448E - 08*PI
	IF ((-MRT*SIN(Z1)-MRP*COS(Z1) .LT. 0.0)) Z1 = Z1 + PI
C  so toa>0 for z1<az<z2
	Z2 = Z1 + PI - 0.00001
	DZ = INCRMT
	Z = Z1
	J = -1
C  fall through shnnxt entry point
C  calculate next point on nodal curve
	ENTRY SVNNXT(TOA, AZ)
	IF (Z .GE. Z2) THEN 
C  done
	  SVNODE = (.FALSE.)
	  RETURN
	END IF
	J = J + 1
	Z = MIN(Z1 + J*DZ, Z2)
	AZ = Z
	SZ = SIN(Z)
	CZ = COS(Z)
	S2Z = 2.0*SZ*CZ
	C2Z = 2.0*CZ**2 - 1.0
	NUM = MRT*CZ - MRP*SZ
	DEN = DC0Z + DC2Z*C2Z + DS2Z*S2Z
	IF ((NUM .EQ. 0.0) .AND. (DEN .EQ. 0.0)) THEN
C  degenerate:  use l'hospital's rule
	  NUM =  - MRT*SZ - MRP*CZ
	  DEN = 2.0*(DS2Z*C2Z - DC2Z*S2Z)
	END IF
	TOA = 0.5*ATAN2(NUM, DEN)
	SVNODE = (.TRUE.)
	RETURN
	END
