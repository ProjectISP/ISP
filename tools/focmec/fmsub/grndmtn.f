C+
	Subroutine grndmtn(RDDEG,IPS,VRAT,aemrg,AR,AV,PhR,PhV)
C
C  Version of gdmot using equations from Hudson (or Aki & Richards)
C  aemrg IS EMERGENCE ANGLE IN DEGREES RELATIVE TO UPWARD VERTICAL
C  VRAT=VS/VP AT THE SURFACE
C  AV AND AR ARE THE VERTICAL AND RADIAL GROUND DISPLACEMENT AMPLITUDES
C  PhR and PhV are the phases (in degrees).
C  A and B are the amplitudes for the P and S parts of the reflected
C	waves.  Beyond the critical angle, the phases Ap and Bp are nonzero
C	as A and B become complex.  Below the critical angle A and B
C	can be negative.  In this routine, A and B are not returned.
C	Opposite sign convention for incident SV from Hudson.  My convention
C	is positive incident SV is towards the surface (up and back),
C	his is away from the interface (down and forward).  We have the
C	same convention for reflected SV (up and forward).  Note that in
C	my convention, SV has the same direction relative to the ray
C	propagation for incident and reflected.  Not so for theirs.
C     
C	jas/vt July 2002
C-
	real mb,mbsts
	mb(x) = 1 - 2.0*x*x
	Ap = 0.0
	Bp = 0.0
	PhR = 0.0
	PhV = 0.0
      IF (aemrg .EQ. 0.0) THEN
        AR=2.*(IPS-1)
	PhR = (IPS-1)*180.0
	AV = 2*(2 - IPS)
	A = ips - 2
	B = ips - 1
      else if (aemrg .eq. 90.0) then
	ar = 0.0
	av = 0.0
	A = ips - 2
	B = ips - 1
	Bp = (ips-1)*180.0/RDDEG
      else if (aemrg.eq.45.0 .and. ips.eq.2) then
        A = 0.0
	B = 1.0
	PhR = -90.0
	Ap = 90.0
	av = sqrt(2.0)
	ar = 0.0
      ELSE
        IF (IPS .EQ. 1) THEN
          tp = aemrg/RDDEG
          ts=ASIN(SIN(tp)*VRAT)
	  sts = sin(ts)
	  mbsts = mb(sts)
	  ctp = cos(tp)
	  stp = sin(tp)
	  cts = cos(ts)
	  Anum1 = 4.0*sts*sts*ctp*cts*vrat
	  den = Anum1 + mbsts**2
	  A = (Anum1 - mbsts**2)/den
	  B = 4*sts*ctp*mbsts/den
	  AR = stp*(1.0 + A) + cts*B
	  if (AR .lt. 0.00001) AR = 0.0
	  AV = ctp*(1.0 - A) +sts*B
	  if (AV .lt. 0.00002) AV = 0.0
	else if (ips .eq. 2) then
          ts=aemrg/RDDEG
	  sts = sin(ts)
	  mbsts = mb(sts)
	  d2 = mbsts**2
	  cts = cos(ts)
	  Anum1 = 4.0*sts*cts*vrat
	  Anum = -Anum1*mbsts
          stp = sts/VRAT
          IF(stp .LE. 1.0) THEN
            tp = ASIN(stp)
	    ctp = cos(tp)
	    d1 = sts*ctp*Anum1
	    den = d1 + d2
	    A = Anum/den
	    B = (d1-d2)/den
	    AR = (2.*cts*d2 - stp*Anum)/den
	    PhR = 180.0
	    AV = (2.0*d1*sts - ctp*Anum)/den
	    if (abs(AV) .lt. 0.0001) AV = 0.0
	  else
	    ctp = sqrt(stp*stp-1.0)   ! this is pure imaginary
	    d1 = sts*ctp*Anum1
	    den = sqrt(d1*d1 + d2*d2)
	    B = 1.0
	    A = Anum/den
	    if (abs(A) .lt. 0.0001) A = 0.0
	    if (A .lt. 0.0) then
	      A = -A
	      Ap = rddeg*atan2(-d1,d2)
	    else
	      Ap = rddeg*atan2(d1,-d2)
	    end if
	    Bp = rddeg*atan2(2.0*d1*d2,-d2*d2+d1*d1)
	    AR = (-2.0*d2*cts + Anum*stp)/den 
	    if (abs(AR) .lt. 0.0001) AR = 0.0
	    if (AR .lt. 0.0) then
	      AR = -AR
	      PhR = rddeg*atan2(d1,-d2)
	    else
	      PhR = rddeg*atan2(-d1,d2)
	    end if
	    AV = (2.0*d1*sts - Anum*ctp)/den
	    if (AV .lt. 0.0) then
	      AV = -AV
	      PhV = rddeg*atan2(d1,-d2) + 90.
	    else
	      PhV = rddeg*atan2(-d1,d2) + 90.
	    end if
          END IF
        END IF
      END IF
      RETURN
      END
