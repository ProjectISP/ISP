        SUBROUTINE LRATIO(JR,DIP,STRIKE,RAKE,XYZ,XYZden,VPVS3,LOGRAT,
     1       TOP,CUTP,CUTS,FLAG)
c
c  calculates theoretical value of log10 amplitude ratio for a given
c      fault dip, strike, and slip (assuming a double-couple sorce)
C      SV/P for JR = 1, SH/P for JR = 2, SV/SH for JR = 3
C     Uses A&R convention for RAKE - negative of Kisslinger's
C
C  TOP contains the sign of the S polarity: + for SV if towards
C    the station (down on vertical), + for SH if to
C    left facing station (opposite from others, sorry).  For SV/SH
C    the sign will be for SV.
C  Written by Arthur Snoke Virginia Tech June 1984
C  10 June 1986:  Fixed case for 0/0 by taking derivatives of
C    both numerator and denominator with respect to strike
C    (derivatives with respect to dip did not work).
C  7 July 1990:  Decided that case for 0/0 was irrelevant.  If one
C    is near a nodal surface for either the numerator or denominator,
C    that should suffice in practise.  For FOCMEC therefore simply
C    want it flagged if near a nodal surface.
C  8 July 1990:  Intorduced CUT as a calling argument to limit the
C    range of ratios considered to stay away from nodal surfaces.
C    If both the numerator and denominator are less than CUT,
C    LOGRAT returned as -3, if only num., as -2, den. +2.
C  29 August 1991:  adapted from LSPRAT to include SV/SH
C  3 August 1993:  Changed procedure for dealing with near-nodal
C    cases:  Now separate CUTP and CUTS for P and S terms.  No more
C    FACTOR.  Returns a FLAG for near-nodal -- 'NUM' if numerator,
C    DEN if denominator, 'N&D' if both, '   ' if non-nodal.
C  6 January 1997: Corrected an error in solutions near a
C    nodal surface for either the numerator or the denominator
C    but not both.  Previously the ratio returned was not updated
C    for such cases, so was what had been found in the previous call.
C  27 February 1997: If numerator is near nodal surface, now returns
C    correct sign in TOP
C  December 2013:  BUG FIX: Now use correct TOA for denominator,
C    Previously, it used same TOA as for numerator
C-
      character*3 flag
      REAL*4 LOGRAT,XYZ(9),XYZden(6),A(3),N(3)
      A(1) = COS(RAKE)*COS(STRIKE) + SIN(RAKE)*COS(DIP)*SIN(STRIKE)
      A(2) = COS(RAKE)*SIN(STRIKE) - SIN(RAKE)*COS(DIP)*COS(STRIKE)
      A(3) = -SIN(RAKE)*SIN(DIP)
      N(1) = -SIN(STRIKE)*SIN(DIP)
      N(2) = COS(STRIKE)*SIN(DIP)
      N(3) = -COS(DIP)
      RAn = 0.0
      RNn = 0.0
      TAn = 0.0
      TNn = 0.0
      RAd = 0.0
      RNd = 0.0
      TAd = 0.0
      TNd = 0.0
      PA = 0.0
      PN = 0.0
      DO J=1,3
        RAn = RAn + XYZ(J)*A(J)
        RNn = RNn + XYZ(J)*N(J)
        TAn = TAn + XYZ(J+3)*A(J)
        TNn = TNn + XYZ(J+3)*N(J)
        RAd = RAd + XYZden(J)*A(J)
        RNd = RNd + XYZden(J)*N(J)
        TAd = TAd + XYZden(J+3)*A(J)
        TNd = TNd + XYZden(J+3)*N(J)
        PA = PA + XYZ(J+6)*A(J)
        PN = PN + XYZ(J+6)*N(J)
      END DO
      if (JR .lt. 3) then
        BOT = 2*RAd*RNd
      else
        BOT = RAd*PN + RNd*PA
      end if
      if (JR .eq. 2) then
        TOP = RAn*PN + RNn*PA
      else
        TOP = RAn*TNn + RNn*TAn
      end if
      flag = '   '
      IF (JR.ne.3.and.ABS(BOT).LE.CUTP .or. JR.eq.3.and.ABS(BOT)
     1      .le.CUTS) THEN
        IF (ABS(TOP) .LE. CUTS) THEN
c   ratio solution is not acceptable
          flag = 'N&D'
          lograt = 0.0
          RETURN
        ELSE
          flag = 'DEN'
          if (jr .ne. 3) then
            bot = CUTP
          else
            bot = CUTS
          end if
        END IF
      ELSE IF (ABS(TOP) .LE. CUTS) THEN
        flag = 'NUM'
        top = CUTS*sign(1.0,top)
      end if
c
c   BOT and TOP are normalized to unity.  Assuming a double-couple
c   source, ovserved S arrival are enhanced by a factor of VPVS3
c   compared to P arrivals.  VPVS3 = (VP/VS)**3 calculated at the source
c
      IF (JR .NE. 3) THEN
        LOGRAT = ALOG10(VPVS3*ABS(TOP/BOT))
      ELSE
        LOGRAT = ALOG10(ABS(TOP/BOT))
      end if
      RETURN
      END
