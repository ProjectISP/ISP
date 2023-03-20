C+
      LOGICAL FUNCTION OKPOL(A,N,NBADP,NBADSV,NBADSH)
C
C      Called by OKSOL, which is called by SRCHFM, which is called by FOCMEC
C      Compares observed polarities with trial mechanism
C            designated by A and N
C      NBADP and NBADS return the number of bad P and S polarities
C        If any of the (weighted) numbers of errors is bigger than the
C        appropriate maximum allowed, the function is .FALSE. on return
C      23 July 1985:  Added SH polarity check 
C      10 August 1985:  Added SH + P polarity option
C      27 September 1985:  included variable weighting option for
C        polarities
C      last perturbed 12 October 1990
C      29 August 1991:  sun version.  Now can have SV polarities
c      September 2009 changed a comment line above.
C-
      INCLUDE 'FOCMEC.INC'
      REAL*4 A(3),N(3)
      INTEGER NBADP, NBADSH, NBADSV
      OKPOL = .FALSE.
      NBADP = 0
      NBADSV = 0
      NBADSH = 0
      BADP = 0.0
      BADSV = 0.0
      BADSH = 0
      BAD = 0.0
      DO K=1,NPOL
          KKK = KEYPOL(K)
          IF (KKK-1000 .LT. 0) THEN
            KK = KKK
            JR = 1
          ELSE IF (KKK-2000 .LT. 0) THEN
            KK = KKK - 1000
            JR = 2
            TA = 0.0
            TN = 0.0
            do j=1,3
              TA = TA + XYZ(J+3,KK)*A(J)
              TN = TN + XYZ(J+3,KK)*N(J)
            end do
          ELSE
            KK = KKK - 2000
            JR = 3
            PA = 0.0
            PN = 0.0
            do j=1,3
              PA = PA + XYZ(J+6,KK)*A(J)
              PN = PN + XYZ(J+6,KK)*N(J)
            end do
          END IF
          RA = 0.0
          RN = 0.0
          DO J=1,3
            RA = RA + XYZ(J,KK)*A(J)
            RN = RN + XYZ(J,KK)*N(J)
          END DO
        IF (JR .eq. 1) THEN    !  This is a P polarity
          TEST = POLRTY(K)*2*RA*RN
          IF (TEST .LE. 0.0) THEN
            BADNOW = AMAX1(-TEST,THRESH)
            BADP = BADP + BADNOW
            BAD = BAD + BADNOW
            IF (BAD .GT. ERR) RETURN
            IF (BADP .GT. ERRP) RETURN
            NBADP = NBADP + 1
            NBAD = NBAD + 1
            BADPP(NBADP) = PSTATN(K)
            WBADP(NBADP) = BADNOW
          ENDIF
        ELSE IF (JR .eq. 2) then   ! This one is an SV polarity
          TEST = POLRTY(K)*(RA*TN + RN*TA)
          IF (TEST .LE. 0.0) THEN
            BADNOW = AMAX1(-TEST,THRESH)
            BADSV = BADSV + BADNOW
            BAD = BAD + BADNOW
            IF (BAD .GT. ERR) RETURN
            IF (BADSV .GT. ERRSV) RETURN
            NBADSV = NBADSV + 1
            BADSVP(NBADSV) = PSTATN(K)
            WBADSV(NBADSV) = BADNOW
          END IF
        ELSE IF (JR .eq. 3) then   ! This one is an SH polarity
          TEST = POLRTY(K)*(RA*PN + RN*PA)
          IF (TEST .LE. 0.0) THEN
            BADNOW = AMAX1(-TEST,THRESH)
            BADSH = BADSH + BADNOW
            BAD = BAD + BADNOW
            IF (BAD .GT. ERR) RETURN
            IF (BADSH .GT. ERRSH) RETURN
            NBADSH = NBADSH + 1
            BADSHP(NBADSH) = PSTATN(K)
            WBADSH(NBADSH) = BADNOW
          END IF
        END IF
      END DO
      OKPOL = .TRUE.
      RETURN
      END
