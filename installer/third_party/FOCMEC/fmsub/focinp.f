C+
      SUBROUTINE FOCINP
C
C      Input routine for FOCMEC
C
C      Arthur Snoke  Virginia Tech  May 1984
C      Last perturbed 12 October 1990
C      20 August 1991:  sun version.  call assign replaced by open
C      31 August:  expanded to include SV polarities and SV/SH ratios
C      15 May 1992:  Changed name of listing file from lp.lst to 
C            focmec.lst
C      19 June 1993: Separate P and S errors should now work
C      20 Jun 1993:  Belts and braces:  If someone uses a ratio with
C        no input S take-off angle, now sets it equal to P angle
C      2 August 1993:  Changed rules for ratios near nodal surfaces
C      6 January 1997:  format change
C      7 April 2000: sense='R' no longer allowed for ratios.  First TOANG
C        is not necessarily for P (if an SH or SV polarity is to be used)
C      1 October 2001: When porting to a PC, found some compilers could
C            not handle reading in an integer for a floating-pint
C            variable (number of polarity errors), so "fixed" it.
C      26 March 2002: If emergent polarity data are included, now it
C            will ask if you want to include it.  Previously it 
C            ignored such data.
C      5 July 2002: Some compilers do not like the way I had used
C        CVALUE -- CVALUE(1:NFILE) with NFILE defined on the right-hand side.
c      19 November 2008: If INFO was left out for a ratio, did not read line.
C            Now it does.
C      June 2009: formatting changes
C       December 2013: Added some printed comments to explain terms
C          For searches, default maxes are calculated from increments
c          BUG FIX: Fixed take-off angles in ratios: den no longer same as num
C       June, 2014: Let Angle run from 0 to 180.  Easier to find Focmec input
C         from a focal mechanism.  Used to be A angle, but may be with N.
C      July 2016: Changed wording and order in ratio formats.  Now
C        the number of ratio errors is relative to the input total number
C        rather than that number minus stations for which N & D are both near
C        nodal surfaces.  Changed defaults for P and S cutoffs for ratios.
C        Now 0.1 and 0.1.  Now P/S focus velocity ratio read from file.
C     December 2016:  Changed threshold for polarity errors from 0.1 to 0.0 and
C       there is no prompt.  Now RW is the default. (Previously it was UW.)
C       Added emergent SV (f and b).  Cleaned up formatting for output.
C     January 2017: Changed default for polarity error to sekparate phases
C     September 2017: changed printed output values for ratios, so format change.
C-
      INCLUDE 'FOCMEC.INC'
      CHARACTER*1 SENSE,SNUMPOL
      CHARACTER*80 COMMNT,FILENA,CVALUE,DUMMY
      LOGICAL TRUTH, relweight, pguess
      CHARACTER*40 INFO
      RD = 45.0/ATAN(1.0)            
      open(2,file='focmec.lst',status='unknown')
      FILENA = 
     1    CVALUE('Output file name (for plotting) [focmec.out]',
     2    'focmec.out',NFILE)
      open(3,file=filena(1:nfile),status='unknown')
      CALL TIMDAT(2,'Focmec')
      CALL TIMDAT(3,'Focmec')
      COMMNT =
     1    CVALUE('Comment - up to 80 characters',DUMMY,NCOM)
      WRITE(2,'(1X,A)') COMMNT(1:NCOM)
      WRITE(3,'(1X,A)') COMMNT(1:NCOM)
100      FILENA = CVALUE('Input filespec',DUMMY,NFILE)
      OPEN(UNIT=1,FILE=FILENA(1:NFILE),STATUS='OLD',ERR=100)
      READ(1,'(A)') COMMNT
      ncom = lenc(commnt)
      WRITE(*,'(1X,A)') COMMNT(1:NCOM)
      IF (.NOT.TRUTH('Correct file?...[Y]')) GO TO 100
      WRITE(2,3) FILENA(1:NFILE)
      WRITE(3,3) FILENA(1:NFILE)
      WRITE(2,'(1X,A)') COMMNT(1:NCOM)
      WRITE(3,'(1X,A)') COMMNT(1:NCOM)
      WRITE(2,4)
      NRAT = 0
      NPOL = 0
      NPPOL = 0
      NSVPOL = 0
      NSHPOL = 0
      J = 0
      NGUESS = 0
200      read(1,'(a)',end=300) commnt
      ncom = lenc(commnt)
      if (ncom.lt.22 .or. commnt(24:29).eq.'      ') then
c   toang1 hers is the take-off angle for the P or S ray with the polarity
        read(commnt(1:21),'(A4,2F8.2,A1)') STA,AZIN,TOANG1,SENSE
        if (ncom .gt. 21) then
            read(commnt(40:ncom),'(a)') info
c            write(*,'(i3, '' '',a)') lenc(info),info(1:lenc(info))
        else
            info = ' '
        endif
      else
C   toang1 here is the take-off angle for the numerator ray
C   toang2 is the take-off angle for the denominator ray in a ratio
        READ(commnt,5) STA,AZIN,TOANG1,SENSE,
     .          RATLOG,SNUMPOL,TOANG2,VPVS
        info = ' '
        if (ncom .gt. 45) read(commnt(46:ncom),'(a)') info
      endif
      IF (SENSE.EQ.'V' .OR. SENSE.EQ.'S' .OR. SENSE.EQ.'H') THEN
          J = J + 1
          IF (J .GT. MAX2) GO TO 300
          NRAT = NRAT + 1
          LOGRAT(NRAT) = RATLOG
          SVSH(2,NRAT) = SNUMPOL
          RSTATN(NRAT) = STA
          NADD = 0
          SVSH(1,NRAT) = SENSE
          IF (SENSE .EQ. 'H') NADD = 1000
          IF (SENSE .EQ. 'S') NADD = 2000
          WRITE(2,6) RSTATN(NRAT),AZIN,TOANG1,SENSE,LOGRAT(NRAT),
     .            SVSH(1,NRAT),SVSH(2,NRAT),TOANG2,INFO(1:lenc(info))
          KEYRAT(NRAT) = J + NADD
      ELSE
          WRITE(2,10) STA,AZIN,TOANG1,SENSE,INFO(1:lenc(info))
          IF (SENSE .EQ. 'U') SENSE = 'C'
            if (sense .eq. 'R') sense = '>'
            if (sense .eq. 'L') sense = '<'
          if (sense.eq.'+' .or. sense.eq.'-' .or. sense.eq.'l'
     1              .or. sense.eq.'r') then
            if (nguess .eq. 0) then
              nguess = 1
              pguess = truth('Include emergent polarity picks?..[Y]')
              write(*,*) nguess,pguess
              if (pguess) then
                write(2,*) 'Including emergent polarity picks'
                write(3,*) 'Including emergent polarity picks'
              else
                write(2,*) 'Not including emergent polarity picks'
                write(3,*) 'Not including emergent polarity picks'
              end if
            end if
            if (pguess) then
              IF (SENSE .EQ. '+') SENSE = 'C'
              IF (SENSE .EQ. '-') SENSE = 'D'
              IF (SENSE .EQ. 'l') SENSE = '<'
              IF (SENSE .EQ. 'r') SENSE = '>'
              IF (SENSE .EQ. 'f') SENSE = 'F'
              IF (SENSE .EQ. 'b') SENSE = 'B'
            end if
        end if
        IF(.NOT.(SENSE.EQ.'C' .OR. SENSE.EQ.'D'
     .        .OR. SENSE .EQ. 'F' .OR. SENSE .EQ. 'B'
     .            .OR. SENSE .EQ. '<' .OR. SENSE .EQ. '>')) GO TO 200
            J = J + 1
            IF (J .GT. MAX2) GO TO 300
            IF (SENSE .EQ. '<' .OR. SENSE .EQ. '>') THEN
              NSHPOL = NSHPOL + 1
              NADD = 2000
            ELSE IF (SENSE .EQ. 'F' .OR. SENSE .EQ. 'B') THEN
              NSVPOL = NSVPOL + 1
              NADD = 1000
            ELSE
              NPPOL = NPPOL + 1
              NADD = 0
            END IF
            NPOL = NPOL + 1
            KEYPOL(NPOL) = J + NADD
            PSTATN(NPOL) = STA
            IF (SENSE .EQ. 'C' .OR. SENSE .EQ. '<'
     .          .OR. SENSE .EQ. 'F') THEN
                   POLRTY(NPOL) = 1
            ELSE
                   POLRTY(NPOL) = -1
            END IF
        ENDIF
        TREND = AZIN/RD
        toa = TOANG1/RD 
        COST = COS(TREND)
        SINT = SIN(TREND)
        sinP = sin(toa)
        cosP = cos(toa)
        XYZ(1,J) = COST*sinp
        XYZ(2,J) = SINT*sinP
c   Positive z is down
        XYZ(3,J) = cosP
C  Next two vectors reversed in sign from A&R convention because
C   of my convention for SV and SH (down and left, facing the station)
        XYZ(4,J) = -COST*cosP
        XYZ(5,J) = -SINT*cosP
        XYZ(6,J) = sinp 
        XYZ(7,J) = SINT
        XYZ(8,J) = -COST
        XYZ(9,J) = 0.0
        IF (SENSE.EQ.'V' .OR. SENSE.EQ.'S' .OR. SENSE.EQ.'H') THEN
            toa = toang2/rd
            sinP = sin(toa)
            cosP = cos(toa)
            XYZden(1,J) = COST*sinp
            XYZden(2,J) = SINT*sinP
            XYZden(3,J) = cosP
            XYZden(4,J) = -COST*cosP
            XYZden(5,J) = -SINT*cosP
            XYZden(6,J) = sinp
        endif
        GO TO 200
300     CLOSE(UNIT=1)
        WRITE(*,7) NPPOL,NSVPOL,NSHPOL,NRAT
        IF (NPOL .LE. 0) THEN
          WRITE(2,8)
          WRITE(3,8)
          GO TO 400
       ELSE
        write(*,*) ' Can have relative weighting for polarity errors'
        write(*,*) '  for which weight = theor. rad. factor (0 to 1)'
        IF (TRUTH('Relative weighting?..[Y]')) THEN
c2016          THRESH = RVALUE('Lower threshold [0.01]',0.01)
          thresh = 0.0
          relweight = .true.
        ELSE
          THRESH = 1.0
          relweight = .false.
        END IF
      END IF
      IF (NPPOL .GT. 0 .AND. (NSHPOL .GT. 0
     .      .OR. NSVPOL .GT. 0)) THEN
        WRITE(*,*) 'Options:  (1) Total polarity errors'
        WRITE(*,*) '          (2) Separate P, SV, and SH (default)'
        IF (TRUTH('Total polarity error option?..[N]')) THEN
          if (relweight) then
            ERR = VALUE('Total number of errors (floating point)')
            WRITE(3,17) NPPOL,NSVPOL,NSHPOL,ERR
            WRITE(2,17) NPPOL,NSVPOL,NSHPOL,ERR
          else
            NERR = VALUE('Total number of errors (integer)')
            err = nerr
            WRITE(3,18) NPPOL,NSVPOL,NSHPOL,NERR
            WRITE(2,18) NPPOL,NSVPOL,NSHPOL,NERR 
          end if
          ERRP = ERR
            ERRSV = ERR
          ERRSH = ERR
          GO TO 400
        END IF
      END IF
      IF (NPPOL .GT. 0) THEN
        if (relweight) then
          ERRP = RVALUE('Allowed P polarity errors..[0.0]',0.0)
        else
          NERRP = IVALUE('Allowed P polarity errors..[0]',0)
          errp = nerrp
        end if
        IF (ERRP .GT. float(NPPOL)) ERRP = NPPOL
      else
        errp = 0.0
      END IF
      IF (NSVPOL .GT. 0) THEN
        if (relweight) then
          ERRSV = RVALUE('Allowed SV polarity errors..[0.0]',0.0)
        else
          NERRSV = IVALUE('Allowed SV polarity errors..[0]',0)
          errSV = nerrsv
        end if
        IF (ERRSV .GT. float(NSVPOL)) ERRSV = NSVPOL
      else
        errsv = 0.0
      END IF
      IF (NSHPOL .GT. 0) THEN
        if (relweight) then
          ERRSH = RVALUE('Allowed SH polarity errors..[0.0]',0.0)
        else
          NERRSH = IVALUE('Allowed SH polarity errors..[0]',0)
          errsh = nerrsh
        end if
        IF (ERRSH .GT. float(NSHPOL)) ERRSH = NSHPOL
      else
        errsh = 0.0
      END IF
      err = errp + errsv + errsh
      if (thresh .lt. 1.0) then
        write(2,19) nppol,errp,nsvpol,errsv,nshpol,errsh
        write(3,19) nppol,errp,nsvpol,errsv,nshpol,errsh
      else
        nerrp = errp
        nerrsv = errsv
        nerrsh = errsh
        write(2,9) NPPOL,NERRP,NSVPOL,NERRSV,NSHPOL,NERRSH
        write(3,9) NPPOL,NERRP,NSVPOL,NERRSV,NSHPOL,NERRSH
      end if
400      IF (NRAT .LE. 0) THEN
        WRITE(2,12)
        WRITE(3,12)
      ELSE
        ERRRAT = RVALUE('Max permitted log10 |o-c| error..[0.6]',0.6)
        NERRR = IVALUE('Number permitted |o-c| errors..[0]',0)
        write(*,*) 'Next two entries are for near-nodal amplitudes'
        write(*,*) 'CUTP is the lower bound for P radiation factor'
        write(*,*) 'CUTS is the lower bound for S radiation factor'
        write(*,*) 'Ratio is indeterminate if both calculated'
        write(*,*) '    values less than the chosen CUT values'
        CUTP = RVALUE('CUTP: lower-limit P cutoff...[0.1]',0.1)
        CUTS = RVALUE('CUTS: lower-limit S cutoff...[0.1]',0.1)
        IF (NERRR .GT. NRAT) NERRR = NRAT
        WRITE(2,13) NRAT,NERRR,ERRRAT,VPVS
        WRITE(2,'(a,2(f5.3,a))') 'For ratios,  ', CUTP,
     1     ' = P radiation cutoff  ',CUTS,' = S radiation cutoff'
        WRITE(3,13) NRAT,NERRR,ERRRAT,VPVS
        WRITE(3,'(a,2(f5.3,a))') 'For ratios,  ', CUTP,
     1     ' = P radiation cutoff  ',CUTS,' = S radiation cutoff'
        VPVS3 = VPVS**3
      ENDIF
      if (NRAT .gt. 0) then
        write(*,*) 'FLAG is NUM, DEN, N&D if n, d, both below curoff'
        write(*,*) 'If FLAG N&D, total ratios used decreased by 1'
        write(*,*) 'R Ac/Tot: # acceptable / total ratios minus M&D'
        write(*,*) 'RMS Err RMS of acceptable obs ratios (no flag)'
        write(*,*) 'AbsMaxDiff: max abs difference for ok solutions'
        write(3,*) 'FLAG is NUM, DEN, N&D if n, d, both below curoff'
        write(3,*) 'If FLAG N&D, total ratios used decreased by 1'
        write(3,*) 'R Ac/Tot: # acceptable / total ratios minus M&D'
        write(3,*) 'RMS Err: RMS of acceptable obs ratios'
        write(3,*) 'AbsMaxDiff: max abs difference for ok solutions'
        write(2,*) 'FLAG is NUM, DEN, N&D if n, d, both below curoff'
        write(2,*) 'If FLAG N&D, total ratios used decreased by 1'
      endif
      MAXSOL = IVALUE('Exit after this many acceptable sols...[100]',
     .        100)
      BTMIN = RVALUE('Minimum search value B trend..[0]',0.0)
      BTDEL = ABS(RVALUE('Increment for B trend..[5 degree]',5.0))
      BTMAX = RVALUE('Maximum B trend..[360-btdel]',360-btdel)
        BTMAX = AMAX1(BTMIN,AMIN1(BTMAX,360-btdel))
      BPMIN = RVALUE('Minimum search value B plunge..[0]',0.0)
      BPDEL = ABS(RVALUE('Increment for B plunge..[5 degree]',5.0))
      BPMAX = RVALUE('Maximum B plunge..[90 degrees]',90.0)
        BPMAX = AMAX1(BPMIN,AMIN1(BPMAX,90.0))
      WRITE(*,*) '"Angle" in vertical plane of B trend)'
      AAMIN = RVALUE('Minimum search value Angle..[0]',0.0)
      AADEL = ABS(RVALUE('Increment for Angle..[5 degree]',5.0))
      AAMAX = RVALUE('Maximum Angle..[180-aadel]',180-aadel)
        AAMAX = AMAX1(AAMIN,AMIN1(AAMAX,180.0 - aadel))
      WRITE(2,14) BTMIN,BTDEL,BTMAX,BPMIN,BPDEL,BPMAX,AAMIN,AADEL,
     1        AAMAX
      WRITE(3,14) BTMIN,BTDEL,BTMAX,BPMIN,BPDEL,BPMAX,AAMIN,AADEL,
     1        AAMAX
      write(2,16)
      if (nrat .gt. 0) then
        WRITE(3,15)
        write(*,'('' '')')
        WRITE(*,15)
      else
        WRITE(3,20)
        WRITE(*,20)
      endif
      write(*,'('' '')')
      RETURN
C
3      FORMAT(1X,'Input from a file ',A)
4      FORMAT(/' Statn',T9,'Azimuth',T20,'TOA',T26,
     1    'Key',T31,'Log10 Ratio',T44,
     2    'NumPol',T52,'DenTOA',T60,'Comment')
5      FORMAT(A4,2F8.2,A1,F8.4,1X,A1,1X,F6.2,1X,f6.4)
6      FORMAT(2X,A4,T10,F5.1,T19,F5.1,T27,A1,T32,F8.4,T41,'S',A1,T47,
     .    A1,T53,F6.2,T60,A)
10      FORMAT(2X,A4,T10,F5.1,T19,F5.1,T27,A1,T60,A)
7      FORMAT(' Input:',I4,' P ',I4,' SV and ',I4,' SH polarities and,',
     .    I4,' ratios')
8      FORMAT(' There are no polarity data')
9      FORMAT(' Polarities/Errors:  P ',I3.3,'/',I2.2,'  SV ',
     .    I3.3,'/',I2.2,'  SH ',I3.3,'/',I2.2)
12        FORMAT(' There are no amplitude ratio data')
13        FORMAT(I3,' ratios, maximum of',I3,' with |o-c|',
     1      ' diff > ',
     2      F7.4,'    Focus VP/VS: ',F6.4)
14      FORMAT(' The minimum, increment and maximum B axis trend: ',
     1    3F8.2/' The limits for the B axis',
     2    ' plunge: ',3F8.2/' The limits for Angle: ',3F8.2)
15      FORMAT(T5,'Dip',T11,'Strike',T20,'Rake',T28,'Pol:',
     .    T33,'P',T39,'SV',T45,'SH',
     .    T49,'AccR/TotR',T60,'RMS RErr',T70,'AbsMaxDiff')
16       FORMAT(' ',76('+')/)
17      FORMAT(I4,' P Pol.',I3,' SV Pol.',I3,' SH Pol.',F5.1,' allowed',
     .    ' (weighted) errors')
18      FORMAT(I4,' P Pol.',I3,' SV Pol.',I3,' SH Pol.',I4,' allowed',
     .    '  errors')
19      FORMAT(' Polarities/Errors:  P ',I3.3,'/',F4.1,'  SV ',
     .    I3.3,'/',F4.1,'  SH ',I3.3,'/',F4.1)
20      FORMAT(T5,'Dip',T11,'Strike',T20,'Rake',T28,'Pol:',
     .    T33,'P',T39,'SV',T45,'SH')
      END
