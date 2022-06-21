C+
      program ratio_prep_1layer
C
C      Reads in a file with a latitude, longitude, and depth for an event
C        plus a list of statin names.  Calculates epicentral distances,
C        station-based azimuth, P and S travel times and take-off angles 
C        for each station, and the S-to-P surface velocity ratio.
C      This version does not use traveltime tables.
C        It assumes a single layer and all rays go up.
C      Input file format:
C            Line 1: latitude, longitude, depth, v_P, v_S (free format).
C                  Angles degrees/decimal degrees; W and S are negative.
C                  Depth is in km.
C            Lines 2 through N+1: Station names (A4) for N stations
C      Also required:
C            stations.loc one line per station
C                  Station name (A5), latitude, longitude, elevation
C                       free format, but input depth is in meters
C                       subroutine stations_loc converts it to km real*4
C      Output file format:
C            one comment line
C            one line that starts with the S/P surface velocity ratio 
C                and P/S focus velocity ratio
C            one line with column headers describing the station output
C            one line per station. Name, dist, az, Ptoang, Stoang, Pemrga,
C            Semrga,tp, ts.  Format is A4,F9.4,7F9.3 for each line.
C      The output file is used as an input file for focmec_prep.
C      A second output file, fileprep.lst, has a summary of the run.
C      July 2017 version now adds elevation to depth for each station.
C          in .lst file prints out differences
C-
      parameter (nsta=500)
      character*8 sta(nsta),char8
      dimension edist(nsta),Pemrga(nsta),tsz(nsta),tpz(nsta)
      dimension Semrga(nsta),tp(nsta),ts(nsta),az(nsta),selev(nsta),
     1     z(nsta)
      CHARACTER*80 FILENA,CVALUE,commnt
      real*8 elatin/0.0d0/,elongin/0.0d0/
c
c      Open file with hypocenter and up to 500 station names
C
      rd = 45.0/atan(1.0)
      open(unit=2,file='ratio_prep_1layer.lst',status='unknown')
      call timdat(2,'ratio_prep_1layer')
      FILENA =
     1    CVALUE('Station locations filespec [../stations.loc]',
     2    '../stations.loc',NFILE)
      open(3,file=filena(1:nfile),status='old')
      write(2,*) 'Station locations from ', filena(1:nfile)
      FILENA = 
     1    CVALUE('Input filespec with hypocenter and stations',
     2    'dummy',NFILE)
      open(1,file=filena(1:nfile),status='old')
      read(1,*) elatin, elongin, focdep, v_P, v_S
      elat = sngl(elatin)
      elong = sngl(elongin)
      write(2,*) 'Hypocenter & Station names from ',filena(1:nfile)
      write(2,'(a,f10.4,a,f10.4,a,f8.3)') 
     1      'Latitude:',elat,' Longitude:',elong,' Depth:',focdep
      elatr = elat/rd
      elongr = elong/rd
      j = 1
      ierr = 0
      do while (ierr.eq.0 .and. j.le.nsta)
        read(1,'(a)',iostat=ierr) char8
        if (ierr .eq. 0) then
          sta(j) = char8
          call stations(3,sta(j),slat,slong,selev(j))
          slatr = slat/rd
          slongr = slong/rd
          call disaz(elatr,elongr,slatr,slongr,azr,baz,edistr)          
          az(j) = rd*azr
          edist(j) = rd*edistr
          N = j
          j = j + 1
        end if
      end do
      close(1)
      close(3)
      call getrays(N,edist,focdep,Pemrga,tp,ts,
     1            v_P,v_S,selev)
      FILENA = 
     1    CVALUE('Filespec of fmecprep input file',
     2    'dummy',NFILE)
      open(1,file=filena(1:nfile),status='unknown')
      write(2,*) 'fmecprep input filespec: ',filena(1:nfile)
      commnt = CVALUE('Comment line','dummy',ncom)
      write(1,'(f8.4,f10.4,f8.3,2x,a)')elat,elong,focdep,commnt(1:ncom)
      write(2,'(f8.4,f10.4,f8.3,2x,a)')elat,elong,focdep,commnt(1:ncom)
      vsvp_surf = v_s/v_p
      vpvs_foc = v_p/v_s
      write(1,'(f6.4,1x,f6.4,a)') vsvp_surf,vpvs_foc,
     1    ' S/P surface and P/S focus velocity ratios'
      write(2,'(f6.4,1x,f6.4,a)') vsvp_surf,vpvs_foc ,
     1    ' S/P surface and P/S focus velocity ratios'
      write(1,'("Stn",t8,"Dist",t17,"Az",T26,"Ptoa",T35,"Stoa",T44,
     1    "Pemrg",T53,"Semrg",T62,"TT(P)",T71,"TT(S)")')
      do j=1,N
        toang = 180.0 - Pemrga(j)
        write(1,'(A4,F10.5,f8.3,6F9.3)') sta(j),edist(j),az(j),toang,
     1            toang,Pemrga(j),Pemrga(j),tp(j),ts(j)
        z(j) = 0.0
      end do

! Now do it for zero station elevation to find elevation correction

      call getrays(N,edist,focdep,Pemrga,tpz,tsz,
     1            v_P,v_S,z)
      write(2,'("Stn",t8,"EmergA",t17,"TTZ(P)",T26,"TTZ(S)",
     1    t35,"tp_elev",t44,"htp_elev",t53,"ts_elev",t62,"hts_elev",
     2   t71,"stelev")')  
      do j=1,N
        d_lahr = cos(Pemrga(j)/rd)
        write(2,'(A4,F9.4,7F9.3)') sta(j),Pemrga(j),
     1        tpz(j),tsz(j),tp(j)- tpz(j),d_lahr*selev(j)/v_P,
     2       ts(j)-tsz(j),d_lahr*selev(j)/v_S,selev(j)
      end do
      stop
      end

      subroutine getrays(N,edist,focdep,Pemrga,
     1            tp,ts,v_p,v_s,selev)

!  2 July 2017: Adds selev to focal depth for each station

      parameter (max=60)
      logical prnt(3)/3*.false./
      character*8 phcd(max),phlst(10)
      character*50 modnam,cvalue
      dimension edist(*),Pemrga(*)
      dimension tp(*),ts(*),selev(*)
c
      RD = 45.0/ATAN(1.0)
C
      do j=1,N
        edistkm = 111.195*edist(j)
        totdep = focdep + selev(j)
        dist = sqrt(edistkm**2 + totdep**2)
        tp(j) = dist/v_p
        ts(j) = dist/v_s
        Pemrga(j) = rd*acos(totdep/dist)
      end do
      return
      end
C+
      SUBROUTINE STATIONS(NUN,STA,SLAT,SLONG,selev)
C
C  FINDS STATION WITH NAME STA FROM FILE NUN
C  IF STA NOT ON LIST, it stops the program
c  8 June 2011: increased size of a station name to eight
c    not used in this packge, only four at most      
c  2 July 2017: Now reads in elevation.  Using Lahr's convention,
c    stored as integer meters, so converts to km for return
C-
      LOGICAL MORE,ANOTHER,TRUTH
      character*(*) sta
      CHARACTER*8 TSTA
      CHARACTER*100 RECORD
      nsta = lenc(sta)
      tsta = sta(1:nsta)
      call uppercase(tsta)
      MORE = .TRUE.
      DO WHILE (MORE)
        REWIND NUN
        ANOTHER = .TRUE.
        DO WHILE (ANOTHER)
          READ(NUN,'(A)',IOSTAT=IERR) RECORD
          IF (IERR .EQ. 0) THEN
          call uppercase(record(1:nsta))
          if (tsta .eq. record(1:nsta)) then
            I = nsta + 1
                READ(RECORD(I:lenc(record)),*) SLAT,SLONG,IELEV
                selev = ielev/1000.0
                ANOTHER = .FALSE.
                MORE = .FALSE.
            END IF
          ELSE
            WRITE(*,'('' Exiting: No station called '',
     1            A8,'' listed'')') tsta
          stop
          END IF
        END DO
      END DO
      RETURN
      END
C+
C     SUBROUTINE DISAZ(ELAT,ELONG,SLAT,SLONG,AZE,AZS,DIST)
C
C  CALCULATES DISTANCES AND AZIMUTHS FOR A SPHERICAL EARTH
C    USING GEOCENTRIC LATITUDES INSTEAD OF GEOGRAPHIC ONES
C    BETWEEN AN (E)VENT AND A (S)EISMOGRAPH STATION.
C  ALL ANGLES - INCLUDING DIST - ARE IN RADIANS.
C  ASSUMES LATITUDES RUN FROM -PI/2 TO +PI/2 FROM S TO N, AND
C          LONGITUDES FROM -PI TO +PI FROM W TO E.
C  TO DERIVE THE EQUATIONS, DEFINE A SPHERICAL TRAINGLE WITH ONE VERTEX
C    AT THE NORTH POLE, ONE AT S AND THE OTHER AT E.
C   12 April 2011: Only change from 1991 is calculating pi not just data.
C-
      SUBROUTINE DISAZ(ELAT,ELONG,SLAT,SLONG,AZE,AZS,DIST)
      DOUBLE PRECISION B,PI,ELA,SLA,ESLON,PI2,TWOPI,ELON,SLON,
     .    COSDEL,COSELA,SINELA,COSSLA,SINSLA
      DATA B/0.9932773D0/
      pi2 = 2.0d0*datan(1.0d0)
        pi = 2.0d0*pi2
        twopi = 4.0d0*pi2
      if ((elat.eq.slat) .and. (elong.eq.slong)) then
        aze = 0.0
        azs = pi
        dist = 0.0
        return
      end if
      ELA = PI2 - DATAN2(B*DSIN(DBLE(ELAT)),DCOS(DBLE(ELAT)))
      SLA = PI2 - DATAN2(B*DSIN(DBLE(SLAT)),DCOS(DBLE(SLAT)))
      ELON = ELONG
      IF (ELON .LT. 0.0) ELON = ELON + TWOPI
      SLON = SLONG
      IF (SLON .LT. 0.0) SLON = SLON + TWOPI
      ESLON = ELON - SLON
      COSELA = DCOS(ELA)
      SINELA = DSIN(ELA)
      COSSLA = DCOS(SLA)
      SINSLA = DSIN(SLA)
      COSDEL = COSELA*COSSLA+SINELA*SINSLA*DCOS(ESLON)
      DIST = DACOS(COSDEL)
      SINDEL = SIN(DIST)
      FACTOR = -DSIN(ESLON)/SINDEL
      SINAZE = FACTOR*SINSLA
      COSAZE = (COSSLA-COSELA*COSDEL)/(SINELA*SINDEL)
      AZE = ATAN2(SINAZE,COSAZE)
      IF (AZE .LT. 0.0) AZE = AZE + TWOPI
      SINAZS = -FACTOR*SINELA
      COSAZS = (COSELA-COSSLA*COSDEL)/(SINSLA*SINDEL)
      AZS = ATAN2(SINAZS,COSAZS)
      IF (AZS .LT. 0.0) AZS = AZS + TWOPI
      RETURN
      END
