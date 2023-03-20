      program convert_input

!  The program converts a pre-2017 focmec input file that has amplitude data
!    to the current format.  The earlier input files did not include the
!    v_P/v_S ratio at the source  (a quantity needed to calculate the
!    theoretical amplitude ratio); it was prompted for during the focmec run.
!  One is prompted for names of the input and output files as well as VPVS.

      CHARACTER*1 SENSE,SNUMPOL
      CHARACTER*80 COMMNT,FILENA,CVALUE,DUMMY
      LOGICAL more
      CHARACTER*40 INFO

      FILENA = 
     1    CVALUE('focmec 2009 file name [focmec.inp]',
     2    'focmec.inp',NFILE)
      open(1,file=filena(1:nfile),status='unknown')
      FILENA = 
     1    CVALUE('focmec 2017 input file [focmec2017.inp]',
     2    'focmec2017.inp',NFILE2017)
      open(2,file=filena(1:nfile),status='unknown')
      vpvs = rvalue('v_P/v_S at source [1.73205]',1.73205)
      read(1,'(a)') commnt
      ncom = lenc(commnt)
      write(2,'(a)') commnt(1:ncom)
      more = .true.
      do while (more)
        read(1,'(a)',iostat=ierr) commnt
        if (ierr .eq. 0) then
          ncom = lenc(commnt)
          if (ncom.lt.22 .or. commnt(24:29).eq.'      ') then
            write(2,'(a)') commnt(1:ncom)
          else
            READ(commnt,1) STA,AZIN,TOANG1,SENSE,
     .          RATLOG,SNUMPOL,TOANG2,info
1      FORMAT(A4,2F8.2,A1,F8.4,1X,A1,1X,F6.2,a)
            write(2,2) STA,AZIN,TOANG1,SENSE,
     .          RATLOG,SNUMPOL,TOANG2,vpvs,info
2      FORMAT(A4,2F8.2,A1,F8.4,1X,A1,1X,F6.2,1X,f6.4,a)
          endif
        else
          more = .false.
        endif
      enddo
      stop
      end
