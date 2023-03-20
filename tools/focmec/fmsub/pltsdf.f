C+
      SUBROUTINE PLTSDF(nunit,filena)
C
C	PLOT INITIALIZATION ROUTINE FOR PLOTS WRITTEN TO DISK FILES
C
C	does not use the SAC library: everything is in Fortran
c	1 December 2005: Changed buffer size back to 5004
c	November 2006: Made changes to allow for a second .sgf file
c	  to be written within a program.
C-
        integer*2 buffer(5004),line_style,line_width
        common /jasplot/  jplot,line_style,line_width
        common /jasdfplot/ xdforgn,ydforgn,xdf,ydf,ndisk,
     &		nword,ilcout,buffer
	character*80 filena
c	integer system
C
      ndisk = nunit
      xdforgn = 0.0
      ydforgn = 0.0
      xdf = 0.0
      ydf = 0.0
c
c	Clear the brain in the disk writer (needed for second .sgf file)
c	
      nword = -10
      call buf2dsk(ndisk,idummy,nword,ntot)
      CLOSE(UNIT=ndisk)
      nf = lenc(filena)
C
       open(ndisk,file=filena(1:nf),status='unknown',
     1      form='unformatted',recl=512,access='direct',iostat=nerr)
C
c	call zopenc(ndisk,filena(1:nf),.true.,.false.,nf,nerr)
c
	if (nerr .ne. 0) then
	  write(*,*) 'Error opening plot output file, nerr =',nerr
	  stop
	end if
C
C	boiler plate start stuff
C
	buffer(1) = -4
	buffer(2) = 1
	buffer(3) = 7
	buffer(4) = -7
	buffer(5) = 1
	buffer(6) = 1
	buffer(7) = -6
	buffer(8) = 2
	buffer(9) = 426
	buffer(10) = 640
	nword = 10
	ilcout = 0 
C
C	Use sac write to unformatted direct access file
C
	call wsgfbuf
	return
	end
