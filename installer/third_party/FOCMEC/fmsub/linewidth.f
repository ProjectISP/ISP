C+
	subroutine linewidth(inpen)
C
C	sgf plot format:  changes newpen value.  Puts in common
C	If inpen is greater than 10, asks for value with a default
C	  of inpen-10
C	uses sac code -9 to be consistent with Tapley
C	15 June 1992:  Latest perturbation
c	3 January 2004:  Took out the where and plot calls
c	1 December 2005: Changed buffer size back to 5004
C-
        integer*2 buffer(5004),line_style,line_width,ipen_i2
        common /jasplot/  jplot,line_style,line_width
        common /jasdfplot/ xdforgn,ydforgn,xdf,ydf,ndisk,
     &		nword,ilcout,buffer
	character*20 pen
	entry newpen(inpen)
	if (jplot .eq. 1) return
	if (inpen .gt. 10) then
	  ip = inpen - 10
          WRITE(PEN,'(''Linewidth value..['',I1,'']'')') IP
	  itestpen = ivalue(pen,ip)
	else
	  itestpen = inpen
	end if
	if (itestpen .lt. 1) itestpen = 1
	ipen_i2 = itestpen
	if (ipen_i2 .eq. line_width) return
	line_width = ipen_i2
	buffer(nword+1) = -9
	buffer(nword+2) = 1
	buffer(nword+3) = line_width
	nword = nword + 3
        if (nword .ge. 4999) call wsgfbuf
c	call where(xnow,ynow,1.0)
c	call plotdf(xnow,ynow,3)
	return
	end
