C+
	subroutine linestyle(line_in)
C
C	sgf plot format:  changes line_style, puts in plot file
C	If line_in is greater than 10, asks for value with a default
C	  of line_in-10
C	15 June 1992:  Latest perturbation
C	3 January 2003:  No graphics terminal for this routine
c	3 January 2004:  Took out the where and plot calls
c	1 December 2005: Changed buffer size back to 5004
C-
        integer*2 buffer(5004),line_style,line_width,line_i2
        common /jasplot/  jplot,line_style,line_width
        common /jasdfplot/ xdforgn,ydforgn,xdf,ydf,ndisk,
     &		nword,ilcout,buffer
	character*20 inpen
	if (line_in .gt. 10) then
	  ip = line_in - 10
          WRITE(INPEN,'(''Linestyle value..['',I1,'']'')') IP
	  line_i2 = ivalue(inpen,ip)
	else
	  line_i2 = line_in
	end if
	if (line_i2 .lt. 1) line_i2 = 1
	if (line_i2 .eq. line_style) return
	buffer(nword+1) = -7
	buffer(nword+2) = 1
	buffer(nword+3) = line_i2
	line_style = line_i2
	nword = nword + 3
        if (nword .ge. 4999) call wsgfbuf
c	call where(xnow,ynow,1.0)
c	call plotdf(xnow,ynow,3)
	return
	end
