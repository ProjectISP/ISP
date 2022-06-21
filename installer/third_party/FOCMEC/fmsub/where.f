C+
	SUBROUTINE WHERE (X,Y,FACT)
C
C	Returns X=XNOW and Y=YNOW relative to the plotting origin.  FACT
C	  in this version is used to designate whether the coordinates
C	  are for a plot file (FACT>0) or a graphics device (FACT<0)
C	15 June 1992:  Latest perturbation
c	1 December 2005: Changed buffer size back to 5004
C-
        integer*2 buffer(5004),line_style,line_width
        common /jasplot/  jplot,line_style,line_width
        character*16 device
	common /jasgtplot/ xgtorgn,ygtorgn,xgt,ygt,scale_factor,device
        common /jasdfplot/ xdforgn,ydforgn,xdf,ydf,ndisk,
     &		nword,ilcout,buffer
	if (fact.gt.0.0 .and. jplot.ge.2) then
	  x = xdf - xdforgn
	  y = ydf - ydforgn
	else if (fact.lt.0.0 .and. jplot.le.2) then
	  x = xgt - xgtorgn
	  y = ygt - ygtorgn
	else
	  write(*,*) 'ERROR IN WHERE: MISMATCH OF JPLOT and FACT'
	  stop
	end if
	RETURN
	END
