C
      SUBROUTINE PLOTDF(X,Y,IPEN)
C
C      THIS SUBROUTINE WRITES TO THE DISK FILE CREATED BY PLOTS.
C      IT CALLS CLOSE WHEN IPEN = 999 AND SO SEVERAL VERSIONS OF
C      A PLOT FILE COULD BE CREATED.
C
C	15 June 1992: sun version.  For sgf files
C	30 December 2004 Disk only non-Sac version
C	3 January 2005:  don't need a move for -3.
c	1 December 2005: Changed buffer size back to 5004
C-
        integer*2 buffer(5004),bufi2(2)
	integer*4 bufi4
	equivalence(bufi2(1),bufi4)
        common /jasdfplot/ xdforgn,ydforgn,xdf,ydf,ndisk,
     &		nword,ilcout,buffer
      IF (IPEN .EQ. 999) THEN
	call wsgfbuf
	buffer(1) = -2
	buffer(2) = 0
        nword = 2
	call wsgfbuf
	nword = 0
	bufi2(1) = -2
	bufi2(2) = 0
	call buf2dsk(ndisk,bufi4,nword,ilcout)
	close(unit=ndisk)
        RETURN
      ENDIF
      IF (IPEN .EQ. 0) RETURN
      IF (IPEN .GE. 4) return
      XDF = amax1(0.0,amin1(10.0,XDFORGN + X))
      YDF = amax1(0.0,amin1(7.50,YDFORGN + Y))
      IF (IPEN.LT.0) THEN         ! Reset the origin if IPEN is negative
          XDFORGN = XDF
          YDFORGN = YDF
	  if (ipen .eq. -3) return
      ENDIF
      if (Iabs(ipen) .eq. 3) then
        buffer(nword+1) = -3
        buffer(nword+2) = 2
        buffer(nword+3) = max0(0,min0(32000,nint(3200*xdf)))
        buffer(nword+4) = max0(0,min0(24000,nint(3200*ydf)))
        nword = nword + 4
        if (nword .ge. 4999) call wsgfbuf
      else if (iabs(ipen) .eq. 2) then
        buffer(nword+1) = max0(0,min0(32000,nint(3200*xdf)))
        buffer(nword+2) = max0(0,min0(24000,nint(3200*ydf)))
        nword = nword +2 
        if (nword .ge. 4999) call wsgfbuf
      end if
      RETURN
      END
