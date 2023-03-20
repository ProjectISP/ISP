C+
	subroutine wsgfbuf
C
C	Write an sgf buffer into an sgf file.  Rests for more writes.
C	15 June 1992:  Latest perturbation
C	30 December 2004: non-SAC version  jas/vt
c	1 December 2005: Changed buffer size back to 5004
C-
        integer*2 buffer(5004), buffi2 (5004)
	integer*4 buffi4(2502),one/1/
	equivalence (buffi2,buffi4)
        common /jasdfplot/ xdforgn,ydforgn,xdf,ydf,ndisk,
     &		nword,ilcout,buffer
        if (2*(nword/2) .LT. nword) then
          nword = nword + 1
          buffer(nword) = -1
        end if
	do j=1,nword
		buffi2(j) = buffer(j)
	enddo
        ilnbuf = nword/2
	call buf2dsk(ndisk,ilnbuf,one,ilcout)
C        call zwabs(ndisk,ilnbuf,1,ilcout,nerr)
c        ilcout = ilcout + 1
	call buf2dsk(ndisk,buffi4,ilnbuf,ilcout)
C        call zwabs(ndisk,buffer,ilnbuf,ilcout,nerr)
c        ilcout = ilcout + ilnbuf
        nword =0
	return
	end
