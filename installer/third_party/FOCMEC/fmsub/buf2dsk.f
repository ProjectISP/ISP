C+
	subroutine buf2dsk(nf,buffer,nword,ntot)
C
c	Fortran version of the SAC zwabs.c which transfers SGF-format
C	buffers to disk file with LUN nf.  Called by wsgfbuf.  The file
C	is an unformatted direct access file with each block 512 bytes
C
C	buffer is an I*4 array containing the SGF commands
C	nword is the number of words (I*4) being transferred.
C	ntot is the cumulative total number of words written to disk
C		ntot is incremented by nword before return.  (This
C		was not done in zwabs.c.)
C	No error checking is done.
C
C	The original version had direct-access write statements of the
C	form write(nf'nblock) block.  When I tried it using g77 it did 
C	work.  Using  write(nf,rec=nblock) block (thanks to Rick Williams
C	of utk), it compiled using g77 on PC/Linux, but the temp.sgf
C	file for test programs were not correct.  As the rec= version
C	works fine on Sun Solaris using Sun's Fortran, I will stick with
C	that.  
C	December 2004: I realized that g77 needs a save
C	November 2006: Need to clear its brain if writing a second .sgf file
C
C	jas/vt
C-
	integer*4 buffer(2500),block(128)
	data now/0/, nblock/0/
	save
	if (nword .lt. 0) then
		now = 0
		nblock = 0
		return
	elseif (nword .gt. 0) then 
	  ntot = ntot + nword
	  jblock = ntot/128
	  if (jblock .eq. nblock) then
	    do j=1,nword
	      block(j+now) = buffer(j)
	    end do
	    now = now + nword
	    return
	  else
	    do j=now+1,128
	      block(j) = buffer(j-now)
	    end do
	    nadded = 128 - now
	    now = 0
	    nblock = nblock + 1
	    write(nf,rec=nblock) block
	    do j=1,128
	      block(j) = 0
	    end do
	    if (nword .gt. nadded) then
	      nleft = nword - nadded
	      kblocks = nleft/128
	      if (kblocks .eq. 0) then
	        do j=1,nleft
		  block(j) = buffer(nadded+j)
		end do
		now = nleft
		return
	      else
	        do k=1,kblocks
		  do j=1,128
		    block(j) = buffer(128*(k-1) + nadded + j)
		  end do
		  nblock = nblock + 1
		  write(nf,rec=nblock) block
		  do j=1,128
	  	    block(j) = 0
	  	  end do
		end do
		now = nleft -kblocks*128
		if (now .gt. 0) then
		  do j=1,now
		    block(j) = buffer(128*kblocks + nadded +j)
		  end do
		  return
		else
		  return  
		end if
	      end if
	    end if
	  end if  
	else    
	  nblock = nblock+1
	  write(nf,rec=nblock) (block(k),k=1,now)
	  return
	end if 
	end
