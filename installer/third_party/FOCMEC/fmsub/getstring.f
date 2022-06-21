c+
	character*(*) function GETSTRING(prompt)
c
c  outputs 'prompt' using PRINTX
c  and accepts input character string
c				Alan Linde ... Aug 1986
C       27 July 1993: Did input read through cstring so can have 
C         comment lines
C	12 February 95:  Kill leading blanks
c-
	character*(*) prompt
	character*80 temp
	  getstring = ' '
c output 'prompt'
	call printx(prompt)
	kk=lenc(prompt)
	if (prompt(kk:kk).eq.']') then
	  ll=0
	  do i=kk-1,1,-1
	    if (prompt(i:i).eq.'['.and.ll.eq.0) ll=i+1
	  end do
	  if (ll.ne.0) getstring=prompt(ll:kk-1)
	end if
c  get the response
	call cstring(temp,nout)
c  Kill leading blanks
        do while (nout.gt.1 .and. temp(1:1).eq.' ')
          nout = nout - 1
          temp(1:nout) = temp(2:nout+1)
          temp(nout+1:nout+1) = ' '
        end do
	if (nout .gt. 0) getstring=temp(1:nout)
	return
	end
