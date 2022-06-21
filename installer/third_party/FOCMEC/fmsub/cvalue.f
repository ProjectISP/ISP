C+
	character*(*) function cvalue(msg,default,nout)
C
C	MSG gets printed on screen prompting for a character string.
C	DEFAULT is the default string.  NOUT is the number of characters
C	  returned in cvalue.
C	A tab in the input acts as a terminator.
C       27 July 1993: Did input read through cstring so can have 
C         comment lines
C	25 May 2001.  Took out parameter statement for tab.
C-
	character*1 tab
	character*(*) msg,default
	character*80 input
C
        tab = char(9)
	call printx(msg)
	call cstring(input,nout)
c  Kill leading blanks
        do while (nout.gt.0 .and. input(1:1).eq.' ')
          nout = nout - 1
          input(1:nout) = input(2:nout+1)
          input(nout+1:nout+1) = ' '
        end do
	if (nout .eq. 0) then
	  nout = lenc(default)
	  cvalue = default(1:nout)
	else
	  cvalue = input(1:nout)
	end if
	return
	end
