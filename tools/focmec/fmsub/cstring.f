C+
	subroutine cstring(string,nstring)
C
C	Input a character string with a read(*,'(A)') string
C	If first two characters are /* it will read the next entry
C	Tab is a delimiter.
C	Returns string and nstring, number of characters to tab.
C	string stars with first non-blank character.
C       25 May 2001.  Took out parameter statement for tab.
C-
	logical more
	CHARACTER*1 TAB
	CHARACTER*(*) string
C
	tab = char(9)
	more = .true.
	do while (more)
	  read(*,'(A)') string
	  nstring = lenc(string)
	  more = (nstring.ge.2 .and. string(1:2).eq.'/*')
	end do
	IF (nstring .GT. 0) THEN
	  NTAB = INDEX(string(1:nstring),TAB)
	  IF (NTAB .GT. 0) nstring = NTAB - 1
	end if
	return
	end
