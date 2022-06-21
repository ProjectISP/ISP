	function lenc(string)
C+
C	function lenc(string)
C
C	Returns length of character variable STRING excluding right-hand
C	  most blanks or nulls
C-
	character*(*) string
	length = len(string)	! total length
	if (length .eq. 0) then
	  lenc = 0
	  return
	end if
	if(ichar(string(length:length)).eq.0)string(length:length) = ' '
	do j=length,1,-1
	  lenc = j
	  if (string(j:j).ne.' ' .and. ichar(string(j:j)).ne.0) return
	end do
	lenc = 0
	return
	end
