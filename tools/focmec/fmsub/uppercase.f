	subroutine UPPERCASE(string)

c+
c	subroutine UPPERCASE(string)
c
c routine to convert any lower case characters in 'string'
c		      to upper case.
c
c
c					Alan Linde ... January 1987
c-

	character*(*) string

	nchar = lenc(string)

	do i = 1, nchar
	  if (string(i:i).ge.'a' .and. string(i:i).le.'z')
     1			string(i:i) = char(ichar(string(i:i)) - 32)
	enddo

	return
	end
