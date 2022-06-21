C+
	program hypo2foc
C
C	Input:  An archive-phase file written by hypoellipse with one
C	  or more events with P polarity information.
C	Output: An input data file that can be read by focmec.
C	Individual files are written for each event.  Output file names are
C	  the first 8 characters in the summary line for each event
C	  --- YYMMDDHH.  The extension is dat --- YYMMDDHH.dat is the  
C	  final filespec.  Comment line is the first 49 characters in
C	  summary line.  Assumption is that each event starts with
C	  a summary line and ends with a line which that starts with
C	  several blanks.  The comment field in each station-polarity
C	  line in the focmec input file will have the P-phase
C	  description (col. 5-7), a space, and the weight (col. 8).
C 
C	26 March 1995  jas/vtso
C     Januaryb2017  Modified so emergent arrivals are so marked and no
C       line is written if no picked polarity
C-
	logical more
	character STA*4, DES*2, POL*1, SENSE*1, WEIGHT*1
	character filespec*80, getstring*80, line*115
	filespec = getstring('Input hypoellipse archive filespec')
	open(unit=1,file=filespec,status='old')
	ierr = 0
	do while (ierr .eq. 0)
	  read(1,'(a)',iostat=ierr) line
	  if (ierr .eq. 0) then
	    do j=1,10
	      if (line(j:j) .eq. ' ') line(j:j) = '0'
	    end do
	    filespec = line(1:10)//'.inp'
	    open(unit=2,file=filespec,status='unknown')
	    line = getstring('Comment line')
	    write(2,'(a)') line(1:lenc(line))
	    more = .true.
	    do while (more)
	      read(1,'(a)') line
	      if (line(1:4) .ne. '    ') then
	        read(line(1:43),'(a4,a2,2a1,T29,F3.0,T41,F3.0)')
     1		  STA,DES,POL,WEIGHT,AZM, AIN
              SENSE = POL
	        if (POL .eq. 'c' .or. POL .eq. 'u') SENSE = 'C'
	        if (POL .eq. 'd') SENSE = 'D'
              if (sense.eq.'C' .and. des.eq.'eP') sense = '+'
              if (sense.eq.'D' .and. des.eq.'eP') sense = '-'
	        if (pol .ne. ' ') write(2,'(A4,2F8.2,A1,T40,A)') STA,
     1             AZM,AIN,SENSE,DES//' '//WEIGHT
	      else
	        close(unit=2)
	        more = .false.
	      end if
	    end do
	  end if
	end do
	stop
	end
