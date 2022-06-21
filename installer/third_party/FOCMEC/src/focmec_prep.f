C+
	program focmec_prep
C
C	Thefirst input file called is created by program ratio_prep or some
C	  other means which has data for each station such as takeoff and
C	  emergence angles for both P and S, the azimuth, and the surface
C	  S-to-P velocity ratio.  Format for this file is as follows:
C		one comment line
C		one line starts with S/P surface and P/S focus velocity ratios
C		one line with column headers describing the station output
C		one line per station. Name, dist, az, Ptoa, Stoa, aemrgP,
C		aemrgS, tp, ts.  Format is A4,F9.4,5F9.3,2f9.4 for each line.
C	The second input file called is a stripped-down Focmec input file
C	  containing (on separate lines) the station name, the polarity-ratio 
C	  key, and, if an amplitude ratio, the polarity key for the numerator 
C	  in the rato plus either two or six numbers separated by commas
C	  --- the numerator and denominator amplitudes 
C	  plus (if used) attenuation data in the form Qnum, Qden, freqnum, 
C	  freqden, where the freq terms are the frequencies at which the 
C	  amplitudes were determined.  The first line in the file is a
C	  comment.  The remaining lines have the format (A4,1X,A1) for
C	  polarities and (A4,1x,A1,1X,A1,1X,*) for ratio data, where
C	  the * means it is free format so long as comma delimieters are 
C	  used.  
C	Output is a Focmec input file.
C
C	12 April 2000  jas/vt
C	12 September 2002: Reordered the comments above to reflect the
C	  actual order in which the input files are called by fmecprep
C	July 2008: Changed name from fmecprep to focmec_prep
C	July 2016: now both focus and surface velocity ratios are input
C	  from a file and P/S focus ratio is output to focmec input file
C       July 2017: Added a significant figure for distance input 
C-
	parameter (nsta=500)
	character*1 sense
	character*4 sta(nsta)
	dimension Ptoang(nsta),Stoang(nsta),dist(nsta),Pemrga(nsta)
	dimension Semrga(nsta),tp(nsta),ts(nsta),az(nsta)
	CHARACTER*80 COMMNT,FILENA,CVALUE,info
	logical more
C
	pi = 4.0*atan(1.0)
	rddeg = 180.0/pi
	open(2,file='focmec_prep.lst',status='unknown')
	CALL TIMDAT(2,'focmec_prep')
c
c	Open file with stations and epicentral distances.
C	Format is A4 and then a single floating point number in free format
C	There can be up to 500 stations
C
	FILENA = 
     1    CVALUE('Input focmec_prep filespec (prepared by ratio_prep)',
     2    'dummy',NFILE)
	open(3,file=filena(1:nfile),status='old')
	write(2,*) 'focmec_prep input file from file ',
     1		filena(1:nfile)
	read(3,'(a)') commnt
	write(2,*) commnt(1:lenc(commnt))
        read(3,*) vsvp_surf,vpvs_focus
        backspace 3
	read(3,'(a)') commnt
	write(2,*) commnt(1:lenc(commnt))
	read(3,'(a)') commnt
	write(2,*) commnt(1:lenc(commnt))
	j = 1
	ierr = 0
	do while (ierr.eq.0 .and. j.le.nsta)
	  read(3,'(a)',iostat=ierr) commnt
	  if (ierr .eq. 0) then
	    write(2,*) commnt(1:lenc(commnt))
	    read(commnt,'(A4,F10.5,f8.3,6F9.3)') sta(j),dist(j),az(j),
     1	        Ptoang(j),Stoang(j),Pemrga(j),Semrga(j),tp(j),ts(j)
	    call uppercase(sta(j))
	    N = j
	    j = j + 1
	  end if
	end do
	close(3)
	FILENA = 
     1    CVALUE('Input focmec data filespec',
     2    'dummy',NFILE)
	open(3,file=filena(1:nfile),status='old')
	write(2,*) 'Input focmec data filespec ',filena(1:nfile)
	FILENA =
     1    CVALUE('Filespec for new focmec input file',
     2    'dummy',NFILE)
	open(4,file=filena(1:nfile),status='unknown')
	write(2,*) 'focmec input filespec ',filena(1:nfile)
	read(3,'(a)') commnt
	write(4,'(a)') commnt(1:lenc(commnt))
	write(2,*) commnt(1:lenc(commnt))
	ierr = 0
	do while (ierr .eq. 0)
	  read(3,'(a)',iostat=ierr) commnt
	  if (ierr .eq. 0) then
	    lc = lenc(commnt)
	    write(2,*) 'Input: ',commnt(1:lc)
	    more = .true.
	    j = 1
	    do while (more .and. j.le.N)
	      call uppercase(commnt(1:4))
	      if (commnt(1:4) .eq. sta(j)) then
	        k = j
	        more = .false.
		sense = commnt(6:6)
		if (lc .gt. 6) then 
		  info = commnt(7:lc)
		  commnt(30:31) = info(1:2)
		  commnt(39:39) = ' '
		  write(commnt(40:45),'(f6.4)') vpvs_focus
		  commnt(46:46) = ' '
		  commnt(47:47+lc-10) = info(4:lc-6)
		  lc = 46 + lc - 9
		else
		  lc = 21
		end if
		write(commnt(5:12),'(F8.2)') az(j)
		commnt(21:21) = sense
	      else
		j = j + 1
	      end if
	    end do
	    if (more) then
	      write(*,*) 'No station ',commnt(1:4),' in list'
	      write(2,*) 'No station ',commnt(1:4),' in list'
	      stop
	    end if
	    if (sense.eq.'C' .or. sense.eq.'U' .or. sense.eq.'D' .or.
     1		sense.eq.'+' .or. sense.eq.'-' .or. sense.eq.'e') then
	      toang1 = Ptoang(k)
	    else if (sense.eq.'<' .or. sense.eq.'>' .or. sense.eq.'F'
     1		.or. sense.eq.'B' .or. sense.eq.'l' .or. sense.eq.'r'
     2	        .or. sense.eq.'u' .or. sense.eq.'L'
     3		.or. sense.eq.'R') then
	      toang1 = Stoang(k)
	    else
	      if (sense.eq.'V' .or. sense.eq.'H' .or.
     1		   sense.eq.'S') then
		if (lenc(commnt) .le. 39) then
		  write(*,*) 'No ratio info in line ',
     1			commnt(1:lc)
		  stop
		end if
		toang1 = Stoang(k)
		if (sense.eq.'V' .or. sense.eq.'H') toang2 = Ptoang(k)
		if (sense .eq. 'S') toang2 = toang1
	        write(commnt(32:38),'(F7.2)') toang2
		call getrat(commnt,Pemrga(k),Semrga(k),tp(k),ts(k),
     1			vsvp_surf,pi,rddeg)
	      else
		write(*,*) sense,' is not a legal opton'
		stop
	      end if
	    end if
	    write(commnt(13:20),'(F8.2)') toang1
	    write(4,'(a)') commnt(1:lc)
	    write(2,*) commnt(1:lc)
	  end if
	end do
	stop
	end
C+
	subroutine getrat(commnt,Pemrga,Semrga,tp,ts,vsvp_surf,pi,rddeg)
C
C	Starting in position 46, commnt has amplitudes, Q values, and
C	dominant frequencies for focmec input for which plitude ratio
C	data are given.  This subroutine corrects the ampltudes for
C	the free-surface (using the emergance angles for SV and P) and
C	for Q.  The two or six entries are entered as floating-point
C	numbers with commas as delimiters.  If only two numbers are
C	given, there is no Q correction.  The result of the correction,
C	the log10 of the amplitude ratio, is returned in the appropraite
C	positions in the character string commnt.
C
C	29 March 2000: jas/vt
C	13 July 2002: put in warnings for SV if SV greater than the
C	  critical angle
C-
	character commnt*80, sense*1
c-
	sense = commnt(21:21)
	lc = lenc(commnt)
	j1 = 45 + index(commnt(46:lc),',')
	if (j1 .le. 40) then
		write(*,*) 'No delimiter in ',commnt(40:lc)
		stop
	end if
	read(commnt(46:j1-1),*) anum
	j2 = j1 + index(commnt(j1+1:lc),',')
	if (j2 .le. J1) then
	  read(commnt(j1+1:lc),*) aden
	  qfact = 1.0
	  obrat = anum/aden
	else
	  read(commnt(j1+1:j2-1),*) aden
	  obrat = anum/aden
	  j3 = j2 + index(commnt(j2+1:lc),',')
	  if (j3 .le. j2) then
		write(*,*) 'Not enough delimiters in ',commnt(40:lc)
		stop
	  end if
	  j4 = j3 + index(commnt(j3+1:lc),',')
	  if (j4 .le. j3) then
		write(*,*) 'Not enough delimiters in ',commnt(40:lc)
		stop
	  end if
	  j5 = j4 + index(commnt(j4+1:lc),',')
	  if (j5 .le. j4) then
		write(*,*) 'Not enough delimiters in ',commnt(40:lc)
		stop
	  end if
	  read(commnt(j2+1:j3-1),*) qs
	  read(commnt(j3+1:j4-1),*) qp
	  read(commnt(j4+1:j5-1),*) fs
	  read(commnt(j5+1:lc),*) fp
	  factor = pi*ts*fs/qs - pi*tp*fp/qp
	  qfact = exp(factor)
	  write(2,*) 'Q correction:',qfact
	end if
C
C	Need free-surface correction.  If sense .eq. S (SV/SH), the
C	denominator factor is 2, and the numerator factor is for
C	SV on the radial component.  For sense .eq. V (SV/P) 
C	or sense .eq. H (SH/P), the denominator facror is for P on the
C	vertical.  If SV is observed on the vertical, one has to use
C	program freesurf to multiply the observed SV amplitude by Sar/Sav.
C	To correct for the free-surface effect, one must divide by the
C	amplification ratio.  WARNING: if SV emergence is at or above
C	the critical angle, the results are unreliable particularly for
C       SV on the radial.
C
	call grndmtn(rddeg,2,vsvp_surf,Semrga,Sar,Sav,pr,pv)
	angcrit = rddeg*asin(vsvp_surf)
	if (sense .eq. 'S') then
	  fsfact = 2.0/Sar
	  if (semrga .ge. angcrit) then
	     write(*,*)'WARNING SV emerge and critical angle:',semrga,angcrit
	     write(2,*)'WARNING SV emerge and critical angle:',semrga,angcrit
	  end if
	else
	  call grndmtn(rddeg,1,vsvp_surf,Pemrga,Par,Pav,pr,pv)
	  if (sense .eq. 'V') then
	    fsfact = Pav/Sar
	    if (semrga .ge. angcrit) then
	     write(*,*)'WARNING SV emerge and critical angle:',semrga,angcrit
	     write(2,*)'WARNING SV emerge and critical angle:',semrga,angcrit
	     write(2,*) 'Sar/Sav =',Sar/Sav
	    end if
	  end if
	  if (sense .eq. 'H') fsfact = Pav/2.0
	end if
c
c	put it together
c
	ratl10 = alog10(abs(fsfact*qfact*obrat))
	write(commnt(22:29),'(F8.4)') ratl10
	write(2,*) 'Observed ratio:',obrat,' Free-surface:',fsfact,
     1         ' Q correction:',qfact
	return
	end
