C+
	PROGRAM freesurf
C
C	Given a VS/VP ratio, calculates the amplitudes for P on Z,
C	  P on R, SV on Z, and SV on R for input emergence or apparent
C	  angles.  One can choose between these two angle input modes.
C	  If one knows the slowness, the proper choice is emergence
C	  angle.  If the angle is estimated from observed particle
C	  motion or ratio of P on R to P on Z, apparent angle is
C	  appropriate.  However, if one uses apparent angle, it is 
C	  assumed that the S and P have the same emergence angle.
C	An application of this program is the calculation of free-surface
C	  corrections for SV/P, SH/P or SV/SH amplitude ratios.  Another
C	  is to convert from SV measured on the vertical component to SV
C	  which would be measured on the radial component.  Program fmecprep
C	  assumes SV is from the radial component.
C	4 April 2000  jas/vt
C	14 July 2002: replaced gdmot with grndmtn
C	19 July 2008: last perturbation
C-
	LOGICAL TRUTH,more_stat
	character*80 com,getstring
	RD = 45.0/ATAN(1.0)
	ash = 2.0
	open(2,file='freesurf.lst',status='unknown')
	call timdat(2,'freesurf')
	com = getstring('Enter a comment')
	write(2,*) com(1:lenc(com))
	vsvp = value('VS to VP ratio')
	if (vsvp .ge. 1.0) then
	  write(*,*) 'vsvp must be less than 1.0'
	  stop
	endif
	angcrit = rd*asin(vsvp)
	write(*,*) vsvp,' =  VS to VP ratio  ', 
     1		angcrit,' is the critical angle'
	write(*,*) 'Avoid SV emergence at or above critical angle'
	write(2,*) vsvp,' =  VS to VP ratio  ', 
     1		angcrit,' is the critical angle'
        write(2,*) ' Avoid SV emergence at or above critical angle'
	write(*,*) 'All angles defined with respect to up'
	more_stat = .true.
	do while (more_stat)
	  write(2,*) ' '
	  com = getstring('Station identifier')
	  Pang = value('Emergence P-arrival Angle in degrees')
	  Sang = value('Emergence S-arrival Angle in degrees')
	  write(2,*) com(1:lenc(com)),'  Emerg P Angle:',Pang,
     1		'  S Angle:',Sang
          if (Sang .gt. angcrit) then
            write(*,*)'WARNING: SV emergence at or above critical angle'
            write(2,*)'WARNING: SV emergence at or above critical angle'
	  end if
	  call grndmtn(rd,1,vsvp,Pang,apr,apv,phr,phv)
	  call grndmtn(rd,2,vsvp,Sang,asr,asv,phr,phv)
	  write(2,*) 'P amplitudes on Z and R:',apv, apr
	  write(2,*) 'S amplitudes on Z and R:',asv, asr
	  if (asr.ne.0.0) write(2,*) apv/asr,' = SV(R)/P(Z) correction'
	  if (asv.ne.0.0) write(2,*) apv/asv,' = SV(Z)/P(Z) correction'
	  write(2,*) apv/ash,' = SH(T)/P(Z) correction'
	  if (asr.ne.0.0) write(2,*) ash/asr,' = SV(R)/SH(T) correction'
	  if (asv.ne.0.0) write(2,*) ash/asv,' = SV(Z)/SH(T) correction'
	  if (asv.ne.0.0) write(2,*) asr/asv,' = SV(R)/SV(Z)'
	  more_stat = truth('More stations?..[Y]')
	end do
	stop
	end
