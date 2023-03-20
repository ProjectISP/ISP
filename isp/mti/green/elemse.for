      program elemse

c          6 elementary seismograms for basic focal mech.
c          (the 6th is isotropic)
c          moment rate = delta fctn. , output = velocity

c         time function from file soutype.dat !!!!!!!

ccc       fixed  N=8192 (mm=13)

c
c   Elementary seismograms
c   Program  modified from CONVOL (by O. Coutant)
c   J.Zahradnik, 2003
ccccc
c

	
      integer      nsp,nrp,ncp,ntp,nrtp,nrtp2,icc,ics,ifr
      parameter    (nsp=1,ncp=200,nrp=31)
      
!       parameter   (nsp=1,ncp=200,nrp=31,ntp=8192,nrtp=nrp*ntp) ! JV


      integer     jf,ir,it,nc,ns,nr,nfreq,ikmax,mm,nt  ! ,iwk(ntp)
      integer,allocatable :: iwk(:)
      real        tl,xl,uconv,
     3            dfreq,freq,
     4            pi,pi2,aw,ck,cc,
     5            pas,
     7            t1
c     real        a(6)
      real        delay ! JV
      complex*16,allocatable :: ux(:,:,:),uy(:,:,:),uz(:,:,:)
      complex*16  omega, ! ux(ntp,nrp,6),uy(ntp,nrp,6),uz(ntp,nrp,6),
     1            uxf(6),uyf(6),uzf(6),ai,deriv,fsource,us


      real bux,buy,buz
      CHARACTER*255 infile,dum

      namelist    /input/ nc,nfreq,tl,aw,nr,ns,xl,
     &                    ikmax,uconv,fref

      CALL getarg(1,dum);read(dum,*)mm ! JV
      ntp=2**mm
      nrtp=nrp*ntp
      nrtp2=nrtp*6
      allocate(iwk(ntp))
      allocate(ux(ntp,nrp,6))
      allocate(uy(ntp,nrp,6))
      allocate(uz(ntp,nrp,6))

c      data        ux,uy,uz/nrtp2*(0.,0.),nrtp2*(0.,0.),
c     &                     nrtp2*(0.,0.)/

 	do it=1,6	 ! for large 3D files the DAT statement cannot be used 
 	do ir=1,nrp	 ! (compilation last long time or even cannot succssfully end)
 	do itim=1,ntp  			   ! thn THIS initiation solves the problem
 	ux(itim,ir,it)=(0.,0.)
 	uy(itim,ir,it)=(0.,0.)
 	uz(itim,ir,it)=(0.,0.)
      enddo
      enddo
      enddo
 
      pi=3.1415926535
      pi2=2.*pi
      ai=(0.,1.)        

	CALL getarg(2,infile)
      open (12,form='unformatted',file='gr'//trim(infile)//'.hes')
      open (10,form='formatted',file='gr'//trim(infile)//'.hea')
c      open (130,form='formatted',file='spefct.dat')
cc    open(350,form='formatted',file='elemsef.dat') ! elem seis formated
      open(300,form='unformatted',file='elemse'//trim(infile)//'.dat') ! elem seis
                                                   ! (all in a SINGLE file)
                                                   ! SUCCESSION OF STATS
                                                   ! AS IN GREEN
      CALL getarg(3,dum);read(dum,*)delay ! JV



      read(10,input)

      if ((ns.gt.nsp).or.(nr.gt.nrp).or.(nc.gt.ncp)) then
       write(0,*) "check dimensions!"
       stop
      endif
      if(ns.gt.1) then
       write(*,*) 'more than one source not allowed in this version'
       stop
      endif

c  true step
c      ics=7   !true step
c      t1=0.
c      icc=2   !output = velocity

cccccccc  read the source type from file !!! thimios 28/3/10

	open(301,file='soutype.dat')
	read(301,'(i1)') ics
	read(301,'(f4.1)') t0
	read(301,'(f3.1)') t1
      read(301,'(i1)') icc
	close(301)
cccccccccccccccccccccccccc
c  moment time fctn fsource
c     ics=2                ! Bouchon's smooth step
c     t0=0.5
c     t1=1.
c     icc=2                ! output time series will be velocity
cccccccccccccccccccccccccc
c  moment time fctn fsource
c      ics=4                !triangle
c      t0=15.     ! duration
c      t1=0.5     ! no meaning
c      icc=1                ! output = velocity
cccccccccccccccccccccccccc
c  moment time fctn fsource
c     ics=9                !Brune
c     t0=5.0
c     t1=1.
c     icc=1                ! output = velocity
cccccccccccccccccccccccccc
c  moment time fctn fsource
c     ics=3                !  complex spectrum from a file
c     t0=6  ! no meaning
c     t1=1. ! no meaning
c     icc=1                ! output =  velocity
cccccccccccccccccccccccccc
      icc=icc-1

ccccccccccccccccccccccccccc
!       mm=12          !!!!fixed
      nt=2**mm
       if (nt.gt.ntp) then
   	write(0,*) 'check dimen. nt !!!'
	stop
       endif
      dfreq=1./tl
      pas=tl/float(nt)      
      aw=-pi*aw/tl            ! aw re-defined here
    
      do 5 jf=1,nfreq  ! loop over frequency
      freq=float(jf-1)/tl
      omega=cmplx(pi2*freq,aw)
      deriv=(ai*omega)**icc
      us=fsource(ics,omega,t0,t1,pas) * deriv * exp(-ai*omega*delay)

ccccccccc modification for 0 freq

c         if(jf.eq.1)  us=us*0.000   !! or 0.0001
ccccccccc

	 do ir=1,nr   ! loop over stations
	 read(12)(uxf(it),it=1,6)
	 read(12)(uyf(it),it=1,6)
	 read(12)(uzf(it),it=1,6)

            do it=1,6    !  moment tensor components
	    ux(jf,ir,it)=uxf(it) * us
	    uy(jf,ir,it)=uyf(it) * us
	    uz(jf,ir,it)=uzf(it) * us
            enddo

         enddo          ! end loop over stations
   5  continue          ! ending loop over frequency

c     goto 2001
c2000 nfreq=jf-1
c2001 continue


c++++++++++++
c                ELEMENTARY  SEISMO  CALCUL
c++++++++++++

      do 30 ir=1,nr  ! stations

      do it=1,6    ! moment tensor

      do ifr=nt+2-nfreq,nt   ! frequency
      ux(ifr,ir,it)=conjg(ux(nt+2-ifr,ir,it))
      uy(ifr,ir,it)=conjg(uy(nt+2-ifr,ir,it))
      uz(ifr,ir,it)=conjg(uz(nt+2-ifr,ir,it))
      enddo

      call fft2cd(ux(1,ir,it),mm,iwk)
      call fft2cd(uy(1,ir,it),mm,iwk)
      call fft2cd(uz(1,ir,it),mm,iwk)
      
   
      do itim=1,nt           ! time
	ck=float(itim-1)/nt
	cc=exp(-aw*tl*ck)/tl   !!! removing artific. absorption

!!!!!!!!!!		(factor 1./tl has nothing to do with it, must be always) 
!!!!!!!!!! in other words - if we experimentate with calcelling this operation
!!!!!!!!!!!!!!    we must at least divite u(...) by TL  
c	ux(itim,ir,it)=ux(itim,ir,it)/tl ! this is non-regular (then in filter_stat we do NOT need artif. atten)
c	uy(itim,ir,it)=uy(itim,ir,it)/tl
c	uz(itim,ir,it)=uz(itim,ir,it)/tl

	ux(itim,ir,it)=ux(itim,ir,it)*cc ! this is regular (then in filter_stat we need artif. atten)
	uy(itim,ir,it)=uy(itim,ir,it)*cc
	uz(itim,ir,it)=uz(itim,ir,it)*cc
      enddo

      enddo ! moment tensor

  30  continue          !stations

cccc    ELEMENTARY SEISMO  (it=1,2...6)  OUTPUT   !!!  always all 6 !!!

      dt=tl/float(nt)

      do ir=1,nr           ! stations
        do it=1,6            ! seismo for 6 basic moment tensors
          do itim=1,nt         ! time
          time=float(itim-1)*dt
          bux=real(ux(itim,ir,it))       
          buy=real(uy(itim,ir,it))
          buz=real(uz(itim,ir,it))
          write(300) time,bux,buy,buz
          enddo               ! time
        enddo                ! basic moment tensors
      enddo                ! stations



	stop
	end



ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c        SUBROUTINES
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

	subroutine  fft2cd (a,m,iwk)                                       
c                                  specifications for arguments         
	implicit real*8 (a-h,o-z)
	integer            m,iwk(*)                                       
	complex*16      a(*)
c                                  specifications for local variables   
	integer     i,isp,j,jj,jsp,k,k0,k1,k2,k3,kb,
     &              kn,mk,mm,mp,n,n4,n8,n2,lm,nn,jk 
	real*8      rad,c1,c2,c3,s1,s2,s3,ck,sk,sq,a0,a1,a2,a3,    
     &              b0,b1,b2,b3,twopi,temp,                        
     &              zero,one,z0(2),z1(2),z2(2),z3(2)               
	complex*16  za0,za1,za2,za3,ak2                            
	equivalence  (za0,z0(1)),(za1,z1(1)),(za2,z2(1)),           
     &              (za3,z3(1)),(a0,z0(1)),(b0,z0(2)),(a1,z1(1)),  
     &              (b1,z1(2)),(a2,z2(1)),(b2,z2(2)),(a3,z3(1)),   
     &              (b3,z3(2))                                     
	data        sq,sk,ck,twopi/.7071068,.3826834,
     &              .9238795,6.283185/                                
	data        zero/0.0/,one/1.0/                             
c                   sq=sqrt2/2,sk=sin(pi/8),ck=cos(pi/8) 
c                   twopi=2*pi                           
c                                  first executable statement           
	mp = m+1                                                          
	n = 2**m                                                          
	iwk(1) = 1                                                        
	mm = (m/2)*2                                                      
	kn = n+1                                                          
c                                  initialize work vector               
	do 5  i=2,mp                                                      
	   iwk(i) = iwk(i-1)+iwk(i-1)                                     
5       continue                                                          
	rad = twopi/n                                                     
	mk = m - 4                                                        
	kb = 1                                                            
	if (mm .eq. m) go to 15                                           
	k2 = kn                                                           
	k0 = iwk(mm+1) + kb                                               
10      k2 = k2 - 1                                                       
	k0 = k0 - 1                                                       
	ak2 = a(k2)                                                       
	a(k2) = a(k0) - ak2                                               
	a(k0) = a(k0) + ak2                                               
	if (k0 .gt. kb) go to 10                                          
15      c1 = one                                                          
	s1 = zero                                                         
	jj = 0                                                            
	k = mm - 1                                                        
	j = 4                                                             
	if (k .ge. 1) go to 30                                            
	go to 70                                                          
20      if (iwk(j) .gt. jj) go to 25                                      
	jj = jj - iwk(j)                                                  
	j = j-1                                                           
	if (iwk(j) .gt. jj) go to 25                                      
	jj = jj - iwk(j)                                                  
	j = j - 1                                                         
	k = k + 2                                                         
	go to 20                                                          
25      jj = iwk(j) + jj                                                  
	j = 4                                                             
30      isp = iwk(k)                                                      
	if (jj .eq. 0) go to 40                                           
c                                  reset trigonometric parameter(s       )
	c2 = jj * isp * rad                                               
	c1 = cos(c2)                                                      
	s1 = sin(c2)                                                      
35      c2 = c1 * c1 - s1 * s1                                            
	s2 = c1 * (s1 + s1)                                               
	c3 = c2 * c1 - s2 * s1                                            
	s3 = c2 * s1 + s2 * c1                                            
40      jsp = isp + kb                                                    
c                                  determine fourier coefficients       
c                                    in groups of 4                     
	do 50 i=1,isp                                                     
	   k0 = jsp - i                                                   
	   k1 = k0 + isp                                                  
	   k2 = k1 + isp                                                  
	   k3 = k2 + isp                                                  
	   za0 = a(k0)                                                    
	   za1 = a(k1)                                                    
	   za2 = a(k2)                                                    
	   za3 = a(k3)                                                    
	   if (s1 .eq. zero) go to 45                                     
	   temp = a1                                                      
	   a1 = a1 * c1 - b1 * s1                                         
	   b1 = temp * s1 + b1 * c1                                       
	   temp = a2                                                      
	   a2 = a2 * c2 - b2 * s2                                         
	   b2 = temp * s2 + b2 * c2                                       
	   temp = a3                                                      
	   a3 = a3 * c3 - b3 * s3                                         
	   b3 = temp * s3 + b3 * c3                                       
45         temp = a0 + a2                                                 
	   a2 = a0 - a2                                                   
	   a0 = temp                                                      
	   temp = a1 + a3                                                 
	   a3 = a1 - a3                                                   
	   a1 = temp                                                      
	   temp = b0 + b2                                                 
	   b2 = b0 - b2                                                   
	   b0 = temp                                                      
	   temp = b1 + b3                                                 
	   b3 = b1 - b3                                                   
	   b1 = temp                                                      
	   a(k0) = cmplx(a0+a1,b0+b1)                                     
	   a(k1) = cmplx(a0-a1,b0-b1)                                     
	   a(k2) = cmplx(a2-b3,b2+a3)                                     
	   a(k3) = cmplx(a2+b3,b2-a3)                                     
50      continue                                                          
	if (k .le. 1) go to 55                                            
	k = k - 2                                                         
	go to 30                                                          
55      kb = k3 + isp                                                     
c                                  check for completion of final        
c                                    iteration                          
	if (kn .le. kb) go to 70                                          
	if (j .ne. 1) go to 60                                            
	k = 3                                                             
	j = mk                                                            
	go to 20                                                          
60      j = j - 1                                                         
	c2 = c1                                                           
	if (j .ne. 2) go to 65                                            
	c1 = c1 * ck + s1 * sk                                            
	s1 = s1 * ck - c2 * sk                                            
	go to 35                                                          
65      c1 = (c1 - s1) * sq                                               
	s1 = (c2 + s1) * sq                                               
	go to 35                                                          
70      continue                                                          
c                                  permute the complex vector in        
c                                    reverse binary order to normal     
c                                    order                              
	if(m .le. 1) go to 9005                                           
	mp = m+1                                                          
	jj = 1                                                            
c                                  initialize work vector               
	iwk(1) = 1                                                        
	do 75  i = 2,mp                                                   
	   iwk(i) = iwk(i-1) * 2                                          
75      continue                                                          
	n4 = iwk(mp-2)                                                    
	if (m .gt. 2) n8 = iwk(mp-3)                                      
	n2 = iwk(mp-1)                                                    
	lm = n2                                                           
	nn = iwk(mp)+1                                                    
	mp = mp-4                                                         
c                                  determine indices and switch a       
	j = 2                                                             
80      jk = jj + n2                                                      
	ak2 = a(j)                                                        
	a(j) = a(jk)                                                      
	a(jk) = ak2                                                       
	j = j+1                                                           
	if (jj .gt. n4) go to 85                                          
	jj = jj + n4                                                      
	go to 105                                                         
85      jj = jj - n4                                                      
	if (jj .gt. n8) go to 90                                          
	jj = jj + n8                                                      
	go to 105                                                         
90      jj = jj - n8                                                      
	k = mp                                                            
95      if (iwk(k) .ge. jj) go to 100                                     
	jj = jj - iwk(k)                                                  
	k = k - 1                                                         
	go to 95                                                          
100     jj = iwk(k) + jj                                                  
105     if (jj .le. j) go to 110                                          
	k = nn - j                                                        
	jk = nn - jj                                                      
	ak2 = a(j)                                                        
	a(j) = a(jj)                                                      
	a(jj) = ak2                                                       
	ak2 = a(k)                                                        
	a(k) = a(jk)                                                      
	a(jk) = ak2                                                       
110     j = j + 1                                                         
c                                  cycle repeated until limiting number 
c                                    of changes is achieved             
	if (j .le. lm) go to 80                                           
c                                                                       
9005    return                                                            
	end                                                               
c @(#) fsource.F       FSOURCE 1.4      11/17/92 1
c**********************************************************
c       FSOURCE
c       
c       Different kind of source function defined in the 
c       frequency domain
c       Source function for dislocation (step, ramp, haskell)
c       are normalized so that in far-field the low-frequency 
c       level is proportionnal to the seismic moment with a factor
c       equals to: Rad/(4.PI.rho.beta^3) * 1./r 
c       where Rad= Radiation coefficient with (possibly) 
c       free surface effect
c
c       input:  
c               type    ->      see below
c               omega   ->      angular frequency
c               t0,t1   ->      time constant when needed
c               dt      ->      sampling rate
c**********************************************************

	function        fsource (type, omega, t0, t1, dt)

	implicit        none
	integer         type
	real            pi,pi2,dt,t0,t1
	real*8          uur,uui,trise,trupt
	complex*16      fsource,uu,uex,uxx,omega,shx,ai,omegac

      pi=3.1415926535
      pi2=2.*pi
      ai=(0.,1.)        

c TYPE=0               Source = Dirac en deplacement
	if (type.eq.0) then
	  fsource=1
	endif
 
c TYPE=1        Source = Ricker en deplacement
	if (type.eq.1) then
	  uu=omega*t0
	  uu=uu*uu/pi2/pi2
	  uu=exp(-uu)
	  uu=omega*omega*uu*dt
	  fsource= uu
	endif
 
c TYPE=2        Source = step en deplacement
c               2 steps possibles (1) real=1/(ai*omega)
c                                 (2) bouchon's
	if (type.eq.2) then
	  shx=exp(omega*pi*t0/2.)       !Bouchon's
	  shx=1./(shx-1./shx)
	  uu=-ai*t0*pi*shx
	  fsource= uu
	endif
c TYPE=7        Source = step en deplacement
	if (type.eq.7) then
	  uu=1./ai/omega
	  fsource= uu
	endif

c TYPE=3        Source = file
	if (type.eq.3) then
	  read(130,*) uur,uui
	  fsource=cmplx(uur,uui)
	endif
 
c TYPE=4        Source = triangle en deplacement
	if (type.eq.4) then
c	  t0=.5
	  uu=exp(ai*omega*t0/4.)
	  uu=(uu-1./uu)/2./ai
	  uu=uu/(omega*t0/2.)
	  fsource=uu*uu    *    4.   ! 4 was missing in original code
c      write(3578,*) real(fsource),imag(fsource), abs(fsource)
	endif
 
c TYPE=5        Source = rampe causale
c               rise time T=t0
	if (type.eq.5) then
 	  trise=t0
	  uu=ai*omega*trise
	  uu=(1.-exp(-uu))/uu
	  fsource=uu/(ai*omega)
	endif
	
	
c TYPE=9        Source = Brune
c               corner freq= 1 / t0
	if (type.eq.9) then
 	  trise=t0
          omegac=pi2*(1./trise)
          uu=omegac+ai*omega
	  fsource=omegac*omegac/(uu*uu)
	endif
	
c TYPE=6,8        Source = modele d'haskell, trapezoide
c       1 ere cste de temps rise time: riset
c       2 eme cste de temps, duree de la rupture
c         trupt = Length/2/rupt_velocity (Haskell)
	if ((type.eq.6).or.(type.eq.8)) then
	  trise=t0
	  trupt=t1
	  uu=ai*omega*trise
	  uu=(1.-exp(-uu))/uu           ! ramp
	  uxx=ai*omega*trupt/2.         ! finite fault
	  uex=exp(uxx)
	  uxx=(uex-1./uex)/uxx/2.
	  fsource=uu*uxx/(ai*omega)
	endif
    
	return
	end