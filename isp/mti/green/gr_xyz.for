      program gr_xyz
c
c
c Program modified from AXITRA of O. Coutant.
c J. Zahradnik, 1997.
c
c  
c
c
c******************************************************************************
c*                                                                            *
c*                      PROGRAMME AXITRA                                      *
c*                                                                            *
c*      Calcul de sismogrammes synthetiques en milieu stratifie a symetrie    *
c*      cylindrique.                                                          *
c*      Propagation par la methode de la reflectivite, avec coordonnees       *
c*      cylindriques (r, theta, z)                                            *
c*      Attenuation sur les ondes P et S                                      *
c*                                                                            *
c*      auteur : Coutant O. 
c*      Bibliographie :                                                       *
c*                      Kennett GJRAS vol57, pp557R, 1979                     *
c*                      Bouchon JGR vol71, n4, pp959, 1981                    *
c*                                                                            *
c******************************************************************************



c Global
      include   "param.inc"
      include   "dimen1.inc"
      
c Local
c     character*20 sourcefile,statfile
      integer      ic,ir,is
      real*8         dfreq,freq,pil
      logical      tconv(nrp,nsp)
      namelist    /input/ nc,nfreq,tl,aw,nr,ns,xl,
     &                    ikmax,uconv,fref
      data        ai,pi,pi2/(0.,1.),3.14159265359,6.28318530718/
      CHARACTER*255 infile, model

      model = ''
      if (command_argument_count() >= 5) then
          CALL getarg(5, model)
          if (trim(model) /= '') then
              model = '-'//model
          endif
      endif
      open (in1,form='formatted',file='grdat'//trim(model)//'.hed')
      open (110,form='formatted',file='crustal'//trim(model)//'.dat')
      CALL getarg(4,infile)
      open (out,form='formatted',file='gr'//trim(infile)//'.hea')
      open (out2,form='unformatted',file='gr'//trim(infile)//'.hes')
      rewind(out)
      rewind(out2)
c++++++++++
c           LECTURE DES PARAMETRES D'ENTREE
c              
c               sismogramme : nfreq,tl,xl
c               recepteurs  : nr,xr(),yr(),zr()
c               source      : xs,ys,zs
c               modele      : nc,hc(),vp(),vs(),rho()
c                            
c               si hc(1)=0 on donne les profondeurs des interfaces, sinon
c               on donne les epaisseurs des couches
c++++++++++

      read(in1,input)
      read(110,*)
      read(110,*)
      read(110,*) nc
      read(110,*)
      read(110,*)
      do ic=1,nc
	read(110,*) hc(ic),vp(ic),vs(ic),rho(ic),qp(ic),qs(ic)
         hc(ic)=hc(ic)*1000.
         vp(ic)=vp(ic)*1000.
         vs(ic)=vs(ic)*1000.
         rho(ic)=rho(ic)*1000.
      enddo
!       open (in2,form='formatted',file='source.dat')
      open (in3,form='formatted',file='station'//trim(model)//'.dat')
      
      write(out,input)
      write(out,*) 'hc,vp,vs,rho,Qp,Qs'
      do 3 ic=1,nc
 3    write(out,1001) hc(ic),vp(ic),vs(ic),rho(ic),qp(ic),qs(ic)

c               Test sur les dimensions

      if ((nr.gt.nrp).or.(nc.gt.ncp).or.(ns.gt.nsp)) then
      write(6,*) 'nombre de parametres superieur au dimensionnement'
      stop
      endif

c++++++++++
c           INITIALISATIONS
c++++++++++

      uconv=uconv*uconv
      dfreq=1./tl
      aw=-pi*aw/tl
      freq=-dfreq
      pil=pi2/xl
      iklast=0

      call initdata

c               ***************************
c               ***************************
c               **  BOUCLE EN FREQUENCE  **
c               ***************************
c               ***************************
      do 10 if=1,nfreq
c      if (if.eq.1) lastik=0  !!! new JZ may2005   

      freq=freq+dfreq
      rw=pi2*freq
      omega=cmplx(rw,aw)
      omega2=omega*omega
      a1=.5/omega2/xl
      zom=sqrt(rw*rw+aw*aw)
      if (if.eq.1) then
	phi=-pi/2
      else
	phi=atan(aw/rw)
      endif
	do ir=1,nr
	 do is=1,ns
	   tconv(ir,is)=.false.
	 enddo
	enddo
      ttconv=.false.
c      xlnf=(ai*phi+dlog(zom))/pi
      xlnf=(ai*phi+dlog(zom/(pi2*fref)))

c            ******************************************
c            ******************************************
c            **  RESOLUTION PAR BOUCLE EXTERNE EN Kr **
c            ******************************************
c            ******************************************

      do 20 ik=0,ikmax
      
      kr=(ik+.258)*pil
      kr2=kr*kr


c+++++++++++++
c              Calcul de nombreux coefficients et des fonctions de Bessel
c+++++++++++++

      call reflect0 (ik+1, iklast)

c+++++++++++++
c              Calcul des coefficients de reflexion/transmission
c              Matrice de Reflection/Transmission et Dephasage
c+++++++++++++

      call reflect1
      
c+++++++++++++
c              Calcul des matrices de reflectivite : mt(),mb(),nt(),nb()
c              (rapport des differents potentiels montant/descendant
c                        en haut et en bas de chaque couche)
c+++++++++++++

      call reflect2

c+++++++++++++
c              Calcul des matrices de passage des vecteurs potentiel 
c               source, aux vecteurs potentiel PHI, PSI et KHI au sommet
c               de chaque couche
c+++++++++++++
      call reflect3

c+++++++++++++
c              Calcul des potentiels et des deplacement dus aux sources du
c               tenseur, en chaque recepteur (termes en kr, r, z)
c+++++++++++++
      call reflect4 ((ik.gt.ikmin).and.(ik.gt.lastik),tconv)

      if (ttconv) goto 21
      
 20   continue
 21   continue

c+++++++++++++
c               Calcul des deplacements aux recepteurs 
c               Sortie des resultats
c+++++++++++++

      lastik=ik-1
      write(out,*) 'freq =',freq,'iter =',lastik
      if (if.eq.1) lastik=0

      call reflect5

      if (ik.ge.ikmax) then
      write(6,*) 'Depassement du nombre d iteration maximum'
      stop
      endif
      
 1001 format(8f12.3)	   ! f9.3 produced problems with depths>100km; JZ 27.8.2013 
!       write(*,*) if
 10   continue
      
      stop
      end
c @(#) ff0ad.F  AXITRA 4.12     12/7/93 4
C/     ADD NAME=FF01AD          HSL     F77     DOUBLE
C######DATE   01 JAN 1984     COPYRIGHT UKAEA, 1.
C######ALIAS FF01AD
      SUBROUTINE FF01AD(VJ0,VY0,XD,N)
C  STANDARD FORTRAN 66(A VERIFIED PFORT SUBROUTINE)
      DOUBLE PRECISION VJ0,VY0,X,Y,Z,Q1,Q2,Q3,FX,X1,X2,X3,
     1                 X4,XD,XLG,A,B,C,D,E
      DIMENSION A(73),B(18),C(19),D(18),E(18)
      EQUIVALENCE (A(1),B(1)),(A(19),C(1)),(A(38),D(1)),(A(56),E(1))
      DATA XLG /1.0D+70/
      DATA B(1),B(2),B(3),B(4),B(5),B(6),B(7),B(8),B(9),B(10),B(11),
     1     B(12),B(13),B(14),B(15),B(16),B(17),B(18)    /
     1   -.17D-18                  , .1222D-16             ,
     2   -.75885D-15               , .4125321D-13          ,
     3   -.194383469D-11           , .7848696314D-10       ,
     4   -.267925353056D-8         , .7608163592419D-7     ,
     5   -.176194690776215D-5      , .3246032882100508D-4  ,
     6   -.46062616620627505D-3    , .48191800694676045D-2 ,
     7   -.34893769411408885D-1    , .15806710233209726D0  ,
     8   -.37009499387264978D0     , .26517861320333681D0  ,
     9   -.87234423528522213D-2    , .31545594294978024D0  /
      DATA C(1),C(2),C(3),C(4),C(5),C(6),C(7),C(8),C(9),C(10),C(11),
     1     C(12),C(13),C(14),C(15),C(16),C(17),C(18),C(19)    /
     A   -.1D-19                   , .39D-18               ,
     B   -.2698D-16                , .164349D-14           ,
     C   -.8747341D-13             , .402633082D-11        ,
     D   -.15837552542D-9          , .524879478733D-8      ,
     E   -.14407233274019D-6       , .32065325376548D-5    ,
     F   -.5632079141056987D-4     , .75311359325777423D-3 ,
     G   -.72879624795520792D-2    , .47196689595763387D-1 ,
     H   -.17730201278114358D0     , .26156734625504664D0  ,
     I    .17903431407718266D0     ,-.27447430552974527D0  ,
     J   -.66292226406569883D-1     /
      DATA D(1),D(2),D(3),D(4),D(5),D(6),D(7),D(8),D(9),D(10),D(11),
     1     D(12),D(13),D(14),D(15),D(16),D(17),D(18)    /
     K   -.1D-19                   , .2D-19                ,
     L   -.11D-18                  , .55D-18               ,
     M   -.288D-17                 , .1631D-16             ,
     N   -.10012D-15               , .67481D-15            ,
     O   -.506903D-14              , .4326596D-13          ,
     O   -.43045789D-12            , .516826239D-11        ,
     P   -.7864091377D-10          , .163064646352D-8      ,
     Q   -.5170594537606D-7        , .307518478751947D-5   ,
     R   -.53652204681321174D-3    , .19989206986950373D1 /
      DATA E(1),E(2),E(3),E(4),E(5),E(6),E(7),E(8),E(9),E(10),E(11),
     1     E(12),E(13),E(14),E(15),E(16),E(17),E(18)   /
     S    .1D-19                   ,-.3D-19                ,
     T    .13D-18                  ,-.62D-18               ,
     U    .311D-17                 ,-.1669D-16             ,
     V    .9662D-16                ,-.60999D-15            ,
     W    .425523D-14              ,-.3336328D-13          ,
     X    .30061451D-12            ,-.320674742D-11        ,
     Y    .4220121905D-10          ,-.72719159369D-9       ,
     Z    .1797245724797D-7        ,-.74144984110606D-6    ,
     1    .683851994261165D-4      ,-.31111709210674018D-1 /
      X=XD
      Y=DABS(X)
      Z=Y*.125D0
      IF(Z .LE.1.0D0)GO TO 10
      Z=1.0D0/Z
      X2=4.0D0*Z*Z-2.0D0
      N1=38
      N2=55
      GO TO 70
   10 IF(Z .EQ. 0.0D0)GO TO  78
      X2=4.0D0*Z*Z-2.0D0
      N1=1
      N2=18
   70 DO 80 J=1,2
      Q3=0.0D0
      Q2=0.0D0
      DO 40  I=N1,N2
      Q1=Q2
      Q2=Q3
   40 Q3=X2*Q2-Q1+A(I)
      FX=(Q3-Q1)*.5D0
      IF(N1-19)50,51,52
   50 VJ0=FX
      IF(N .LE. 0)GO TO 75
      N1=19
      N2=37
      GO TO 80
   52 IF(N1.EQ.56)GO TO 53
      X1=FX
      N1=56
      N2=73
   80 CONTINUE
   78 VJ0=1.0D0
      VY0=-XLG
      GO TO 75
   51 VY0=.6366197723675813D0*DLOG(Y)*VJ0+FX
      GO TO 75
   53 X2=DCOS(Y-.7853981633974483D0)
      X3=DSIN(Y-.7853981633974483D0)
      X4=.7978845608028654D0/DSQRT(Y)
      FX=FX*Z
      VJ0=X4*(X1*X2-FX*X3)
      VY0=X4*(FX*X2+X1*X3)
   75 RETURN
      END
C/     ADD NAME=FF02AD          HSL     F77     DOUBLE
C######DATE   01 JAN 1984     COPYRIGHT UKAEA, 1.
C######ALIAS FF02AD
      SUBROUTINE FF02AD(VJ1,VY1,XD,N)
C  STANDARD FORTRAN 66(A VERIFIED PFORT SUBROUTINE)
      DOUBLE PRECISION VJ1,VY1,X,Y,Z,Q1,Q2,Q3,FX,X1,X2,X3,X4,
     1                 XD,XLG,A,B,C,D,E
      DIMENSION A(72),B(18),C(18),D(18),E(18)
      EQUIVALENCE (A(1),B(1)),(A(19),C(1)),(A(37),D(1)),(A(55),E(1))
      DATA XLG/1.0D+70 /
      DATA B(1),B(2),B(3),B(4),B(5),B(6),B(7),B(8),B(9),B(10),B(11),
     1     B(12),B(13),B(14),B(15),B(16),B(17),B(18)   /
     1   -.4D-19                   , .295D-17              ,
     2   -.19554D-15               , .1138572D-13          ,
     3   -.57774042D-12            , .2528123664D-10       ,
     4   -.94242129816D-9          , .2949707007278D-7     ,
     5   -.76175878054003D-6       , .1588701923993213D-4  ,
     6   -.26044438934858068D-3    , .32402701826838575D-2 ,
     7   -.29175524806154208D-1    , .17770911723972828D0  ,
     8   -.66144393413454325D0     , .12879940988576776D1  ,
     9   -.11918011605412169D1     , .12967175412105298D1  /
      DATA C(1),C(2),C(3),C(4),C(5),C(6),C(7),C(8),C(9),C(10),C(11),
     1     C(12),C(13),C(14),C(15),C(16),C(17),C(18)   /
     A    .9D-19                   ,-.658D-17              ,
     B    .42773D-15               ,-.2440949D-13          ,
     C    .121143321D-11           ,-.5172121473D-10       ,
     D    .187547032473D-8         ,-.5688440039919D-7     ,
     E    .141662436449235D-5      ,-.283046401495148D-4   ,
     F    .44047862986709951D-3    ,-.51316411610610848D-2 ,
     G    .42319180353336904D-1    ,-.22662499155675492D0  ,
     H    .67561578077218767D0     ,-.76729636288664594D0  ,
     I   -.12869738438135000D0     , .40608211771868508D-1 /
      DATA D(1),D(2),D(3),D(4),D(5),D(6),D(7),D(8),D(9),D(10),D(11),
     1     D(12),D(13),D(14),D(15),D(16),D(17),D(18)   /
     J    .1D-19                   ,-.2D-19                ,
     K    .12D-18                  ,-.58D-18               ,
     L    .305D-17                 ,-.1731D-16             ,
     M    .10668D-15               ,-.72212D-15            ,
     N    .545267D-14              ,-.4684224D-13          ,
     O    .46991955D-12            ,-.570486364D-11        ,
     P    .881689866D-10           ,-.187189074911D-8      ,
     Q    .6177633960644D-7        ,-.398728430048891D-5   ,
     R    .89898983308594085D-3    , .20018060817200274D1  /
      DATA E(1),E(2),E(3),E(4),E(5),E(6),E(7),E(8),E(9),E(10),E(11),
     1     E(12),E(13),E(14),E(15),E(16),E(17),E(18)   /
     S   -.1D-19                   , .3D-19                ,
     T   -.14D-18                  , .65D-18               ,
     U   -.328D-17                 , .1768D-16             ,
     V   -.10269D-15               , .65083D-15            ,
     W   -.456125D-14              , .3596777D-13          ,
     X   -.32643157D-12            , .351521879D-11        ,
     Y   -.4686363688D-10          , .82291933277D-9       ,
     Z   -.2095978138408D-7        , .91386152579555D-6    ,
     1   -.9627723549157079D-4     , .93555574139070650D-1 /
      X=XD
      Y=DABS(X)
      Z=Y*.125D0
      IF(Z.LE.1.0D0)GO TO 10
      Z=1.0D0/Z
      X2=4.0D0*Z*Z-2.0D0
      N1=37
      N2=54
      GO TO 70
   10 IF(Z .LE. 0.0D0)GO TO 78
      X2=4.0D0*Z*Z-2.0D0
      N1=1
      N2=18
   70 DO 80 J=1,2
      Q3=0.0D0
      Q2=0.0D0
      DO 40 I=N1,N2
      Q1=Q2
      Q2=Q3
      Q3=X2*Q2-Q1+A(I)
   40 CONTINUE
      FX=(Q3-Q1)*.5D0
      IF(N1-19)50,51,52
   50 VJ1=FX*Z
      IF(N.LE.0)GO TO 75
      N1=19
      N2=36
      GO TO 80
   52 IF(N1.EQ.55)GO TO 53
      X1=FX
      N1=55
      N2=72
   80 CONTINUE
   78 VJ1=0.0D0
      VY1=-XLG
      GO TO 75
   51 VY1=.6366197723675813D0*(DLOG(Y)*VJ1-1.0D0/Y)+FX*Z
      GO TO 75
   53 X2=DCOS(Y-2.356194490192345D0)
      X3=DSIN(Y-2.356194490192345D0)
      X4=.7978845608028654D0/DSQRT(Y)
      FX=FX*Z
      VJ1=X4*(X1*X2-FX*X3)
      VY1=X4*(FX*X2+X1*X3)
   75 RETURN
      END
c @(#) initdata.F       AXITRA 4.12     12/7/93 4
c******************************************************************************
c*                                                                            *
c*                     SUBROUTINE INITDATA                                    *
c*                                                                            *
c*    Initialisation de divers parametres                                     *
c*                                                                            *
c*    Input:
c*      hc,zr,zs,nc,nr
c*    Output:
c*      ncr,irc,nzr,irzz,nzrr,rr
c*      ncs,isc,nzs,iszz,nzss,rr,iss
c*    Modified:
c*      hc
c******************************************************************************

      subroutine initdata

c Global
      include "param.inc"
      include "dimen1.inc"
      include "dimen2.inc"
c Local
      integer   ir,ir1,ir2,ic,jr,jrr,js,jss,is,is1,is2,index(nsp)
      logical   tc
      real      hh,tmp,r(nrp,nsp)
      CHARACTER*255 dum1 , dum2 , dum3

c++++++++++++
c        Lecture coordonnees stations et recepteurs
c++++++++++++


c       do is=1,ns               ! new jz 
c       do ic=1,nc
c       nzss(is,ic)=0
c       nzrr(is,ic)=0
c       enddo
c       enddo

c      do is=1,ns               ! new JZ 
c      do ic=1,nc
c      izss(is,is,ic)=0
c      izrr(is,is,ic)=0
c      enddo
c      enddo



      do is=1,ns
c	read(in2,*) index(is),xs(is),ys(is),zs(is)
      if(is.gt.1) then
      write(*,*) 'more than one source not allowed in this version'
      stop
      endif
ccccccc
!         read(in2,*)
!         read(in2,*)
!         read(in2,*) xs(is),ys(is),zs(is)
      CALL getarg(1,dum1)
      CALL getarg(2,dum2)
      CALL getarg(3,dum3)
      read(dum1,*)xs(is)
      read(dum2,*)ys(is)
      read(dum3,*)zs(is)
      xs(is)=xs(is)*1000.
      ys(is)=ys(is)*1000.
      zs(is)=zs(is)*1000.
      index(1)=1
ccccccc
c xs(is)=xs(is)-xs(is)
c ys(is)=ys(is)-ys(is)
c zs(is)=zs(is)-zs(is)
      enddo
      read(in3,*)
      read(in3,*)
      do ir=1,nr
      read(in3,*) xr(ir),yr(ir),zr(ir)
      xr(ir)=xr(ir)*1000.
      yr(ir)=yr(ir)*1000.
      zr(ir)=zr(ir)*1000.
c xr(ir)=xr(ir)-xs(1)
c yr(ir)=yr(ir)-ys(1)
c zr(ir)=zr(ir)-zs(1)
      enddo

c++++++++++++
c        conversion interface -> epaisseur des couches     
c++++++++++++

      if (hc(1).eq.0.) then
      do ic=1,nc-1
      hc(ic)=hc(ic+1)-hc(ic)
      enddo
      endif

c++++++++++++
c        on reordonne les sources par profondeur croissante
c++++++++++++
      do is1=1,ns-1
      do is2=is1,ns
      if (zs(is1).gt.zs(is2)) then
      tmp=xs(is1)
      xs(is1)=xs(is2)
      xs(is2)=tmp
      tmp=ys(is1)
      ys(is1)=ys(is2)
      ys(is2)=tmp
      tmp=zs(is1)
      zs(is1)=zs(is2)
      zs(is2)=tmp
      tmp=index(is1)
      index(is1)=index(is2)
      index(is2)=tmp
      endif
      enddo
      enddo
      rewind (in2)
      do is=1,ns
c	write(in2,*) index(is),xs(is),ys(is),zs(is)
      enddo
      close(in2)

c++++++++++++
c       on calcule :
c       ncs: nombre de couches contenant un source
c       isc(): liste des couches contenant un source
c       nzs(i): nbre de sources de prof. differente dans couche i
c       nzss(j,i): nbre de sources a la prof j, dans la couche i
c       izss(,j,i): indice dans xr(),yr(),zr() des sources a la prof j
c                   dans la couche i
c++++++++++++
 
      do is=1,ns
c                       compute ic,zc
      ic=1    
      hh=hc(1)
      do while ((zs(is).gt.hh).and.(ic.lt.nc))
      zs(is)=zs(is)-hh
      ic=ic+1
      hh=hc(ic)
      enddo
       cff(is)=1./rho(ic)
c                       compute isc(),ncs,js
      if (is.eq.1) then
      isc(1)=ic
      ncs=1
      js=1
      else
      is1=1
      tc=.true.
      do while (is1.le.ncs)
      if (ic.eq.isc(is1)) then
      js=is1
      tc=.false.
      endif
      is1=is1+1
      enddo
      if (tc) then
      ncs=ncs+1
      isc(ncs)=ic
      js=ncs
      nzs(js)=0
      endif
      endif
c                       compute nzs(),jss
       if (is.eq.1) then
	nzs(1)=1
	jss=1
	tc=.false.
       else
	is2=1
	tc=.true.
	do while (is2.le.nzs(js))
	 if (zs(is).eq.zs(izss(1,is2,js))) then
	  jss=is2
	  tc=.false.
	 endif
	 is2=is2+1
	enddo
       endif
       if (tc) then
	nzs(js)=nzs(js)+1
	jss=nzs(js)
       endif
c                       compute nzss(,),izss(,,)
       nzss(jss,js)=nzss(jss,js)+1
       izss(nzss(jss,js),jss,js)=is
      enddo
 

c++++++++++++
c        on reordonne les stations par profondeur croissante
c++++++++++++
      do ir1=1,nr-1
       do ir2=ir1,nr
	if (zr(ir1).gt.zr(ir2)) then
	 tmp=xr(ir1)
	 xr(ir1)=xr(ir2)
	 xr(ir2)=tmp
	 tmp=yr(ir1)
	 yr(ir1)=yr(ir2)
	 yr(ir2)=tmp
	 tmp=zr(ir1)
	 zr(ir1)=zr(ir2)
	 zr(ir2)=tmp
	endif
       enddo
      enddo
 
      rewind(in3)
      do ir=1,nr
c	write(in3,*) xr(ir),yr(ir),zr(ir)
      enddo
      close(in3)

c++++++++++++
c       on calcule :
c       ncr: nombre de couches contenant un recepteur
c       irc(): liste des couches contenant un recept
c       nzr(i): nbre de recept. de prof. differente dans couche i
c       nzrr(j,i): nbre de recept a la prof j, dans la couche i
c       izrr(,j,i): indice dans xr(),yr(),zr() des recept a la prof j
c                   dans la couche i
c++++++++++++

      do ir=1,nr
c                       compute ic,zc
       ic=1     
       hh=hc(1)
       do while ((zr(ir).gt.hh).and.(ic.lt.nc))
	zr(ir)=zr(ir)-hh
	ic=ic+1
	hh=hc(ic)
       enddo
c                       compute irc(),ncr,jr
       if (ir.eq.1) then 
	irc(1)=ic
	ncr=1
	jr=1
       else
	ir1=1
	tc=.true.
	do while (ir1.le.ncr)
	 if (ic.eq.irc(ir1)) then
	  jr=ir1
	  tc=.false.
	 endif
	 ir1=ir1+1
	enddo
	if (tc) then
	 ncr=ncr+1
	 irc(ncr)=ic
	 jr=ncr
	 nzr(jr)=0
	endif
       endif
c                       compute nzr(),jrr
       if (ir.eq.1) then
	nzr(1)=1
	jrr=1
	tc=.false.
       else
	ir2=1
	tc=.true.
	do while (ir2.le.nzr(jr))
	 if (zr(ir).eq.zr(izrr(1,ir2,jr))) then
	  jrr=ir2
	  tc=.false.
	 endif
	 ir2=ir2+1
	enddo
       endif
       if (tc) then
	nzr(jr)=nzr(jr)+1
	jrr=nzr(jr)
       endif
c                       compute nzrr(,),izrr(,,)
       nzrr(jrr,jr)=nzrr(jrr,jr)+1
       izrr(nzrr(jrr,jr),jrr,jr)=ir
      enddo

c++++++++++++
c         distances radiales / source
c         on ne garde que les distances differentes, stockees dans 
c         rr(). tableau d'indirection irr().
c++++++++++++
      nrs=0             !calcule dist. rad.
      do is=1,ns
      do ir=1,nr
	 nrs=nrs+1
	 r(ir,is)=sqrt((xr(ir)-xs(is))*(xr(ir)-xs(is))+
     &                 (yr(ir)-ys(is))*(yr(ir)-ys(is)))
	 rr(nrs)=r(ir,is)
      enddo
      enddo
 
      ir1=1             !elimine dist. rad. egales
      do while (ir1.lt.nrs)
	ir2=ir1+1
	do while (ir2.le.nrs)
	if (rr(ir1).eq.rr(ir2)) then
	  rr(ir2)=rr(nrs)
	  nrs=nrs-1
	else
	  ir2=ir2+1
	endif
	enddo
	ir1=ir1+1
      enddo

c Tableau d'indirection
      do is=1,ns
       do ir=1,nr
	do ir2=1,nrs
	  if (r(ir,is).eq.rr(ir2)) irs(ir,is)=ir2
	enddo
       enddo
      enddo

c coef azimut.
      do is=1,ns
       do ir=1,nr
	if (r(ir,is).ne.0.) then
	 cosr(ir,is)=(xr(ir)-xs(is))/r(ir,is)
	 sinr(ir,is)=(yr(ir)-ys(is))/r(ir,is)
	else
	 cosr(ir,is)=1.
	 sinr(ir,is)=0.
	endif
       enddo
      enddo

      do 2 ic=1,nc
      vs2(ic)=vs(ic)*vs(ic)
      vp2(ic)=vp(ic)*vp(ic)
 2    continue

      return
      end
c @(#) reflect0.F       AXITRA 4.12     12/7/93 4
c******************************************************************************
c*                                                                            *
c*                     SUBROUTINE REFLECT0                                    *
c*                                                                            *
c*    Calcul de coefficients dependant du nombre d'onde kr et de fonctions    *
c*    de Bessel.                                                              *
c*                                                                            *
c******************************************************************************


      subroutine reflect0 (ik, iklast)
c Global
      include "param.inc"
      include "dimen1.inc"
      include "dimen2.inc"
c Local
c     integer ir,ier
      integer ir
      real*8    fj0,arg
      real*8    jj0(nkmax,nrsp),jj1(nkmax,nrsp)
      real*8    vy
      save        jj0,jj1
      dimension   cu(11*nrsp)
      equivalence (cu,u)

c     initialisations pour kr=0.


       if (kr.eq.0.) then
       do 5 i=1,11*nrp
       cu(i)=0.
   5   continue
cc    write(*,*) 'old init'  ! never
       endif

c       if (ik.eq.1) then       ! NEW jz 2005
c       do 5 i=1,11*nrp         
c       cu(i)=0.               
c   5   continue
cc    write(*,*) 'new init'  !  1x for each frequency
c       endif


c     Calcul des fonctions de Bessel J0 et J1, k1,k2,k3,k4,k5,k0

      do 10 ir=1,nrs
      arg=rr(ir)*kr
      if (ik.gt.nkmax) then
	call ff01ad(fj0,vy,arg,0)
	call ff02ad(fj1(ir),vy,arg,0)
      else
       if (ik.gt.iklast) then
	call ff01ad(jj0(ik,ir),vy,arg,0)
	call ff02ad(jj1(ik,ir),vy,arg,0)
       endif
       fj0=jj0(ik,ir)
       fj1(ir)=jj1(ik,ir)
      endif
       
      if (rr(ir).ne.0) then
       k0(ir)=kr*fj0
       k2(ir)=fj1(ir)/rr(ir)
       k1(ir)=k0(ir)-2.*k2(ir)
       k4(ir)=k1(ir)/rr(ir)
       k3(ir)=-(2.*k4(ir)+kr2*fj1(ir))
      else
c               Lorsque rr=0. il faut utiliser
c               un developpement limite
       k1(ir)=0.
       k2(ir)=kr/2.
       k3(ir)=0.
       k4(ir)=0.
      endif
      k5(ir)=k0(ir)-k2(ir)


 10   continue
      if(ik.gt.iklast) iklast=ik

c               Calcul des nombres d'onde verticaux

      do 11 ic=1,nc
c      ccv=1.+ai/(qp(ic)+qs(ic))
 
      ccvp=1.+ai/2./qp(ic)/(1.-xlnf/(qp(ic)*pi))
      ccvs=1.+ai/2./qs(ic)/(1.-xlnf/(qs(ic)*pi))

c      ccvp=1.+ai/(qp(ic)+qs(ic))/(1.-xlnf/(qp(ic)*pi))
c      ccvs=1.+ai/(qs(ic)+qp(ic))/(1.-xlnf/(qs(ic)*pi))
c      ccvp=1.+ai/2./qp(ic)
c      ccvs=1.+ai/2./qs(ic)
 
      cvp=vp(ic)*ccvp/(1.-xlnf/(qp(ic)*pi))/
     *    (1.+.25/(qp(ic)*qp(ic)*(1.-xlnf/(qp(ic)*pi))**2.))
      
      cvs=vs(ic)*ccvs/(1.-xlnf/(qs(ic)*pi))/
     *    (1.+.25/(qs(ic)*qs(ic)*(1.-xlnf/(qs(ic)*pi))**2.))
      
      cka(ic)=omega/cvp
      ckb(ic)=omega/cvs
      ckb2(ic)=ckb(ic)*ckb(ic)
      cka2(ic)=cka(ic)*cka(ic)
      cc=cka2(ic)-kr2
      cnu(ic)=cdsqrt(cc)
      if (dimag(cnu(ic)).gt.0.d0) cnu(ic)=-cnu(ic)
      cc=ckb2(ic)-kr2
      cgam(ic)=cdsqrt(cc)
      if (dimag(cgam(ic)).gt.0.d0) cgam(ic)=-cgam(ic)
 11   continue
      do 12 ic=1,nc
      c2(ic)=kr*kr/ai/cnu(ic)
 12   continue
      
      return
      end
c @(#) reflect1.F       AXITRA 4.13     12/7/93 4
c******************************************************************************
c                                                                             *
c                         SUBROUTINE REFLECT1                                 *
c                                                                             *
c               Calcul des coefficients de reflexion/transmission             *
c                  Matrice de Reflexion/Transmission et Dephasage             *
c       (Les coefficients de reflexion/transmission utilisent les memes       *
c            termes intermediaires que Aki-Richards p149, MAIS :              *
c            Aki utilise la convention inverse pour la TF (exp(-iwt)),        *
c        et travaille avec le parametre de rai et les angles d'incidences)    *
c                                                                             *
c      Le potentiel PSI utilise pour l'onde SV est defini par :               *
c                  u = rot ( rot (PSI) )                                      *
c      i.e. un terme de derivation supplementaire par rapport a la convention *
c      habituelle : u= rot (PSI)                                              *
c                                                                             *
c   On deduit les coefficients de REF/TRANS de ceux definis par la convention *
c      classique en divisant le potentiel PSI par 1./ai/kr = coef             *
c                                                                             *
c       Ordre de stockage :                                                   *
c                              pp=(1,1)   sp=(1,2)                            *
c                              ps=(2,1)   ss=(2,2)                            *
c******************************************************************************


      subroutine reflect1

      include "param.inc" 
      include "dimen1.inc"
      include "dimen2.inc"

c Coefficient pour la convention sur PSI (coef) et sur la TF (aki=-1.)
      coef=1./ai
      aki=-1.
      
c               CONDITIONS AUX LIMITES a la profondeur z=0, coefficients
c               de reflexion/transmission
c      2 possibilites : 1) surface libre (mettre en commentaire de B1 a B2)
c                       2) 1/2 espace sup. infini (commentaires de A1 a A2)

cA1                    SURFACE LIBRE 
      cf1=(ckb2(1)-2.*kr2)
      cf2=cf1*cf1
      cf3=4.*cnu(1)*kr2*cgam(1)
      cdd=1./(cf2+cf3)

      ru(1,1,1)=(-cf2+cf3)*cdd
      ru(1,2,1)=4.*cnu(1)*cf1*cdd*coef*aki
      ru(1,2,2)=(cf2-cf3)*cdd*aki
      ru(1,1,2)=4.*kr2*cgam(1)*cf1*cdd*ai
      tu(1,1,1)=0.
      tu(1,1,2)=0.
      tu(1,2,1)=0.
      tu(1,2,2)=0.
      rush(1)=1.
      tush(1)=0.
cA2

cB1                   1/2 ESPACE SUP. INFINI  
c     ru(1,1,1)=0.
c     ru(1,2,1)=0.
c     ru(1,2,2)=0.
c     ru(1,1,2)=0.
c     tu(1,1,1)=1.
c     tu(1,1,2)=0.
c     tu(1,2,1)=0.
c     tu(1,2,2)=1.
c     rush(1)=0.
c     tush(1)=1.
cB2

c               Coefficients aux interfaces entre couches
      cnurho = cnu(1)*rho(1)
      cgarho = cgam(1)*rho(1)
      cnugam = cnu(1)*cgam(1)
      cb2 = kr2/ckb2(1)
      ckb2i2 = 1./ckb2(1)

      do 24 ic=2,nc

      ic1=ic-1
      ckb2i1 = ckb2i2
      ckb2i2 = 1./ckb2(ic)
      cb1 = cb2
      cb2 = kr2*ckb2i2
      ca1d= rho(ic1)*(1.-2.*cb1)
      ca2d= rho(ic)*(1.-2.*cb2)
      ca  = ca2d-ca1d
      cb  = ca2d+2.*rho(ic1)*cb1
      cc  = ca1d+2.*rho(ic)*cb2
      cd  = 2.*(rho(ic)*ckb2i2-rho(ic1)*ckb2i1)
      ce  = cb*cnu(ic1)+cc*cnu(ic)
      cf  = cb*cgam(ic1)+cc*cgam(ic)
      cg  = ca-cd*cnu(ic1)*cgam(ic)
      cgkr2=cg*kr2
      ch  = ca-cd*cnu(ic)*cgam(ic1)
      chkr2=ch*kr2
      cdd = 1./(ce*cf+cg*chkr2)
      cdd2= 2.*cdd

      ctmp2 = cnurho*cdd2
      ctmp3 = cgarho*cdd2
      ctmp4 = (ca*cc+cb*cd*cnugam)*cdd2
      cnurho = cnu(ic)*rho(ic)
      cgarho = cgam(ic)*rho(ic)
      cnugam = cnu(ic)*cgam(ic)
      ctmp1 = (ca*cb+cc*cd*cnugam)*cdd2
      ctmp5 = cnurho*cdd2
      ctmp6 = cgarho*cdd2
      ctmp7 = cf*(cb*cnu(ic1)-cc*cnu(ic))
      ctmp8 = chkr2*(ca+cd*cnu(ic1)*cgam(ic))
      ctmp9 = ce*(cb*cgam(ic1)-cc*cgam(ic))
      ctmp10= cgkr2*(ca+cd*cnu(ic)*cgam(ic1))
      

      rd(ic,1,1)= (ctmp7 - ctmp8 )*cdd
      rd(ic,1,2)=-kr2*cgam(ic1)*ctmp1*ai*aki
      rd(ic,2,2)=-(ctmp9 - ctmp10)*cdd*aki
      rd(ic,2,1)=-cnu(ic1)*ctmp1*coef
      td(ic,1,1)= ctmp2*cf
      td(ic,1,2)=-ctmp3*cgkr2*ai*aki
      td(ic,2,2)= ctmp3*ce
      td(ic,2,1)= ctmp2*ch*coef*aki

      ru(ic,1,1)=-(ctmp7 + ctmp10)*cdd
      ru(ic,1,2)= kr2*cgam(ic)*ctmp4*ai
      ru(ic,2,2)= (ctmp9 + ctmp8)*cdd*aki
      ru(ic,2,1)= cnu(ic)*ctmp4*coef*aki
      tu(ic,1,1)= ctmp5*cf
      tu(ic,1,2)= ctmp6*chkr2*ai
      tu(ic,2,2)= ctmp6*ce
      tu(ic,2,1)=-ctmp5*cg*coef

c   Modification pour calculateur a faible dynamique [1.e-300; 1.e+300]
      cdeph=exp(-ai*cnu(ic1)*hc(ic1))
      rdeph=dreal(cdeph)
      adeph=dimag(cdeph)
      if (dabs(rdeph).lt.1.D-150) rdeph=0.
      me1(ic1)=cmplx(rdeph,adeph)
      
      cdeph=exp(-ai*cgam(ic1)*hc(ic1))
      rdeph=dreal(cdeph)
      adeph=dimag(cdeph)
      if (dabs(rdeph).lt.1.D-150) rdeph=0.
      me2(ic1)=cmplx(rdeph,adeph)

      cs1=rho(ic1)*ckb2i1*cgam(ic1)
      cs2=rho(ic)*ckb2i2*cgam(ic)
      cdelt=1./(cs1+cs2)

      rush(ic)=(cs2-cs1)*cdelt
      rdsh(ic)=-rush(ic)
      tush(ic)=2.*cs2*cdelt
      tdsh(ic)=2.*cs1*cdelt

 24   continue

      return
      end
c @(#) reflect2.F       AXITRA 4.12     12/7/93 4
c*******************************************************************************
c*                                                                             *
c*                         SUBROUTINE REFLECT2                                 *
c*                                                                             *
c*       Calcul des matrices de reflectivite mt,mb ou nt,nb dans chaque couche *
c*         (rapport des potentiels montant/descendant ou descendant/montant)   *
c*       Le suffixe t ou b precise si la matrice est donnee au sommet (top)    *
c*       ou au bas (bottom) de la couche.                                      *
c*       fup et fdo sont des matrices intermediaires utilisees dans le calcul  *
c*       des potentiels.                                                       *
c*       Ordre de stockage :                                                   *
c*                    pp=(1,1)   sp=(1,2)                                      *
c*                    ps=(2,1)   ss=(2,2)                                      *
c*******************************************************************************

      subroutine reflect2

      include "param.inc"
      include "dimen1.inc"
      include "dimen2.inc"

      complex*16    nb(2,2),mb(2,2),nbsh,mbsh
      integer   ic,ic1

c
c                       Calcul pour les couches au dessus de la source
c

      nt(1,1,1)=ru(1,1,1)
      nt(1,1,2)=ru(1,1,2)
      nt(1,2,1)=ru(1,2,1)
      nt(1,2,2)=ru(1,2,2)
      ntsh(1)=rush(1)
   
      do 10 ic=1,nc-1

      ic1=ic+1
      nb(1,1)=me1(ic)*me1(ic)*nt(ic,1,1)
      nb(1,2)=me1(ic)*me2(ic)*nt(ic,1,2)
      nb(2,1)=me2(ic)*me1(ic)*nt(ic,2,1)
      nb(2,2)=me2(ic)*me2(ic)*nt(ic,2,2)
      nbsh=me2(ic)*me2(ic)*ntsh(ic)
      
      ca1=1.-(rd(ic1,1,1)*nb(1,1)+rd(ic1,1,2)*nb(2,1))
      ca2=-(rd(ic1,1,1)*nb(1,2)+rd(ic1,1,2)*nb(2,2))
      ca3=-(rd(ic1,2,1)*nb(1,1)+rd(ic1,2,2)*nb(2,1))
      ca4=1.-(rd(ic1,2,1)*nb(1,2)+rd(ic1,2,2)*nb(2,2))
      cadet=ca1*ca4-ca2*ca3
      cash=1./(1.-rdsh(ic1)*nbsh)

      cb1=td(ic1,1,1)*nb(1,1)+td(ic1,1,2)*nb(2,1)
      cb2=td(ic1,1,1)*nb(1,2)+td(ic1,1,2)*nb(2,2)
      cb3=td(ic1,2,1)*nb(1,1)+td(ic1,2,2)*nb(2,1)
      cb4=td(ic1,2,1)*nb(1,2)+td(ic1,2,2)*nb(2,2)
      cbsh=tdsh(ic1)*nbsh

      cc1=(ca4*tu(ic1,1,1)-ca2*tu(ic1,2,1))/cadet
      cc2=(ca4*tu(ic1,1,2)-ca2*tu(ic1,2,2))/cadet
      cc3=(-ca3*tu(ic1,1,1)+ca1*tu(ic1,2,1))/cadet 
      cc4=(-ca3*tu(ic1,1,2)+ca1*tu(ic1,2,2))/cadet
      ccsh=cash*tush(ic1)

      nt(ic1,1,1)=ru(ic1,1,1)+cb1*cc1+cb2*cc3
      nt(ic1,1,2)=ru(ic1,1,2)+cb1*cc2+cb2*cc4
      nt(ic1,2,1)=ru(ic1,2,1)+cb3*cc1+cb4*cc3
      nt(ic1,2,2)=ru(ic1,2,2)+cb3*cc2+cb4*cc4
      ntsh(ic1)=rush(ic1)+cbsh*ccsh
 
      fup(ic,1,1)=cc1*me1(ic)
      fup(ic,1,2)=cc2*me1(ic)
      fup(ic,2,1)=cc3*me2(ic)
      fup(ic,2,2)=cc4*me2(ic)
      fupsh(ic)=ccsh*me2(ic)
  
 10   continue
c
c                       Calcul pour les couches au dessous de la source
c
      
      mt(nc,1,1)=0.
      mt(nc,1,2)=0.
      mt(nc,2,1)=0.
      mt(nc,2,2)=0.
      mtsh(nc)=0.

      do 20 ic=nc-1,1,-1

      ic1=ic+1
      ca1=1.-(ru(ic1,1,1)*mt(ic1,1,1)+ru(ic1,1,2)*mt(ic1,2,1))
      ca2=-(ru(ic1,1,1)*mt(ic1,1,2)+ru(ic1,1,2)*mt(ic1,2,2))
      ca3=-(ru(ic1,2,1)*mt(ic1,1,1)+ru(ic1,2,2)*mt(ic1,2,1))
      ca4=1.-(ru(ic1,2,1)*mt(ic1,1,2)+ru(ic1,2,2)*mt(ic1,2,2))
      cadet=ca1*ca4-ca2*ca3
      cash=1./(1.-rush(ic1)*mtsh(ic1))
 
      cb1=tu(ic1,1,1)*mt(ic1,1,1)+tu(ic1,1,2)*mt(ic1,2,1)
      cb2=tu(ic1,1,1)*mt(ic1,1,2)+tu(ic1,1,2)*mt(ic1,2,2)
      cb3=tu(ic1,2,1)*mt(ic1,1,1)+tu(ic1,2,2)*mt(ic1,2,1)
      cb4=tu(ic1,2,1)*mt(ic1,1,2)+tu(ic1,2,2)*mt(ic1,2,2)
      cbsh=tush(ic1)*mtsh(ic1)

      cc1=(ca4*td(ic1,1,1)-ca2*td(ic1,2,1))/cadet
      cc2=(ca4*td(ic1,1,2)-ca2*td(ic1,2,2))/cadet
      cc3=(-ca3*td(ic1,1,1)+ca1*td(ic1,2,1))/cadet 
      cc4=(-ca3*td(ic1,1,2)+ca1*td(ic1,2,2))/cadet
      ccsh=cash*tdsh(ic1)

      mb(1,1)=rd(ic1,1,1)+cb1*cc1+cb2*cc3
      mb(1,2)=rd(ic1,1,2)+cb1*cc2+cb2*cc4
      mb(2,1)=rd(ic1,2,1)+cb3*cc1+cb4*cc3
      mb(2,2)=rd(ic1,2,2)+cb3*cc2+cb4*cc4
      mbsh=rdsh(ic1)+cbsh*ccsh

      mt(ic,1,1)=me1(ic)*me1(ic)*mb(1,1)
      mt(ic,1,2)=me1(ic)*me2(ic)*mb(1,2)
      mt(ic,2,1)=me2(ic)*me1(ic)*mb(2,1)
      mt(ic,2,2)=me2(ic)*me2(ic)*mb(2,2)
      mtsh(ic)=me2(ic)*me2(ic)*mbsh

      fdo(ic1,1,1)=cc1*me1(ic)
      fdo(ic1,1,2)=cc2*me2(ic)
      fdo(ic1,2,1)=cc3*me1(ic)
      fdo(ic1,2,2)=cc4*me2(ic)
      fdosh(ic1)=ccsh*me2(ic)

 20   continue

      return
      end
c @(#) reflect3.F       AXITRA 4.12     12/7/93 4
c**************************************************************
c                                                             *
c                  SUBROUTINE REFLECT3                        *
c                                                             *
c  Calcul des potentiels dus a 6 sources elementaires, au     *
c  sommet de la couche source ISC. Les 6 sources elementaires *
c  sont 6 sources de potentiels, 2 PHI, 2 PSI, 2 KHI. Ces     *
c  sources different par l'existence d'un terme sign(z-z0).   *
c**************************************************************

      subroutine reflect3

      include "param.inc"
      include "dimen1.inc"
      include "dimen2.inc"

      integer     is0
      complex*16     rud1,rud2,rud3,rud4,rdu1,rdu2,rdu3,rdu4,tud1,
     &            tud2,tud3,tud4,tdu1,tdu2,tdu3,tdu4,tsh,egam,
     &            enu,rup,rdo,rupsh,rdosh,arg

      dimension cu1(2),cd1(2),cu2(2),cd2(2),cu3(2),cd3(2),
     &          cu4(2),cd4(2),rup(2,2),rdo(2,2)

c++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
c
c  Matrice de passage des vecteurs potentiels sources su0, sd0 
c  aux vecteurs potentiel de la couche source isc: su, sd
c
c                    [tud] et [tdu]
c
c                 ------------------------
c
c  su(,) : potentiel montant au sommet de la couche
c  sd(,) : potentiel descendant au sommet de la couche
c
c     (*,) : type de source (1 a 5)
c     (,*) : type de potentiel PHI ou PSI=KHI (1, 2)
c
c                 ------------------------
c  Les vecteurs potentiels su() et sd() sont obtenus a 
c  partir des potentiels des sources su0(), sd0() au 
c  sommet de la couche source par :
c
c     su = 1/[1 - rup*rdo] . (su0 + [rup].sd0)
c
c     sd = 1/[1 - rdo*rup] . (sd0 + [rdo].su0)
c
c
c     ou les matrices rup et rdo sont donnees par les
c matrices reflectivite du sommet de la couche source isc :
c   
c                [rup] = [mt(isc)]
c                [rdo] = [nt(isc)]
c
c       [rdo] = matrice reflectivite DOWN
c           (potentiel descendant/potentiel montant) due      
c     a l'empilement de couches situe au dessus de la source                
c
c       [rup] = matrice reflectivite UP
c           (potentiel montant/potentiel descendant) due      
c     a l'empilement de couches situe au dessous de la source               
c
c       [rud] = [rup] * [rdo]  
c
c       [rdu] = [rdo] * [rup] 
c                      
c     On pose [tud] = 1/[1 - rup*rdo]
c             [tdu] = 1/[1 - rdo*rup]
c++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      do is1=1,ncs      !boucle sur les couches sources
      ic=isc(is1)

      rdo(1,1)=nt(ic,1,1)
      rdo(1,2)=nt(ic,1,2)
      rdo(2,1)=nt(ic,2,1)
      rdo(2,2)=nt(ic,2,2)
      rdosh=ntsh(ic)
      rup(1,1)=mt(ic,1,1)
      rup(1,2)=mt(ic,1,2)
      rup(2,1)=mt(ic,2,1)
      rup(2,2)=mt(ic,2,2)
      rupsh=mtsh(ic)

      rud1=rup(1,1)*rdo(1,1)+rup(1,2)*rdo(2,1)
      rud2=rup(1,1)*rdo(1,2)+rup(1,2)*rdo(2,2)
      rud3=rup(2,1)*rdo(1,1)+rup(2,2)*rdo(2,1)
      rud4=rup(2,1)*rdo(1,2)+rup(2,2)*rdo(2,2)

      rdu1=rdo(1,1)*rup(1,1)+rdo(1,2)*rup(2,1)
      rdu2=rdo(1,1)*rup(1,2)+rdo(1,2)*rup(2,2)
      rdu3=rdo(2,1)*rup(1,1)+rdo(2,2)*rup(2,1)
      rdu4=rdo(2,1)*rup(1,2)+rdo(2,2)*rup(2,2)
 
      cdet=(1.-rud1)*(1.-rud4)-rud2*rud3
	
      tud1=(1.-rud4)/cdet
      tud2=rud2/cdet
      tud3=rud3/cdet
      tud4=(1.-rud1)/cdet
      
      cdet=(1.-rdu1)*(1.-rdu4)-rdu2*rdu3
      
      tdu1=(1.-rdu4)/cdet
      tdu2=rdu2/cdet
      tdu3=rdu3/cdet
      tdu4=(1.-rdu1)/cdet

      tsh=1./(1.-rupsh*rdosh)
      
c+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
c
c  Vecteurs potentiel source pour 4 sources elementaires :
c       (dephasage calcule / sommet de la couche)
c
c              cui = su0 + [rup].sd0  (i=1,4)
c              cdi = sd0 + [rdo].su0 
c
c  et potentiel KHI couche source pour 2 sources elementaires :
c
c                      cuish = su0sh + rupsh*sd0sh (i=1,2)
c                      cdish = sd0sh + rdosh*su0sh
c+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      do is2=1,nzs(is1)         !boucle sur les prof sources
      zsc=zs(izss(1,is2,is1))
      is0=izss(1,is2,is1)
      arg=-ai*cgam(ic)*zsc
      if (real(arg).lt.explim) arg=cmplx(explim,dimag(arg))
      egam=cdexp(arg)
      arg=-ai*cnu(ic)*zsc
      if (real(arg).lt.explim) arg=cmplx(explim,dimag(arg))
      enu=cdexp(arg)

c                       Source PHI
      cu1(1)=     enu + rup(1,1)/enu
      cu1(2)=           rup(2,1)/enu
      cd1(1)=  1./enu + rdo(1,1)*enu
      cd1(2)=           rdo(2,1)*enu
c                       Source PHI*sign(z-z0)
      cu2(1)=    -enu + rup(1,1)/enu
      cu2(2)=           rup(2,1)/enu
      cd2(1)=  1./enu - rdo(1,1)*enu
      cd2(2)=         - rdo(2,1)*enu
c                       Source PSI
      cu3(1)=           rup(1,2)/egam
      cu3(2)=    egam + rup(2,2)/egam
      cd3(1)=           rdo(1,2)*egam
      cd3(2)= 1./egam + rdo(2,2)*egam
c                       Source PSI*sign(z-z0)
      cu4(1)=           rup(1,2)/egam
      cu4(2)=   -egam + rup(2,2)/egam
      cd4(1)=         - rdo(1,2)*egam
      cd4(2)= 1./egam - rdo(2,2)*egam
c                       Source KHI
      cu1sh=     egam + rupsh/egam
      cd1sh=  1./egam + rdosh*egam
c                       Source KHI*sign(z-z0)
      cu2sh=    -egam + rupsh/egam
      cd2sh=  1./egam - rdosh*egam

c+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
c
c   Potentiels PHI, PSI et KHI, montant et descendant, dans la couche
c   source (dephasage / sommet) pour les 6 sources elementaires.
c
c+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

c                       Source PHI
      su1(is0,1)=tud1*cu1(1)+tud2*cu1(2)
      su1(is0,2)=tud3*cu1(1)+tud4*cu1(2)
      sd1(is0,1)=tdu1*cd1(1)+tdu2*cd1(2)
      sd1(is0,2)=tdu3*cd1(1)+tdu4*cd1(2)
c                       Source PHI*sign(z-z0)
      su2(is0,1)=tud1*cu2(1)+tud2*cu2(2)
      su2(is0,2)=tud3*cu2(1)+tud4*cu2(2)
      sd2(is0,1)=tdu1*cd2(1)+tdu2*cd2(2)
      sd2(is0,2)=tdu3*cd2(1)+tdu4*cd2(2)
c                       Source PSI
      su3(is0,1)=tud1*cu3(1)+tud2*cu3(2)
      su3(is0,2)=tud3*cu3(1)+tud4*cu3(2)
      sd3(is0,1)=tdu1*cd3(1)+tdu2*cd3(2)
      sd3(is0,2)=tdu3*cd3(1)+tdu4*cd3(2)
c                       Source PSI*sign(z-z0)
      su4(is0,1)=tud1*cu4(1)+tud2*cu4(2)
      su4(is0,2)=tud3*cu4(1)+tud4*cu4(2)
      sd4(is0,1)=tdu1*cd4(1)+tdu2*cd4(2)
      sd4(is0,2)=tdu3*cd4(1)+tdu4*cd4(2)
c                       Source KHI
      su1sh(is0)=tsh*cu1sh
      sd1sh(is0)=tsh*cd1sh
c                       Source KHI*sign(z-z0)
      su2sh(is0)=tsh*cu2sh
      sd2sh(is0)=tsh*cd2sh

      enddo
      enddo
      return
      end
c @(#) reflect4.F       AXITRA 4.12     12/7/93 4
c**************************************************************************
c*                                                                        *
c*                     SUBROUTINE REFLECT4                                *
c*                                                                        *
c*  Calcul des potentiels et des deplacements a chaque recepteur.         *
c*  - Matrice de passage des potentiels de la couche source aux couches   *
c*    recepteur (FTUP et FTDO)                                            *
c*  - Calcul des potentiels dans toutes les couches (PU et PD)            *
c*  - Calcul de 11 deplacements (termes intermediaires) a chaque recepteur*
c*    (U)                                                                 *
c*                                                                        *
c**************************************************************************

      subroutine reflect4 (tmin,tconv)

c Global
      include "param.inc"
      include "dimen1.inc"
      include "dimen2.inc"
c Local
      integer      ic,ir,ir1,ir2,ir3,idel,is,is1,is2,is3,jrs
      logical      tmin,abovecomp,belowcomp,tconv(nrp,nsp)

      complex*16     egam,enu,s1phiu,s1phid,pu,pd,push,pdsh,
     1            s1psiu,s1psid,s2phiu,s2phid,s2psiu,s2psid,s3phiu,
     2            s3phid,s3psiu,s3psid,s4phid,s4phiu,s4psid,s4psiu,
     3            s5,s6,ftup,ftdo,ftupsh,ftdosh,arg,enuinv,egaminv,
     4            cdu(nrp,nsp,11)
      real*8      zc,dz,r1,i1,r2,i2
     
      dimension   pu(2,4),pd(2,4),push(2),pdsh(2),ftdosh(ncp),
     &            ftup(ncp,2,2),ftdo(ncp,2,2),ftupsh(ncp)
c      
c++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
c
c  Matrices de passage des vecteurs potentiels dans la 
c  couche source aux vecteurs potentiel dans chaque couche
c
c                    [ftup] et [ftdo]
c       
c               ------------------------
c
c  Les vecteurs potentiels pu() et pd() sont obtenus a
c  partir des vecteurs potentiels su() et sd() dans la 
c  couche source par :
c
c  Couche (n) au dessus de la couche source :
c
c   pu(n) = [fup(n)]*[fup(n+1)]* *[fup(ics-1] . su
c
c   d'ou l'on tire pd(n) par  pd(n) = [nt(n)] . pu(n)
c
c  Couche (m) au dessous de la couche source :
c
c   pd(m) = [fdo(m)]*[fdo(m-1)]* *[fdo(ics+1)] . sd
c
c   d'ou l'on tire pu(m) par  pu(m) = [mt(m)] . pd(m)
c
c                -------------------------
c   On pose :
c
c        [ftup(n)] = [fup(n)]*...*[fup(ics-1)]*[tud]
c
c        [ftdo(m)] = [fdo(m)]*...*[fdo(ics+1)]*[tdu]
c
c++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

      do is1=1,ncs                      ! boucle sur les couches sources
	ics=isc(is1)

c               Couches au dessus de la couche source
	ftup(ics,1,1)=1.
	ftup(ics,1,2)=0.
	ftup(ics,2,1)=0.
	ftup(ics,2,2)=1.
	ftupsh(ics)=1.

	do ic=ics-1,1,-1
	 ftup(ic,1,1)=fup(ic,1,1)*ftup(ic+1,1,1)+
     &                fup(ic,1,2)*ftup(ic+1,2,1)
	 ftup(ic,1,2)=fup(ic,1,1)*ftup(ic+1,1,2)+
     &                fup(ic,1,2)*ftup(ic+1,2,2)
	 ftup(ic,2,1)=fup(ic,2,1)*ftup(ic+1,1,1)+
     &                fup(ic,2,2)*ftup(ic+1,2,1)
	 ftup(ic,2,2)=fup(ic,2,1)*ftup(ic+1,1,2)+
     &                fup(ic,2,2)*ftup(ic+1,2,2)
	 ftupsh(ic)=fupsh(ic)*ftupsh(ic+1)
	enddo

c               Couches au dessous de la couche source
	ftdo(ics,1,1)=1.
	ftdo(ics,1,2)=0.
	ftdo(ics,2,1)=0.
	ftdo(ics,2,2)=1.
	ftdosh(ics)=1.

	do ic=ics+1,nc
	 ftdo(ic,1,1)=fdo(ic,1,1)*ftdo(ic-1,1,1)+
     &                fdo(ic,1,2)*ftdo(ic-1,2,1)
	 ftdo(ic,1,2)=fdo(ic,1,1)*ftdo(ic-1,1,2)+
     &                fdo(ic,1,2)*ftdo(ic-1,2,2)
	 ftdo(ic,2,1)=fdo(ic,2,1)*ftdo(ic-1,1,1)+
     &                fdo(ic,2,2)*ftdo(ic-1,2,1)
	 ftdo(ic,2,2)=fdo(ic,2,1)*ftdo(ic-1,1,2)+
     &                fdo(ic,2,2)*ftdo(ic-1,2,2)
	 ftdosh(ic)=fdosh(ic)*ftdosh(ic-1)
	enddo

c                          Termes : C2(kr)
c                          ---------------
c               Constantes dependant du nombre d'onde et
c               de la couche source
 
	cs2=c2(ics)
	cs4=kr2
	cs5=ai*cgam(ics)
	cs3=ckb2(ics)/cs5
	cs6=ckb2(ics)
	cs8=ai*cnu(ics)
	cs7=cs8*kr2
	cs9=(ckb2(ics)-2.*kr2)/cs5


	do ir1=1,ncr            !boucle sur les couches recepteur
	 ic=irc(ir1)
	 cr1=ai*cgam(ic)
	 cr2=cgam(ic)*cgam(ic)
	 cr3=ai*cnu(ic)
	 idel=ics-ic
	 abovecomp=.true.
	 belowcomp=.true.

	 do is2=1,nzs(is1)      !boucle sur les prof. sources dans ics
	 is0=izss(1,is2,is1)
	 zsc=zs(is0)

	 do ir2=1,nzr(ir1)              !boucle sur les prof. recept. dans ic=irc(ir1)
	 ir0=izrr(1,ir2,ir1)
	 zc=zr(ir0)

	 if (idel.eq.0) then
	  dz = zsc-zc
	 else
	  dz = idel
	 endif

c+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
c
c  Vecteurs potentiel montant (pu) et descendant (pd),
c  dans chaque couche recepteur, pour les 6 sources elementaires
c
c+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

c               Recepteurs au dessus de la source
	if ((dz.gt.0.).and.abovecomp) then
	 abovecomp=.false.
	 pu(1,1)=ftup(ic,1,1)*su1(is0,1)+ftup(ic,1,2)*su1(is0,2)
	 pu(2,1)=ftup(ic,2,1)*su1(is0,1)+ftup(ic,2,2)*su1(is0,2)
	 pd(1,1)=nt(ic,1,1)*pu(1,1)+nt(ic,1,2)*pu(2,1)
	 pd(2,1)=nt(ic,2,1)*pu(1,1)+nt(ic,2,2)*pu(2,1)
   
	 pu(1,2)=ftup(ic,1,1)*su2(is0,1)+ftup(ic,1,2)*su2(is0,2)
	 pu(2,2)=ftup(ic,2,1)*su2(is0,1)+ftup(ic,2,2)*su2(is0,2)
	 pd(1,2)=nt(ic,1,1)*pu(1,2)+nt(ic,1,2)*pu(2,2) 
	 pd(2,2)=nt(ic,2,1)*pu(1,2)+nt(ic,2,2)*pu(2,2)
   
	 pu(1,3)=ftup(ic,1,1)*su3(is0,1)+ftup(ic,1,2)*su3(is0,2)
	 pu(2,3)=ftup(ic,2,1)*su3(is0,1)+ftup(ic,2,2)*su3(is0,2)
	 pd(1,3)=nt(ic,1,1)*pu(1,3)+nt(ic,1,2)*pu(2,3) 
	 pd(2,3)=nt(ic,2,1)*pu(1,3)+nt(ic,2,2)*pu(2,3)
   
	 pu(1,4)=ftup(ic,1,1)*su4(is0,1)+ftup(ic,1,2)*su4(is0,2)
	 pu(2,4)=ftup(ic,2,1)*su4(is0,1)+ftup(ic,2,2)*su4(is0,2)
	 pd(1,4)=nt(ic,1,1)*pu(1,4)+nt(ic,1,2)*pu(2,4) 
	 pd(2,4)=nt(ic,2,1)*pu(1,4)+nt(ic,2,2)*pu(2,4)
   
	 push(1)=ftupsh(ic)*su1sh(is0)
	 pdsh(1)=ntsh(ic)*push(1)
   
	 push(2)=ftupsh(ic)*su2sh(is0)
	 pdsh(2)=ntsh(ic)*push(2)

c             Recepteurs au dessous de la source
	else if ((dz.lt.0.).and.belowcomp) then
	 belowcomp=.false.
	 pd(1,1)=ftdo(ic,1,1)*sd1(is0,1)+ftdo(ic,1,2)*sd1(is0,2)
	 pd(2,1)=ftdo(ic,2,1)*sd1(is0,1)+ftdo(ic,2,2)*sd1(is0,2)
	 pu(1,1)=mt(ic,1,1)*pd(1,1)+mt(ic,1,2)*pd(2,1)
	 pu(2,1)=mt(ic,2,1)*pd(1,1)+mt(ic,2,2)*pd(2,1)
    
	 pd(1,2)=ftdo(ic,1,1)*sd2(is0,1)+ftdo(ic,1,2)*sd2(is0,2)
	 pd(2,2)=ftdo(ic,2,1)*sd2(is0,1)+ftdo(ic,2,2)*sd2(is0,2)
	 pu(1,2)=mt(ic,1,1)*pd(1,2)+mt(ic,1,2)*pd(2,2)
	 pu(2,2)=mt(ic,2,1)*pd(1,2)+mt(ic,2,2)*pd(2,2)
    
	 pd(1,3)=ftdo(ic,1,1)*sd3(is0,1)+ftdo(ic,1,2)*sd3(is0,2)
	 pd(2,3)=ftdo(ic,2,1)*sd3(is0,1)+ftdo(ic,2,2)*sd3(is0,2)
	 pu(1,3)=mt(ic,1,1)*pd(1,3)+mt(ic,1,2)*pd(2,3)
	 pu(2,3)=mt(ic,2,1)*pd(1,3)+mt(ic,2,2)*pd(2,3)
    
	 pd(1,4)=ftdo(ic,1,1)*sd4(is0,1)+ftdo(ic,1,2)*sd4(is0,2)
	 pd(2,4)=ftdo(ic,2,1)*sd4(is0,1)+ftdo(ic,2,2)*sd4(is0,2)
	 pu(1,4)=mt(ic,1,1)*pd(1,4)+mt(ic,1,2)*pd(2,4)
	 pu(2,4)=mt(ic,2,1)*pd(1,4)+mt(ic,2,2)*pd(2,4)
    
	 pdsh(1)=ftdosh(ic)*sd1sh(is0)
	 push(1)=mtsh(ic)*pdsh(1)
   
	 pdsh(2)=ftdosh(ic)*sd2sh(is0)
	 push(2)=mtsh(ic)*pdsh(2)
	endif

c+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
c
c  Deplacements pour chaque sources du tenseur, exprime a l'aide de 
c  de source intermediaires. Chaque source intermediaire correspond
c  aux rayonnements des trois potentiels PHI, PSI, KHI de chaque
c  moment du tenseur.
c
c  ex : Mxy -> PHI0, PSI0, KHI0
c
c              PHI0 -> PHI, PSI apres conversion sur les interfaces
c              PSI0 -> PHI, PSI       "                "
c              KHI0 -> KHI            "                "
c
c                       -------------------------
c
c               u = C2(kr)*C4(kr.r)*C5(kr,z)
c
c+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      
c                          Termes : C5(kr,z)
c                          -----------------
 
c               Dephasage par rapport au sommet de la couche pour
c               les ondes PHI, PSI et KHI

      arg=-ai*cgam(ic)*zc
      if (dble(arg).lt.explim) arg=cmplx(explim,dimag(arg))
      egam=exp(arg)
      egaminv=1./egam
      arg=-ai*cnu(ic)*zc
      if (dble(arg).lt.explim) arg=cmplx(explim,dimag(arg))
      enu=exp(arg)
      enuinv=1./enu

c               termes sources 
      s1phiu=pu(1,1)*enuinv
      s1phid=pd(1,1)*enu
      s1psiu=pu(2,1)*egaminv
      s1psid=pd(2,1)*egam
      s2phiu=pu(1,2)*enuinv
      s2phid=pd(1,2)*enu
      s2psiu=pu(2,2)*egaminv
      s2psid=pd(2,2)*egam
      s3phiu=pu(1,3)*enuinv
      s3phid=pd(1,3)*enu
      s3psiu=pu(2,3)*egaminv
      s3psid=pd(2,3)*egam
      s4phiu=pu(1,4)*enuinv
      s4phid=pd(1,4)*enu
      s4psiu=pu(2,4)*egaminv
      s4psid=pd(2,4)*egam
      s5=push(1)*egaminv+pdsh(1)*egam
      s6=push(2)*egaminv+pdsh(2)*egam
 
c                               Source phi
      cz1=(s1phiu+s1phid)+cr1*(s1psiu-s1psid)
      cz1b=cr3*(s1phiu-s1phid)+cs4*(s1psiu+s1psid)
c                               Source phi*sign(z-z0)
      cz2=(s2phiu+s2phid)+cr1*(s2psiu-s2psid)
      cz2b=cr3*(s2phiu-s2phid)+cs4*(s2psiu+s2psid)
c                               Source psi
      cz3=(s3phiu+s3phid)+cr1*(s3psiu-s3psid)
      cz3b=cr3*(s3phiu-s3phid)+cs4*(s3psiu+s3psid)
c                               Source psi*sign(z-z0)
      cz4=(s4phiu+s4phid)+cr1*(s4psiu-s4psid)
      cz4b=cr3*(s4phiu-s4phid)+cs4*(s4psiu+s4psid)

      do is3=1,nzss(is2,is1)    !boucle distances radiales
       is=izss(is3,is2,is1)

	 do ir3=1,nzrr(ir2,ir1)         !boucle sur distances radiales
	  ir=izrr(ir3,ir2,ir1)
	  jrs=irs(ir,is)
	  if (.not.tconv(ir,is)) then   !on a pas encore converge

c                       Termes u=C2(kr)*C4(kr.r)*C5(kr,z)
c                       -------------------------------
c               (Tous les termes de deplacement presentant les memes 
c                dependances en theta sont regroupes)

c               Mxx :
c               -----
c                       PHI0 + PSI0
c       ur => u(1) ; u(2)
c       ut => -u(2)
c       uz => u(3) ; u(4)
c                       KHI0
c       ur => u(5)
c       ut => -u(6)
      cu=cs2*cz1+cz4
      cdu1=k3(jrs)*cu
      cdu2=k4(jrs)*cu
      u(ir,is,1)= cdu1 + u(ir,is,1)
      u(ir,is,2)= cdu2 + u(ir,is,2)

      cu2=cs2*cz1b+cz4b
      cdu3=k1(jrs)*cu2
      cdu4=k2(jrs)*cu2
      cdu5=cs3*k4(jrs)*s5
      cdu6=cs3*k3(jrs)*s5
      u(ir,is,3)= cdu3 + u(ir,is,3)
      u(ir,is,4)= cdu4 + u(ir,is,4)
      u(ir,is,5)= cdu5 + u(ir,is,5)
      u(ir,is,6)= cdu6 + u(ir,is,6)



c       Double couple Mxy+Myx :
c       -----------------------
c                       PHI0 + PSI0
c       ur => u(1) 
c       ut => u(2)
c       uz => u(3)
c                       KHI0
c       ur => u(5)
c       ut => u(6)

c       Double couple Mxz+Mzx :
c       -----------------------
c                       PHI0 + PSI0 + KHI0
c       ur =>  u(7)
c       ut => -u(8)
c       uz =>  u(9)

      cu3=-2.*cs4*cz2 + cs9*cz3
      cdu7=k5(jrs)*cu3 - cs6*k2(jrs)*s6
      cdu8=k2(jrs)*cu3 - cs6*k5(jrs)*s6
      cdu9=fj1(jrs)*(-2.*cs4*cz2b + cs9*cz3b)
      u(ir,is,7)= cdu7 + u(ir,is,7)
      u(ir,is,8)= cdu8 + u(ir,is,8)
      u(ir,is,9)= cdu9 + u(ir,is,9)
      

c               Myy :
c               -----
c                       PHI0 + PSI0
c     ur => u(1) ; u(2)
c     ut => u(2)
c     uz => u(3) ; u(4)
c                       KHI0
c     ur => -u(5)
c     ut => u(6)

c       Double couple Myz+Mzy :
c       -----------------------
c                       PHI0 + PSI0 + KHI0
c     ur =>  u(7)
c     ut =>  u(8)
c     uz =>  u(9)

c               Mzz :
c               -----
c                       PHI0 + PSI0
c       ur => u(10)
c       uz => -u(11)

	  cdu10=fj1(jrs)*(cs7*cz1 + kr2*cz4)
	  cdu11=k0(jrs)*(cs8*cz1b + cz4b)
	  u(ir,is,10)= cdu10 + u(ir,is,10)
	  u(ir,is,11)= cdu11 + u(ir,is,11)
 
	  cdu(ir,is,1)=cdu1
	  cdu(ir,is,2)=cdu2
	  cdu(ir,is,3)=cdu3
	  cdu(ir,is,4)=cdu4
	  cdu(ir,is,5)=cdu5
	  cdu(ir,is,6)=cdu6
	  cdu(ir,is,7)=cdu7
	  cdu(ir,is,8)=cdu8
	  cdu(ir,is,9)=cdu9
	  cdu(ir,is,10)=cdu10
	  cdu(ir,is,11)=cdu11

	endif           ! test conv deja obtenue
	 enddo          ! boucle dist. radiale
	 enddo          ! boucle dist. radiale
       enddo            ! boucle prof. dans couche ic
       enddo            ! boucle prof. dans couche ics
      enddo             ! boucle sur couche ic
      enddo             ! boucle sur couche ics

 
      if (tmin) then
       ttconv=.true.
       do is=1,ns
	do ir=1,nr
	  if (.not.tconv(ir,is)) then 
	  tconv(ir,is)=.true.
	  cdu1=cdu(ir,is,1)
	  cdu2=cdu(ir,is,2)
	  cdu3=cdu(ir,is,3)
	  cdu4=cdu(ir,is,4)
	  cdu5=cdu(ir,is,5)
	  cdu6=cdu(ir,is,6)
	  cdu7=cdu(ir,is,7)
	  cdu8=cdu(ir,is,8)
	  cdu9=cdu(ir,is,9)
	  cdu10=cdu(ir,is,10)
	  cdu11=cdu(ir,is,11)

      r1=dble(u(ir,is,1))
      i1=dimag(u(ir,is,1))
      r1=(r1*r1+i1*i1)*uconv
      r2=dble(cdu1)
      i2=dimag(cdu1)
      r2=(r2*r2+i2*i2)
      tconv(ir,is)=((r2.le.r1).and.tconv(ir,is))
 
      r1=dble(u(ir,is,2))
      i1=dimag(u(ir,is,2))
      r1=(r1*r1+i1*i1)*uconv
      r2=dble(cdu2)
      i2=dimag(cdu2)
      r2=(r2*r2+i2*i2)
      tconv(ir,is)=((r2.le.r1).and.tconv(ir,is))

      r1=dble(u(ir,is,3))
      i1=dimag(u(ir,is,3))
      r1=(r1*r1+i1*i1)*uconv
      r2=dble(cdu3)
      i2=dimag(cdu3)
      r2=(r2*r2+i2*i2)
      tconv(ir,is)=((r2.le.r1).and.tconv(ir,is))
 
      r1=dble(u(ir,is,4))
      i1=dimag(u(ir,is,4))
      r1=(r1*r1+i1*i1)*uconv
      r2=dble(cdu4)
      i2=dimag(cdu4)
      r2=(r2*r2+i2*i2)
      tconv(ir,is)=((r2.le.r1).and.tconv(ir,is))
 
      r1=dble(u(ir,is,5))
      i1=dimag(u(ir,is,5))
      r1=(r1*r1+i1*i1)*uconv
      r2=dble(cdu5)
      i2=dimag(cdu5)
      r2=(r2*r2+i2*i2)
      tconv(ir,is)=((r2.le.r1).and.tconv(ir,is))
 
      r1=dble(u(ir,is,6))
      i1=dimag(u(ir,is,6))
      r1=(r1*r1+i1*i1)*uconv
      r2=dble(cdu6)
      i2=dimag(cdu6)
      r2=(r2*r2+i2*i2)
      tconv(ir,is)=((r2.le.r1).and.tconv(ir,is))

      r1=dble(u(ir,is,7))
      i1=dimag(u(ir,is,7))
      r1=(r1*r1+i1*i1)*uconv
      r2=dble(cdu7)
      i2=dimag(cdu7)
      r2=(r2*r2+i2*i2)
      tconv(ir,is)=((r2.le.r1).and.tconv(ir,is))
 
      r1=dble(u(ir,is,8))
      i1=dimag(u(ir,is,8))
      r1=(r1*r1+i1*i1)*uconv
      r2=dble(cdu8)
      i2=dimag(cdu8)
      r2=(r2*r2+i2*i2)
      tconv(ir,is)=((r2.le.r1).and.tconv(ir,is))
 
      r1=dble(u(ir,is,9))
      i1=dimag(u(ir,is,9))
      r1=(r1*r1+i1*i1)*uconv
      r2=dble(cdu9)
      i2=dimag(cdu9)
      r2=(r2*r2+i2*i2)
      tconv(ir,is)=((r2.le.r1).and.tconv(ir,is))

	  r1=dble(u(ir,is,10))
	  i1=dimag(u(ir,is,10))
	  r1=(r1*r1+i1*i1)*uconv
	  r2=dble(cdu10)
	  i2=dimag(cdu10)
	  r2=(r2*r2+i2*i2)
	  tconv(ir,is)=((r2.le.r1).and.tconv(ir,is))
 
	  r1=dble(u(ir,is,11))
	  i1=dimag(u(ir,is,11))
	  r1=(r1*r1+i1*i1)*uconv
	  r2=dble(cdu11)
	  i2=dimag(cdu11)
	  r2=(r2*r2+i2*i2)
	  tconv(ir,is)=((r2.le.r1).and.tconv(ir,is))

	  ttconv=ttconv.and.tconv(ir,is)
	  endif
	enddo
	enddo
      endif

      return
      end
c @(#) reflect5.F       AXITRA 4.12     12/7/93 4
c***********************************************************
c*                                                         *
c*              SUBROUTINE REFLECT5                        *
c*                                                         *
c*        Calcul des deplacements avec diverses rotations  *
c*        et recombinaisons. Passage aux sources du tenseur*
c*        des moments sismiques (M1 a M6)                  *
c*        Multiplication par les termes frequentiel        *
c*        et angulaire :                                   *
c*                     u=u*C3(theta)*CFF(omega )           *
c***********************************************************
     
      subroutine reflect5

      include "param.inc"
      include "dimen1.inc"
      include "dimen2.inc"

      real*8   cor,cor2,co2r
      complex*16    urxx,utxx,uzxx,urxy,utxy,uzxy,urxz,utxz,uzxz,
     1           uryy,utyy,uzyy,uryz,utyz,uzyz,urzz,utzz,uzzz,
     2           ux(6),uy(6),uz(6)

      do is=1,ns
      do ir=1,nr
      do it=1,11
      u(ir,is,it)=u(ir,is,it)*a1*cff(is)
      enddo

c+++++++++++++  
c       Deplacement dus aux sources Mxx,Mxy,Mxz,Myy,Myz,Mzz avec
c       convention de signe inverse pour le deplacement vertical
c       (positif vers le haut)
c+++++++++++++

      cor=cosr(ir,is)
      sir=sinr(ir,is)
      cor2=cor*cor
      sir2=sir*sir
      co2r=cor2-sir2
      si2r=2.*cor*sir

c       Mxx
      
      urxx=-cor2 *  u(ir,is,1) - u(ir,is,2) - co2r*u(ir,is,5)
      utxx= si2r * (u(ir,is,2) + u(ir,is,6)/2.)
      uzxx= cor2 *  u(ir,is,3) + u(ir,is,4)

c       Mxy+Myx

      urxy=-si2r * (u(ir,is,1) + 2.*u(ir,is,5))
      utxy=-co2r * (2.*u(ir,is,2) + u(ir,is,6))
      uzxy= si2r *  u(ir,is,3)

c       Mxz+Mzx

      urxz=-cor * u(ir,is,7)
      utxz= sir * u(ir,is,8)
      uzxz= cor * u(ir,is,9) 

c       Myy

      uryy=-sir2 *  u(ir,is,1) - u(ir,is,2) + co2r*u(ir,is,5)
      utyy=-si2r * (u(ir,is,2) + u(ir,is,6)/2.) 
      uzyy= sir2 *  u(ir,is,3) + u(ir,is,4)

c       Myz+Mzy

      uryz=-sir * u(ir,is,7)
      utyz=-cor * u(ir,is,8)
      uzyz= sir * u(ir,is,9)

c       Mzz

      urzz=-u(ir,is,10)
      utzz= 0.
      uzzz=-u(ir,is,11)

c+++++++++++
c     Passage aux sources bis, 5 dislocations elementaires et
c     une source isotrope + rotation des composantes pour passer de 
c     radial/tangentiel a Ox/Oy
c+++++++++++
c
c+++++++++++
cifdef XYZ  results in x,y,z coordinates
cc      M1 = (Mxy + Myx)
      ux(1)=urxy*cor - utxy*sir
      uy(1)=urxy*sir + utxy*cor
      uz(1)=uzxy
c
cc      M2 = (Mxz + Mzx)
      ux(2)=urxz*cor - utxz*sir
      uy(2)=urxz*sir + utxz*cor
      uz(2)=uzxz
c
cc      M3 = -(Myz + Mzy)
      ux(3)=-uryz*cor + utyz*sir
      uy(3)=-uryz*sir - utyz*cor
      uz(3)=-uzyz
c
cc      M4 = -Mxx + Mzz
      ux(4)=(-urxx + urzz)*cor - (-utxx + utzz)*sir
      uy(4)=(-urxx + urzz)*sir + (-utxx + utzz)*cor
      uz(4)=-uzxx + uzzz
c
cc      M5 = -Myy + Mzz
      ux(5)=(-uryy + urzz)*cor - (-utyy + utzz)*sir
      uy(5)=(-uryy + urzz)*sir + (-utyy + utzz)*cor
      uz(5)=-uzyy + uzzz
c
cc      M6 = Mxx + Myy + Mzz
      ux(6)=(urxx + uryy + urzz)*cor - (utxx + utyy + utzz)*sir
      uy(6)=(urxx + uryy + urzz)*sir + (utxx + utyy + utzz)*cor
      uz(6)=uzxx + uzyy + uzzz
c ++++++++++++++++++++++++++++++++++++
c results in r,t,z coordinates
c       M1 = (Mxy + Myx)
c     ux(1)=urxy
c     uy(1)=utxy
c     uz(1)=uzxy
c
c       M2 = (Mxz + Mzx)
c     ux(2)=urxz
c     uy(2)=utxz
c     uz(2)=uzxz
c
c       M3 = -(Myz + Mzy)
c     ux(3)=-uryz
c     uy(3)=-utyz
c     uz(3)=-uzyz
c
c       M4 = -Mxx + Mzz
c     ux(4)=(-urxx + urzz)
c     uy(4)=(-utxx + utzz)
c     uz(4)=-uzxx + uzzz
c
c       M5 = -Myy + Mzz
c     ux(5)=(-uryy + urzz)
c     uy(5)=(-utyy + utzz)
c     uz(5)=-uzyy + uzzz
c
c       M6 = Mxx + Myy + Mzz
c     ux(6)=(urxx + uryy + urzz)
c     uy(6)=(utxx + utyy + utzz)
c     uz(6)=uzxx + uzyy + uzzz
c
c
      write(out2) (ux(it),it=1,6)
      write(out2) (uy(it),it=1,6)
      write(out2) (uz(it),it=1,6)


      enddo
      enddo
      
      return
      end