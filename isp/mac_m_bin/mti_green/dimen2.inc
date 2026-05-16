c @(#) dimension2.F	AXITRA 4.12	12/7/93 4
c*****************************************************************************
c*                                                                           *
c*                          DIMENSION2                                       *  
c*                                                                           *
c* 	Declaration de variables utilisees dans le calcul par boucle en kr   *
c*	externe	 (routines reflect0 a reflect6)                              *
c*                                                                           *
c*****************************************************************************


      real*8        fj1,k0,k1,k2,k3,k4,k5
      complex*16     me1,me2,nt,mt,ntsh,mtsh,fdo,
     1            fup,fdosh,fupsh,su1,sd1,su2,sd2,su3,sd3,su4,sd4,
     2            su1sh,sd1sh,su2sh,sd2sh,u,ru,rd,tu,td,rdsh,rush,
     3            tush,tdsh

      common /dim2a/ cka(ncp),ckb(ncp),cka2(ncp),ckb2(ncp),
     &               cnu(ncp),cgam(ncp),c2(ncp),cff(nsp)
      common /dim2b/ fj1(nrsp),k0(nrsp),k1(nrsp),k2(nrsp),
     &               k3(nrsp),k4(nrsp),k5(nrsp)
      common /dim2c/ rd(ncp,2,2),ru(ncp,2,2),td(ncp,2,2),
     &               tu(ncp,2,2),rdsh(ncp),rush(ncp),
     &               tdsh(ncp),tush(ncp),me1(ncp),me2(ncp)
      common /dim2d/ nt(ncp,2,2),mt(ncp,2,2),ntsh(ncp),mtsh(ncp)
      common /dim2e/ fdo(ncp,2,2),fup(ncp,2,2),fupsh(ncp),fdosh(ncp)
      common /dim2f/ su1(nsp,2),sd1(nsp,2),su2(nsp,2),sd2(nsp,2),
     &               su3(nsp,2),sd3(nsp,2),su4(nsp,2),sd4(nsp,2),
     &               su1sh(nsp),sd1sh(nsp),su2sh(nsp),sd2sh(nsp)
      common /dim2g/ u(nrp,nsp,11)
