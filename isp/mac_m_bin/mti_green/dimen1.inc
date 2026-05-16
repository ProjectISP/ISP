c @(#) dimension1.F	AXITRA 4.12	12/7/93 4
c*****************************************************************************
c*                                                                           *
c*                          DIMENSION1                                       *  
c*                                                                           *
c* 	Declaration de variables generales :                                 *
c*		constantes, parametres du modele                             *
c*                                                                           *
c*****************************************************************************


      integer	 ncr,irc,nzr,nzrr,izrr,nr,nc,ns,nrs,
     1           ncs,isc,nzs,nzss,izss,irs

      complex*16    ai,omega,omega2,a1
      real*8       cosr,sinr,pi,pi2,hc,kr,kr2,rr,uconv,
     1           xlnf,xs,ys,zs,xr,yr,zr,vp,vs,vp2,vs2,
     2           rho,qp,qs

      logical    ttconv

      common /dim1a/ ai,pi,pi2,omega,omega2,a1,kr,kr2,uconv,xlnf,
     &               ttconv
      common /dim1b/ xr(nrp),yr(nrp),zr(nrp),cosr(nrp,nsp),sinr(nrp,nsp)
      common /dim1c/ xs(nsp),ys(nsp),zs(nsp)
      common /dim1d/ rr(nrsp),nc,nr,ns,irs(nrp,nsp),nrs
      common /dim1e/ hc(ncp),vp(ncp),vs(ncp),vp2(ncp),vs2(ncp),
     &               rho(ncp),qp(ncp),qs(ncp)
      common /dim1f/ ncr,irc(ncp),nzr(ncp),nzrr(nrp,ncp),
     &               izrr(nrp,nrp,ncp)
      common /dim1g/ ncs,isc(ncp),nzs(ncp),nzss(nsp,ncp),
     &               izss(nsp,nsp,ncp)
