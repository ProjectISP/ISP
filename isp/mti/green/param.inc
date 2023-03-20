c @(#) parameter.F      AXITRA 4.11     12/7/93 4
c**********************************************************************
c*                                                                    *
c*                  PARAMETER                                         *
c*                                                                    *
c*    nsp: nbre max source
c*    nrp: nbre max stat
c*    ncp: nbre max couche
c*    ikmin: nbre min d'iterations
c*    nkmax: (si fct de Bessel stockees) dimension tab fct Bessel
c**********************************************************************

      implicit real*8 (a-b,d-h,o-z)
      implicit complex*16 (c)

      integer   in1,in2,in3,out,out2
      parameter (in1=10,in2=11,in3=12,out=13,out2=14)

      integer   nsp,nrp,ncp,nrsp,ikmin
      parameter (nsp=1,nrp=31,ncp=200,nrsp=nrp*nsp,ikmin=100)
      real*8    explim
      parameter (explim=-600.)

      integer   nkmax
      parameter (nkmax=2000)