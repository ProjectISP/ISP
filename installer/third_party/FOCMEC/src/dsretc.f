C+
	program dsretc
C
C	Based on the Aki & Richards convention, gives all 
C	  representations of a focal mechanism for input
C	  Dip, Strike, Rake    or
C	  A and N trend and plunge    or
C	  P and T trend and plunge.
C
C	The code is Fortran 77 for a VAX/VMS system
C	Subroutine s called by this program are
C	  FMREPS,PTTPIN,ANTPIN,DSRIN,AN2MOM,V2TRPL,TRPL2V,AN2DSR, MT_IN
C	21 August 1991:  sun version
C	30 December 1993 Incuded Stuart Sipkin's beachball. who gives
C         credit to Bob Uhrhammer.
C	7 March 2002: Added moment tensor input. other formats use
C         only the double-couple part.
C	June 2009: Added angle of A with plane formed by vertical
C	  and the trend of B.  Also added angle of N.
C	  Also added "save" to two subroutines to fix problem with
C	  printer plots on the screen.
C       March 2014: Added iput choice if P,T.
C	June 2017:  Writes out Angle.  Took out A & N opton for input because
C	  for some inputs,P and T may be reversed.
C-
	LOGICAL PT,DSR,TRUTH,first,MT
	character*80 getstring,commnt
	REAL*4 MOMTEN(6)
	DIMENSION PTTP(4),ANGS(3),ANGS2(3),ANBTP(6)
        rd = 45.0/atan(1.0)
C
C	If your compiler complains about form='print', leave it out.
C
	open(unit=2,file='dsretc.lst',status='unknown')

100	COMMNT = getstring('Comment')
	CALL TIMDAT(2,'dsretc')
200	write(2,'(/,5x,a)') commnt(1:lenc(commnt))
	WRITE(*,*) 'Can enter D, S & R or P & T or A N or moment tensor'
	DSR = .FALSE.
	PT = .FALSE.
	MT = .false.
	IF (TRUTH('Dip, Strike and Rake?..[Y]')) THEN
	  DSR = .TRUE.
	  CALL PRINTX('Enter Dip, Strike and Rake (degrees)')
	  READ(*,*) (ANGS(J),J=1,3)
	ELSE IF (TRUTH('P and T axes trend and plunge?..[Y]')) THEN
	  PT = .TRUE.
	  CALL PRINTX
     .	  ('Enter trend and plunge for P and T (t,p,t,p)')
	  READ(*,*) (PTTP(J),J=1,4)
          WRITE (2,5) PTTP
        else 
          MT = .TRUE.
	END IF
	CALL FMREPS(ANBTP,ANGS,PTTP,ANGS2,PT,DSR,mt,MOMTEN,2,6)
	call get_angle(angs/rd)
	first = .true.
	call bball(momten,pttp(1),pttp(2),pttp(3),pttp(4),2,first)
	call bball(momten,pttp(1),pttp(2),pttp(3),pttp(4),6,first)
	IF (.NOT.TRUTH('Run some more?...[Y]')) STOP
        first = .true.
	IF (TRUTH('Same comment?..[Y]')) GO TO 200
	GO TO 100
5	FORMAT(5X,'Input: Trend, Plunge of P,T ',4F9.2)
	END
C+
	subroutine get_angle(DSR)
C
C   Subroutine get_angle is a bare-bones version of program fm2focmec.
C   Program Focmec creates possible focal mechanisms by values of
C       three angles: the trend and plunge of B and Angle, which is
C       the angle between N and the projection of the B trend into the
C       plane perpendicular to B.
C   Program fm2focmec  finds the values for the Focmec input parameters
C       for a focal mechanism defined by the dip, strike, and rake.
C       The output is three angles: the trend and plunge of B and the
C       angle defined above.  There is an unresolvable ambiguity in
C       finding the third angle using this procedure, as the angle
C       used by Focmec might be either the angle calculated by this
C       programor 180 degrees minus that angle.  Also, the dsr produced
C       by Focmec may be the auxiliary-plane solution, sothere is a
C       total of four possible vaules for Angle.  In addition to printing
C       the angles, lines are provided that could be used in a Focmec
C       input script: the minimun, increment, and maximum for each
C       angle.  These can be cut-and-pasted into a Focmec input scriptt.
C    Only output for this version is to write out Angle to unit 2.
C-
	REAL*4 x(3),a(3),ain(3),BMATRX(3,3),ZROT(3,3),YROT(3,3),ANB(3,3)
        Real*4 anbtrpl(6),b(3),n(3),nin(3),dsr(3),dsr2(3),bmat2(3,3)
	logical ok_ang
        pi = 4.0*atan(1.0)
        rd = 180.0/pi
        sr2 = sqrt(2.0)
        dip = dsr(1)
        str = dsr(2)
        rake = dsr(3)
        A(1) = COS(RAKE)*COS(STR) + SIN(RAKE)*COS(DIP)*SIN(STR)
        A(2) = COS(RAKE)*SIN(STR) - SIN(RAKE)*COS(DIP)*COS(STR)
        A(3) = -SIN(RAKE)*SIN(DIP)
        N(1) = -SIN(STR)*SIN(DIP)
        N(2) = COS(STR)*SIN(DIP)
        N(3) = -COS(DIP)
        B(1) = COS(STR)*SIN(RAKE) - COS(RAKE)*COS(DIP)*SIN(STR)
        B(2) = COS(RAKE)*COS(STR)*COS(DIP) + SIN(RAKE)*SIN(STR)
        B(3) = COS(RAKE)*SIN(DIP)
        CALL V2TRPL(A,ANBTRPL(1),PI)
        CALL V2TRPL(N,ANBTRPL(3),PI)
        CALL V2TRPL(B,ANBTRPl(5),PI)
c
            DO J=1,3
              DO kk=1,3
                ZROT(J,kk) = 0.0
                YROT(J,kk) = 0.0
              end do
              ain(j) = a(j)
              nin(j) = n(j)
            end do
            trend = anbtrpl(5)
            plunge = anbtrpl(6)
C
C	First rotate about Z (= Down) through an angle TREND.
C		X now has the trend of B
C
            ZROT(1,1) = COS(TREND)
            ZROT(2,2) = ZROT(1,1)
            ZROT(1,2) = SIN(TREND)
            ZROT(2,1) = -ZROT(1,2)
            ZROT(3,3) = 1.0
C
C	Now rotate about Y through an angle -(90-PLUNGE).
C		This rotates the Z axis into B.
C		Becasue the rotation is about a horizontal axis
C		perpendicular to the plane defined by the vertical
C		direction and the B trend, the rotated X axis will
C		be in that plane, and its trend will differ from
C               that of B by 180.
C
            YROT(1,1) = SIN(PLUNGE)
            YROT(3,3) = YROT(1,1)
            YROT(1,3) = -COS(PLUNGE)
            YROT(3,1) = -YROT(1,3)
            YROT(2,2) = 1.0
C
C	BMATRX is the product of YROT and ZROT
C
            CALL GMPRD(YROT,ZROT,BMATRX,3,3,3)
            do j=1,3
              x(j) = bmatrx(1,j)
            enddo
        do k=1,2
          if (k .eq. 2) then
            do j=1,3
              n(j) = ain(j)
              a(j) = nin(j)
            enddo
            call an2dsr(a,n,dsr,pi)
          endif
C
C               cos(angle) is the dot product of n and x.  The probem is that
C               by construction, we mauy actually want -n, as both n and -n
C               have the same trend and plunge because we only work with the
C               lower half plane.  So there are two possible values for angle.
C        
          cosangle = 0.0
	  do j=1,3
	    cosangle = cosangle + n(j)*x(j)
	  enddo
	  angledeg = rd*acos(cosangle)

	  CALL FLTSOL(A,N,BMAT2,PLUNGE,TREND,ANGLEDEG/RD,1)
          call an2dsr(a,n,dsr2,pi)
          if (ok_ang(dsr,dsr2)) then
            write(2,'(a,f9.2)') '     Angle: ',angledeg
            write(*,'(a,f9.2)') '     Angle: ',angledeg
            return
          endif
          angledeg = 180.0 - angledeg
	  CALL FLTSOL(A,N,BMAT2,PLUNGE,TREND,ANGLEDEG/RD,1)
          call an2dsr(a,n,dsr2,pi)
          if (ok_ang(dsr,dsr2)) then
            write(2,'(a,f9.2)') '     Angle: ',angledeg
            write(*,'(a,f9.2)') '     Angle: ',angledeg
            return
          endif
        enddo
        stop
        END
C
	logical function ok_ang(dsr,dsr2)
C
	real*4 dsr(3),dsr2(3)
C
	ok_ang = .false.
	do j=1,3
		If (abs(dsr(j)-dsr2(j)) .gt. 0.001) return
	enddo
	ok_ang = .true.
        return
        end
C+
      subroutine bball(g,pazim,pplng,tazim,tplng,unit,first)

c ...... generate printer plot rendition of lower hemisphere 
c        equal area projection
C	g has the six elements of the moment tensor, the rest are the
C	  plunge and trends of the P and T axes in degrees. unit is the output
C	  unit.
C	From Stuart Sipkin and Bob Uhrhammer 1993
C	1 October 2001: Replaced his sind, etc, with sin as not all compilers
C		know about degree versions of sin, cos, etc.
C-
      dimension g(6)
      integer unit
      character*1 ach(39,72),aplus,aminus,apaxis,ataxis,ablank
      logical first
c
      data aplus,aminus,apaxis,ataxis,ablank /'#','-','P','T',' '/
      data radius /1.41/
c
c ...... construct lower hemisphere fps 
c
      save
      rd = 45.0/atan(1.0)
      r0=radius
      x0=r0+0.250
      y0=r0+0.500
      ix0=12.*x0
      iy0=6.5*y0
      do 3 i=1,2*ix0
      do 2 j=1,2*iy0
      dx=real(i-ix0)/12.
      dy=-real(j-iy0)/6.5
      dd=dx*dx+dy*dy
      if(dd.gt.0.) then
        del=sqrt(dd)
      else
        del=0.
      endif
      if((dx.eq.0.).and.(dy.eq.0.)) then
        theta=0.
      else
        theta=rd*atan2(dx,dy)
      endif
      if(del.gt.r0) then
        ach(j,i)=ablank
        go to 1
      endif
      if(del.ge.r0) then
        aoi=90.0
      else
        aoi=90.*del/r0
      endif
      if(polar(g,aoi,theta,first).gt.0.) then
        ach(j,i)=aplus
      else
        ach(j,i)=aminus
      endif
    1 continue
    2 continue
    3 continue
c
c ...... add P & T axis
c
      ixp=nint(r0*12.*(90.-pplng)*sin(pazim/rd)/90.+real(ix0))
      iyp=nint(-r0*6.5*(90.-pplng)*cos(pazim/rd)/90.+real(iy0))
      do 5 i=ixp-1,ixp+1
      do 4 j=iyp-1,iyp+1
      ach(j,i)=ablank
    4 continue
    5 continue
      ach(iyp,ixp)=apaxis
      ixt=nint(r0*12.*(90.-tplng)*sin(tazim/rd)/90.+real(ix0))
      iyt=nint(-r0*6.5*(90.-tplng)*cos(tazim/rd)/90.+real(iy0))
      do 7 i=ixt-1,ixt+1
      do 6 j=iyt-1,iyt+1
      ach(j,i)=ablank
    6 continue
    7 continue
      ach(iyt,ixt)=ataxis
c
c ...... add fps plot
c
      do 8 i=1,2*iy0-2
      write(unit,'(5x,72a1)') (ach(i,j),j=1,2*ix0)
    8 continue
c
      return
      end
C+
      real*4 function polar(g,aoi,theta,first)
c
c ...... compute first motion podsretc_CMT.lstlarity as a function of aoi & theta
c        for a moment tensor for a double-couple solution.
C	Conventions differ slightly from Sipkin.  My moment tensor is derived
C	  from the outer product of two vectors and is hence normalized.  The
C	  order is also different from his, apparently.  I also did not know
C	  cosd and sind existed.
C-
      dimension g(6)
      real mxx,mxy,mxz,myy,myz,mzz
      logical first
c
      save
      rd = 45.0/atan(1.0)
      if(first) then
        mxx= g(2)
        mxy=-g(6)
        mxz= g(4)
        myy= g(3)
        myz=-g(5)
        mzz= g(1)
        first = .false.
      endif
	x = cos(theta/rd)*sin(aoi/rd)
	y = sin(theta/rd)*sin(aoi/rd)
	z = cos(aoi/rd)
c
      polar = x*mxx*x + 2*x*mxy*y + 2*x*mxz*z + 2*y*myz*z +y*myy*y
     1        +z*mzz*z
c
      return
      end

        