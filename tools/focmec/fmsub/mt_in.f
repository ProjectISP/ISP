C+
	subroutine mt_in(pttp,PI)
C
C	eigenvalues/vectors using EISPACK routines from www.netlib.no
C	Much of code adapted from Jost/Herrmann mteig.f and mtdec.f
C	Uses original EISPACK routines for TRED2 and IMTQL2, not NR
C	Also includes subroutine eig, which calls TRED2 and IMTQL2.
C	15 March 2002
C-
	dimension A(3,3), U(3,3), W(3), PTTP(4), XYZ(3)
	real*4 MRR, MTT, MPP, MRT, MRP, MTP, isotrop
      write(*,*) 'Input MRR MTT MPP MRT MRP MTP (free format)'
      read(*,*) MRR, MTT, MPP, MRT, MRP, MTP
      write(2,*) '  Input is moment tensor (Dziewonski convention)'
      write(2,*) 'MRR MTT MPP MRT MRP MTP'
      write(2,'(1p6g11.4)') MRR, MTT, MPP, MRT, MRP, MTP
C
C	Convention is X north, Y east, Z down
C
      A(3,3) = MRR
      A(1,1) = MTT
      A(2,2) = MPP
      A(1,3) = MRT
      A(3,1) = A(1,3)
      A(2,3) = -MRP
      A(3,2) = A(2,3)
      A(1,2) = -MTP
      A(2,1) = A(1,2)
      call eig(a,u,w)
C
C	Ordering returned is from smallest to highest (P, B, T for DC)
C
C      write(*,*) ' '
C     write(*,*) 'EIGENVALUES                EIGENVECTORS'
C     do j=1,3
C       write(*,'(1pg11.4,2x,0p5f11.4)') W(j),(U(i,j),i=1,3)
C     end do
      isotrop = 0.0
      do j=1,3
        isotrop = isotrop + w(j)
      end do
      devmom = 0.0
      do j=1,3
        w(j) = w(j) - isotrop/3.0
	devmom = devmom + w(j)*w(j)
      end do
      devmom = sqrt(0.5*devmom)
      if (devmom .lt. 0.001*isotrop) devmom = 0.0
      write(*,*) ' '
      write(*,'(a,1pg11.4,a,g11.4)') 
     1	'  Trace of moment tensor = ',isotrop,
     2  '  Deviatoric tensor moment = ', devmom
      write(2,*) ' '
      write(2,'(a,1pg11.4,a,g11.4)') 
     1	'  Trace of moment tensor = ',isotrop,
     2  '  Deviatoric tensor moment = ', devmom
      if (devmom .eq. 0.0) then
        write(2,*) 'Exiting because purely isotropic source'
	write(*,*) 'Exiting because purely isotropic source'
	stop
      end if
      write(2,'(a,1p3g11.4)') '  Deviatoric moment tensor eigenvalues:',
     1   (w(j),j=1,3)
      write(*,'(a,1p3g11.4)') '  Deviatoric moment tensor eigenvalues:',
     1   (w(j),j=1,3)
c---- Dziewonski, Chou, Woodhouse,  JGR 1981 2825-2852
c---- eps=0 pure double couple
c---- eps=0.5 pure CLVD
      eps = abs(w(2)/amax1(-w(1),w(3)))
      eps1 = eps*200.0
      eps2 = 100.0 - eps1
      write(*,'(a)') '  EPSILON    % OF CLVD     % OF DC'
      write(*,'(f9.4,2f12.4)') eps, eps1, eps2
      write(2,'(a)') '  EPSILON    % OF CLVD     % OF DC'
      write(2,'(f9.4,2f12.4)') eps, eps1, eps2
      write(2,*) ' '
      if (eps .ge. 0.25) then
        write(2,*) ' Exiting because less than 50% double couple'
	write(*,*) ' Exiting because less than 50% double couple'
	stop
      end if
C
C	Get trend and plunge for P
C
      do j=1,3
        xyz(j) = u(j,1)
      end do
      call V2TRPL(XYZ,PTTP(1),PI)
C
C	Get trend and plunge for T
C
      do j=1,3
        xyz(j) = u(j,3)
      end do
      call V2TRPL(XYZ,PTTP(3),PI)
C     do j=1,4
C        pttp(j) = rdeg*pttp(j)
C      end do
C      write(*,*) '  '
C      write(*,*) '  Trend and Plunge of P and T'
C     write(*,'(4f11.4)') pttp
      return
      end
      subroutine eig (a,u,w)
      dimension A(3,3), U(3,3), W(3), work(3)
      np = 3
      do i=1,np
        do j=1,3
	  u(i,j) = a(i,j)
	end do
      end do
      n = 3
      call tred2(np,n,a,w,work,u)
      call imtql2(np,n,w,work,u,ierr)
C
C	This system has P, B, T as a right-hand coordinate system.
C	I prefer P, T, B
C
      do j=1,3
        u(j,1) = -u(j,1)
      end do
      return
      end
c---------------
      subroutine tred2(nm,n,a,d,e,z)
c
      integer i,j,k,l,n,nm
      real a(nm,n),d(n),e(n),z(nm,n)
      real f,g,h,hh,scale
c
c     this subroutine is a translation of the algol procedure tred2,
c     num. math. 11, 181-195(1968) by martin, reinsch, and wilkinson.
c     handbook for auto. comp., vol.ii-linear algebra, 212-226(1971).
c
c     this subroutine reduces a real symmetric matrix to a
c     symmetric tridiagonal matrix using and accumulating
c     orthogonal similarity transformations.
c
c     on input
c
c        nm must be set to the row dimension of two-dimensional
c          array parameters as declared in the calling program
c          dimension statement.
c
c        n is the order of the matrix.
c
c        a contains the real symmetric input matrix.  only the
c          lower triangle of the matrix need be supplied.
c
c     on output
c
c        d contains the diagonal elements of the tridiagonal matrix.
c
c        e contains the subdiagonal elements of the tridiagonal
c          matrix in its last n-1 positions.  e(1) is set to zero.
c
c        z contains the orthogonal transformation matrix
c          produced in the reduction.
c
c        a and z may coincide.  if distinct, a is unaltered.
c
c     Questions and comments should be directed to Alan K. Cline,
c     Pleasant Valley Software, 8603 Altus Cove, Austin, TX 78759.
c     Electronic mail to cline@cs.utexas.edu.
c
c     this version dated january 1989. (for the IBM 3090vf)
c
c     ------------------------------------------------------------------
c
c?      call xuflow(0)
      do 100 i = 1, n
         do 100 j = i, n
  100       z(j,i) = a(j,i)
c
      do 110 i = 1, n
  110    d(i) = a(n,i)
c
      do 300 i = n, 2, -1
         l = i - 1
         h = 0.0e0
         scale = 0.0e0
         if (l .lt. 2) go to 130
c     .......... scale row (algol tol then not needed) ..........
         do 120 k = 1, l
  120    scale = scale + abs(d(k))
c
         if (scale .ne. 0.0e0) go to 140
  130    e(i) = d(l)
c
c"    ( ignore recrdeps
         do 135 j = 1, l
            d(j) = z(l,j)
            z(i,j) = 0.0e0
            z(j,i) = 0.0e0
  135    continue
c
         go to 290
c
  140    do 150 k = 1, l
            d(k) = d(k) / scale
            h = h + d(k) * d(k)
  150    continue
c
         f = d(l)
         g = -sign(sqrt(h),f)
         e(i) = scale * g
         h = h - f * g
         d(l) = f - g
c     .......... form a*u ..........
         do 170 j = 1, l
  170       e(j) = 0.0e0
c
         do 240 j = 1, l
            f = d(j)
            z(j,i) = f
            g = e(j) + z(j,j) * f
c
            do 200 k = j+1, l
               g = g + z(k,j) * d(k)
               e(k) = e(k) + z(k,j) * f
  200       continue
c
            e(j) = g
  240    continue
c     .......... form p ..........
         f = 0.0e0
c
         do 245 j = 1, l
            e(j) = e(j) / h
            f = f + e(j) * d(j)
  245    continue
c
         hh = -f / (h + h)
c     .......... form q ..........
         do 250 j = 1, l
  250       e(j) = e(j) + hh * d(j)
c     .......... form reduced a ..........
         do 280 j = 1, l
            f = -d(j)
            g = -e(j)
c
            do 260 k = j, l
  260          z(k,j) = z(k,j) + f * e(k) + g * d(k)
c
            d(j) = z(l,j)
            z(i,j) = 0.0e0
  280    continue
c
  290    d(i) = h
  300 continue
c     .......... accumulation of transformation matrices ..........
      do 500 i = 2, n
         l = i - 1
         z(n,l) = z(l,l)
         z(l,l) = 1.0e0
         h = d(i)
         if (h .eq. 0.0e0) go to 380
c
         do 330 k = 1, l
  330       d(k) = z(k,i) / h
c"    ( ignore recrdeps
c"    ( prefer vector
         do 360 j = 1, l
            g = 0.0e0
c
            do 340 k = 1, l
  340          g = g + z(k,i) * z(k,j)
c
            g = -g
c
            do 350 k = 1, l
  350          z(k,j) = z(k,j) + g * d(k)
  360    continue
c
  380    do 400 k = 1, l
  400       z(k,i) = 0.0e0
c
  500 continue
c
c"    ( prefer vector
      do 520 i = 1, n
         d(i) = z(n,i)
         z(n,i) = 0.0e0
  520 continue
c
      z(n,n) = 1.0e0
      e(1) = 0.0e0
      return
      end
      subroutine imtql2(nm,n,d,e,z,ierr)
c
      integer i,j,k,l,m,n,nm,ierr
      real d(n),e(n),z(nm,n)
      real b,c,f,g,p,r,s,tst1,tst2
c
c     this subroutine is a translation of the algol procedure imtql2,
c     num. math. 12, 377-383(1968) by martin and wilkinson,
c     as modified in num. math. 15, 450(1970) by dubrulle.
c     handbook for auto. comp., vol.ii-linear algebra, 241-248(1971).
c
c     this subroutine finds the eigenvalues and eigenvectors
c     of a symmetric tridiagonal matrix by the implicit ql method.
c     the eigenvectors of a full symmetric matrix can also
c     be found if  tred2  has been used to reduce this
c     full matrix to tridiagonal form.
c
c     on input
c
c        nm must be set to the row dimension of two-dimensional
c          array parameters as declared in the calling program
c          dimension statement.
c
c        n is the order of the matrix.
c
c        d contains the diagonal elements of the input matrix.
c
c        e contains the subdiagonal elements of the input matrix
c          in its last n-1 positions.  e(1) is arbitrary.
c
c        z contains the transformation matrix produced in the
c          reduction by  tred2, if performed.  if the eigenvectors
c          of the tridiagonal matrix are desired, z must contain
c          the identity matrix.
c
c      on output
c
c        d contains the eigenvalues in ascending order.  if an
c          error exit is made, the eigenvalues are correct but
c          unordered for indices 1,2,...,ierr-1.
c
c        e has been destroyed.
c
c        z contains orthonormal eigenvectors of the symmetric
c          tridiagonal (or full) matrix.  if an error exit is made,
c          z contains the eigenvectors associated with the stored
c          eigenvalues.
c
c        ierr is set to
c          zero       for normal return,
c          j          if the j-th eigenvalue has not been
c                     determined after 30 iterations.
c
c     Questions and comments should be directed to Alan K. Cline,
c     Pleasant Valley Software, 8603 Altus Cove, Austin, TX 78759.
c     Electronic mail to cline@cs.utexas.edu.
c
c     this version dated january 1989. (for the IBM 3090vf)
c
c     ------------------------------------------------------------------
c
c?      call xuflow(0)
      ierr = 0
      if (n .eq. 1) go to 1001
c
      do 100 i = 2, n
  100 e(i-1) = e(i)
c
      e(n) = 0.0e0
c
      do 240 l = 1, n
         j = 0
c     .......... look for small sub-diagonal element ..........
  105    do 110 m = l, n-1
            tst1 = abs(d(m)) + abs(d(m+1))
            tst2 = tst1 + abs(e(m))
            if (tst2 .eq. tst1) go to 120
  110    continue
c
  120    p = d(l)
         if (m .eq. l) go to 240
         if (j .eq. 30) go to 1000
         j = j + 1
c     .......... form shift ..........
         g = (d(l+1) - p) / (2.0e0 * e(l))
cccccccccccccccccccccccccccccccccccccccccccccccccccc
c *      r = pythag(g,1.0d0)
cccccccccccccccccccccccccccccccccccccccccccccccccccc
         if (abs(g).le.1.0e0) then
            r = sqrt(1.0e0 + g*g)
         else
            r = g * sqrt(1.0e0 + (1.0e0/g)**2)
         endif
cccccccccccccccccccccccccccccccccccccccccccccccccccc
         g = d(m) - p + e(l) / (g + sign(r,g))
         s = 1.0e0
         c = 1.0e0
         p = 0.0e0
c     .......... for i=m-1 step -1 until l do -- ..........
         do 200 i = m-1, l, -1
            f = s * e(i)
            b = c * e(i)
cccccccccccccccccccccccccccccccccccccccccccccccccccc
c *         r = pythag(f,g)
cccccccccccccccccccccccccccccccccccccccccccccccccccc
            if (abs(f).ge.abs(g)) then
               r = abs(f) * sqrt(1.0e0 + (g/f)**2)
            else if (g .ne. 0.0e0) then
               r = abs(g) * sqrt((f/g)**2 + 1.0e0)
            else
               r = abs(f)
            endif
cccccccccccccccccccccccccccccccccccccccccccccccccccc
            e(i+1) = r
            if (r .eq. 0.0e0) then
c     .......... recover from underflow ..........
               d(i+1) = d(i+1) - p
               e(m) = 0.0e0
               go to 105
            endif
            s = f / r
            c = g / r
            g = d(i+1) - p
            r = (d(i) - g) * s + 2.0e0 * c * b
            p = s * r
            d(i+1) = g + p
            g = c * r - b
c     .......... form vector ..........
            do 180 k = 1, n
               f = z(k,i+1)
               z(k,i+1) = s * z(k,i) + c * f
               z(k,i) = c * z(k,i) - s * f
  180       continue
c
  200    continue
c
         d(l) = d(l) - p
         e(l) = g
         e(m) = 0.0e0
         go to 105
  240 continue
c     .......... order eigenvalues and eigenvectors ..........
      do 300 i = 1, n-1
         k = i
         p = d(i)
c
         do 260 j = i+1, n
            if (d(j) .ge. p) go to 260
            k = j
            p = d(j)
  260    continue
c
         d(k) = d(i)
         d(i) = p
c
         do 280 j = 1, n
            p = z(j,i)
            z(j,i) = z(j,k)
            z(j,k) = p
  280    continue
c
  300 continue
c
      go to 1001
c     .......... set error -- no convergence to an
c                eigenvalue after 30 iterations ..........
 1000 ierr = l
 1001 return
      end
