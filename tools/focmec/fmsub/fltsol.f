C+
C	SUBROUTINE FLTSOL(A,N,BMATRX,PLUNGE,TREND,ANGLE,JA)
C
C	Called by SRCHFM, a subroutine of FOCMEC
C	Calculates the A and N vectors (Herrmann's X and Y)
C		for a given trend and plunge for the B axis
C		and ANGLE, which is the angle A makes with the B
C               trend in the plane perpendicular to B.  If ANGLE=0,
C		A and B have the same trend.
C	Rotations obey the right-hand rule.
C
C	Arthur Snoke  Virginia Tech  May 1984
C	June 2009: Added comments to explain more fully the operations
C       June 2014: Tweaked the comments, changed YROT, and moved A to
C         N and N to -A to get FOCMEC to use the same fault plane as
C         input P and T (usually).
C-
	SUBROUTINE FLTSOL(A,N,BMATRX,PLUNGE,TREND,ANGLE,JA)
	INCLUDE 'FOCMEC.INC'
	REAL*4 A(3), N(3), BMATRX(3,3),ZROT(3,3),YROT(3,3),ANB(3,3)
	IF (JA .EQ. 1) THEN
		DO J=1,3
                    DO K=1,3
                        ZROT(J,K) = 0.0
                        YROT(J,K) = 0.0
                    end do
                end do
C
C	Construct a rotation matrix to map X (North), Y (East),
C		Z (Down) into A, N, B.  Input is B and an angle ANGLE
C
C	First rotate about Z (= Down) through an angle TREND.
C		X now has the trend of B
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
C		direction and the B trend,  the rotated X axis will
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
c            write(2,*) 'bmatrx',bmatrx
C
C	BMATRX does not change as ANGLE changes, so only needs to be
C           once for a given TREND and PLUNE and all values of ANGLE
C
	ENDIF
C
C	Rotate about Z (= B) through an angle ANGLE.
C		This rotates X into A and Y into N.
C
	ZROT(1,1) = COS(ANGLE)
	ZROT(2,2) = ZROT(1,1)
	ZROT(1,2) = SIN(ANGLE)
	ZROT(2,1) = -ZROT(1,2)
	ZROT(3,3) = 1.0
C
C	ANB is the product of ZROT and BMATRX
C
	CALL GMPRD(ZROT,BMATRX,ANB,3,3,3)
	DO J=1,3
                A(J) = -ANB(2,J)!  -A is 2nd row of ANB
                N(J) = ANB(1,J)	!  N is 1st row of ANB
	end do
C
C   A => N and -N => A to get the desired fault plane so FOCMEC
C     agrres with focal mechanisms based on P and T input (usually)
C
c         write(2,*) 'a ',a
c         write(2,*) 'n ',n
c         write(2,*) 'b ',(anb(3,j),j=1,3)
c         write(2,*) 'x ',(bmatrx(1,j),j=1,3)
        RETURN
	END
