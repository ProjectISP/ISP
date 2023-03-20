C+
	SUBROUTINE PLTLAB(XLNGTH,YLNGTH)
C
C	ADD A LABEL TO A PLOT SETTING POSITION VIA THE HP OR TEK
C		 CURSOR OR BY TELLING IT WHERE TO GO
C
C	sun version.  jas/vtso 13 September 1991
C	15 June 1992:  Latest perturbation
C	3 January 2003:  No graphics terminal for this version
C-
        common /jasplot/  jplot,jdummy
        character*16 device
	common /jasgtplot/xgtorgn,ygtorgn,xgt,ygt,scale_factor,device
	character*120 com, cvalue, dummy
	LOGICAL TRUTH
	DATA IPEN,HEIGHT,ANGLE/2,0.2,0.0/
	call linestyle(1)
25	CONTINUE
	  write(*,4) XLNGTH,YLNGTH
    4   FORMAT(' The plot field is ',F4.1,' inches wide and ',F4.1,
     .    ' inches high')
        call printx('X and Y values for one corner of label field')
        read(*,*) xstart,ystart
	com = cvalue('Label - up to 120 characters',dummy,n)
	WRITE(*,5) IPEN,HEIGHT,ANGLE
5     FORMAT(1H ,'Width =',I2,', Height =',F6.3,', Angle =',F6.1)
	IF(TRUTH('Change these?')) THEN
	  IPEN = IVALUE('linewidth value [2]',2)
	  HEIGHT = RVALUE('Height of label field in inches [0.2]',0.2)
	  ANGLE = RVALUE('Angle with horizontal [0]',0.0)
	ENDIF
	CALL linewidth(IPEN)
	IF (.NOT.TRUTH('Cursor in lower left-hand corner? [Y]')) THEN
	  ANGR = ANGLE/57.29578
	  COSANG = COS(ANGR)
	  SINANG = SIN(ANGR)
	  IF (TRUTH('Upper left? [Y]')) THEN
	    YSTART = YSTART - HEIGHT*COSANG
	    XSTART = XSTART + HEIGHT*SINANG
	  ELSE
	    CALL LENGTH(XLEN,HEIGHT,COM(1:N),N)
	    XSTART = XSTART - XLEN*COSANG
	    YSTART = YSTART - XLEN*SINANG
	    IF (TRUTH('Upper right? [Y]')) THEN
 	      YSTART = YSTART - HEIGHT*COSANG
	      XSTART = XSTART + HEIGHT*SINANG
	    ENDIF  
	  ENDIF
	ENDIF
	write(*,*) 'Lower left:  X:',XSTART,'   Y:',YSTART
	CALL SYMBOL(XSTART,YSTART,HEIGHT,COM(1:N),ANGLE,N)
	CALL PLOT(XSTART,YSTART,0)
	IF (TRUTH('More labels?')) GO TO 25
	RETURN
	END
