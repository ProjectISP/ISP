C+
C***************************vwbeetle**********************************
C
C     Plots the bug.  Includes option of plotting a label
C-
        REAL*4 X(2000),Y(2000)
        LOGICAL TRUTH
        open(unit=1,file='vwbeetle.dat')
C
C          Skip over title
C
           READ(1,'(A4)') DUMMY
C
C     Read in all coordinates
C
        N = 0
        IERR = 0
        DO WHILE (IERR .EQ. 0)
          READ(1,*,IOSTAT=IERR) X(N+1),Y(N+1)
          IF (IERR .EQ. 0) N = N + 1
        END DO
        close(unit=1)
        CALL MINMAX(X,1,N,XMIN,XMAX,'R4',0,0)
        CALL MINMAX(Y,1,N,YMIN,YMAX,'R4',0,0)
        XSC = 0.6
        YSC = 0.6
C
C  Initialize the plot.  Opens the plot file temp.sgf
C
        CALL PLOTS(4,15,3)
C
C  argumenst os 11, 12, prompt for input with 1, 2, the default
C  If just had linewidth(1), would not prompt but just set it to 1.
C  Same idea for the line style
C
        CALL linewidth(11)
        call linestyle(11)
C
C  CALCOMP conventions.  Moves the pen and changes the origin
C 
        CALL PLOT(1.0,1.0,-3)
        CALL PLOT(X(1)*XSC,Y(1)*YSC,3)
        DO J=2,N
            IF (X(J) + Y(J) .LE. 0.0) THEN
              CALL PLOT(X(J+1)*XSC,Y(J+1)*YSC,3)
            ELSE
              CALL PLOT(X(J)*XSC,Y(J)*YSC,2)
            ENDIF
        END DO
C
C  Flush the plot buffer (not needed)
C
        call tsend
C
C  Option for adding a plot label
C  Prompts come.  Unfortuanately no plot is produced on the screen
C  in current package, so have to iterate.
C
        if (truth('Label?')) call pltlab(xmax*xsc,ymax*ysc)
C
C  Finish plot
C
        CALL PLOT(0.,0.,999)
        STOP
        END
