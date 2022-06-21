C+
	SUBROUTINE PLOTS(NOTHNG,NFLAG,NUNIT)
C
C	PLOT INITIALIZATION FOR PRODUCING PLOTS ON
C	   (3) DEFAULT DISKFILE ONLY - NFLAG = 13
C	IF A DISK FILE IS CREATED, NUNIT IS THE LOGICAL UNIT NUMBER
C
C	SUN version.  jas/vtso  3 September 1991
C	15 June 1992:  Latest perturbation
C	2 Jan 03.  jas/vt disk file only
C-
        integer*2 line_style,line_width
        common /jasplot/  jplot,line_style,line_width
	CHARACTER*80 FILENA
	line_style = 1
	line_width = 1
	jplot = nflag - 10
      FILENA = 'temp.sgf'
      CALL PLTSDF(nunit,filena)
      RETURN
      END
