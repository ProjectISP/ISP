C+
C	PROGRAM SYMDEMO
C
C  MAIN PROGRAM THAT PLOTS OUT COMPLETE SET OF SYMBOLS WITH
C  SYMBOL NUMBERS FROM THE ENHANCED SYMBOL SUBROUTINE OF
C  A.CHAVE,R.L.PARKER,AND L.SHURE
C
C	Modified for RSX/11 by Arthur Snoke, August 1982
c	modified for a SUN by David Taylor, September, 1988
C	8 January 96: jas got rid of interger array passing to symbol
C-
	character*32 ctext
	integer*4 itext(12)
	INTEGER*2 IS(19)
      DATA IS/1,27,53,73,97,121,147,173,193,217,241,267,
     $        293,313,339,365,385,409,433/
c      DATA XO,YO,H/2.0,13.0,0.25/
	data xO,yO,H/2.0,6.8,0.14/
      CALL PLOTS(53,15,2)
      IPEN = 1
      Y=YO
C  GO THROUGH THE SYMBOLS IN NUMERICAL ORDER
       DO 2000 LINE=1,18
      	Y=Y-2.0*H
      	X=XO
      	I1=IS(LINE)
      	I2=IS(LINE+1)-1
C  THE COORDINATE OF THE FIRST CHARACTER ON EACH LINE IS CALCULATED
C  EXPLICITLY, SUBSEQUENT ONES ARE FOUND BY THE ROUTINE SYMBOL WHEN
C  IT SEES X=999.
      	DO 1000 I=I1,I2
      	  I1000=I+1000
	  write(ctext(1:6),'(a,i4,a)',err=999) 
     1      '\\',I1000,'\\'
c	  write(ctext(1:6),'( ''\\'',i4,''\\'' )',err=999) I1000
          CALL symbol(X,Y,H,cTEXT,0.0,6)
      	  X=999.0
 1000		CONTINUE
 	write(ctext(1:9),'( "(",i3,"-",i3,")" )',err=999) i1,i2
       	CALL symbol(XO-7.5*H,Y+.1*H,0.6*H,cTEXT,0.0,9)
 2000	CONTINUE
	Y = Y - 0.3
      DO 4000 I=1,2
 	write(ctext(1:7),'( "(",i2,"-",i2,")" )',err=999) (i-1)*11,i*11-1
        CALL symbol(0.0,Y,0.15,cTEXT,0.0,7)
        CALL PLOT(1.5,0.0,-3)
      	X=0.0
      	DO 3000 J=0,10
      	  ITEXT(1)=11*(I-1)+J
      	  CALL spcsmb(X,Y,.15,ITEXT,0.0,-1)
      	  X=X+.3
 3000     CONTINUE
	Y = Y - 0.3
      	CALL PLOT(-1.5,0.0,-3)
4000    CONTINUE
	Y = Y - 0.3
      CALL PLOT (2.0,0.0,-3)
      cTEXT(1:4)='\\DU\\'
      cTEXT(5:8)='e\\SU'
      cTEXT(9:12)='P{-\\'
      cTEXT(13:16)='al\\}'
      cTEXT(17:20)='\\BS\\'
      cTEXT(21:24)='\\BS\\'
      cTEXT(25:28)='\\SUB'
      cTEXT(29:32)='{ij}'
      CALL symbol(0.0,Y,0.2,cTEXT,0.0,32)
      CALL PLOT(0.0,0.0,999)
999	continue
	stop
      END
