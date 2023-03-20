C+
      SUBROUTINE TIMDAT(NOUT,PROGNM)
C
C      INPUT IS PROGRAM NAME
C         A LINE TO NOUT WITH DATE AND TIME AND PROGNM
C	unix version:  25 June 1991  jas/vtso
C-
      character*24 fdate
      CHARACTER*(*) PROGNM
      if (nout .le. 0) return
      write(nout,*) ' ',fdate(),' for program ',prognm(1:lenc(prognm))
      RETURN
      END
