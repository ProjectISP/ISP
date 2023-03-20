C+
C	PROGRAM FOCMEC
C
C	Calculates acceptable focal mechanisms based on polarities
C	  and (S/P) amplitude ratios.
C	Inspired by the Georgia Tech program with statistics
C	  from Kisslinger's.  Conventions are those in
C	  Herrmann as well as Aki & Richards
C	Updates are documented in a separate file.
C
C	Arthur Snoke   Virginia Tech  1984, 2001, 2014
C-
	CALL FOCINP
	CALL SRCHFM
	STOP
	END
