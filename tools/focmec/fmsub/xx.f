C+
	FUNCTION XX(X,J,DELX,NPREC,XARRAY)
C
C	Called by PLTDAT
C	nonVAX version (different for double precision)
C-
	DIMENSION X(*)
	LOGICAL XARRAY
	IF (.NOT.XARRAY) THEN
	  XX = X(1) + (J-1)*DELX
	ELSE
	  if (NPREC .eq. 1) then
	    XX = X(J)
	  else
	    call r8tor4(X(2*J-1),XX)
	  end if
	END IF
	RETURN
	END
