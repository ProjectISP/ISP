#  Makefile for building library libfm.a   2008 version includes plot stuff
#  July 2008: Built using gfortran (version 4.3) on both OSX platforms.
#  Assumes FCMP is already set
FFLAG =
#  FCMP = g77
#  FCMP = f77 
  FCMP = gfortran -fbackslash
#  f77 has been tested folr Sun Solaris, g77 and gfortran for Mac OSX,
#  g77 has been tested for Linux  For gfortran, need 4.2 or higher to
#  be compatible for the .hed files with g77 and f77. 

.f.o:
	$(FCMP) $(FFLAG) -c $<

LIB =	../lib/libfm.a

OBJECTS = an2dsr.o \
	an2mom.o \
	cstring.o \
	cvalue.o \
	dsrin.o \
	fltsol.o \
	fmreps.o \
	focinp.o \
	focreps.o \
	getstring.o \
	gmprd.o \
	grndmtn.o \
	ivalue.o \
	iyesno.o \
	lenc.o \
	lratio.o \
	maxmin.o \
	minmax.o \
	mt_in.o \
	okpol.o \
	okrat.o \
	oksol.o \
	printx.o \
	pttpin.o \
	radpat.o \
	rvalue.o \
	srchfm.o \
	timdat.o \
	trpl2v.o \
	truth.o \
	v2trpl.o \
	value.o \
	cirplt.o \
	linestyle.o \
	linewidth.o \
	mylabl.o \
	plnplt.o \
	plot.o \
	plotdf.o \
	plots.o \
	pltdat.o \
	pltlab.o \
	pltsdf.o \
	projkt.o \
	polplot.o \
	ptbplt.o \
	r8tor4.o \
	rclip.o \
	shnode.o \
	shnsrf.o \
	solplt.o \
	svnode.o \
	svnsrf.o \
	symbol.o \
	ta2xyi.o \
	tsend.o \
      uppercase.o \
	where.o \
	wsgfbuf.o \
	xx.o \
	xy.o \
	buf2dsk.o

libfm:		${OBJECTS}
	ar rv ${LIB} ${OBJECTS}; ranlib ${LIB}
