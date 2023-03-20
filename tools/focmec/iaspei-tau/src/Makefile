#  Makefile  for creating the library and linking programs in the iaspei91
#  package.  First done jas/vt 27 October 1991
#  To run:  enter make all (or just enter make)
#    If you just want ttimes, just enter make ttimes.
#  To delete all .o and the files in ../bin, enter make clean.
#  Routine emdlv.f differs from that in the original iasp91 package in that
#    it gets the input depth/radius, Vp, and Vs from a (prompted) file
#    rather than from a data statement of polynomial coefficients.  This
#    version uses linear interpolation (as Kennett does).
#  October 2007:  The build works on Sun Solaris using Sun f77 or g77.  Using
#    g77, it works on Mac OS X both PPC and i686, and on Linux for g77.
#  June 2008: Built using gfortran (version 4.3) on both OSX platforms.
#  This Makefile assumes FCMP was set externally.
FFLAG =
#  FCMP = g77
#  FCMP = f77 
#  FCMP = gfortran
#  f77 has been tested folr Sun Solaris, g77 and gfortran for Mac OSX,
#  g77 has been tested for Linux  For gfortran, need 4.2 or higher to
#  be compatible for the .hed files with g77 and f77. 

.f.o:
	$(FCMP) $(FFLAG) -c $<

OBJECTS = libtau.o emdlv.o

all:	remodl setbrn rsetbrn ttimes

clean:
	\rm -f *.o libiasp.a ../bin/*

libiasp.a:	${OBJECTS}
	ar rv libiasp.a ${OBJECTS}
	ranlib libiasp.a

remodl:	remodl.o libiasp.a
	$(FCMP) remodl.o -C -o ../bin/remodl libiasp.a

setbrn:	setbrn.o libiasp.a
	$(FCMP) setbrn.o -C -o ../bin/setbrn libiasp.a

#  rsetbrn is the same as setbrn except that in layout all phases
#    but the primary P and S are commented out.  More almost model other
#    than the iaspei91 model setbrn bombs because of too many branches.
#    The fix is to redimension some arrays, but for my applications all I
#    need are P and S anyway, so this was an easier fix.  (One problem is
#    that the .tbl and .hed size are not cut down as they should be when one
#    makes this change.  Could be a factor if one had many models.)
#    12 March 2003:  Small changes made to remodl.f and librau.f so they
#       would compile on Linux (g77) as well as Mac OS X andSun Solaris.
#     October 2007: appended libsun.f to libtau.f 
#       (nothing that is "sun" specific).

rsetbrn:	rsetbrn.o libiasp.a
	$(FCMP) rsetbrn.o -C -o ../bin/rsetbrn libiasp.a


#  This is a slight variant on the original program.  It allows for
#    distances to be given in kilometers or degrees, and it prints results to
#    a disk file which includes information such as requested depth, 
#    and it prints out the take-off angle for each phase.

ttimes:	ttimes.o libiasp.a
	$(FCMP) ttimes.o -C -o ../bin/ttimes libiasp.a
