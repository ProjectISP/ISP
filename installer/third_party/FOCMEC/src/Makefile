#  Makefile  for creating executables in FOCMEC package.  FCMP is set on entry

LIB = ../lib/libiasp.a ../lib/libfm.a

all:	convert_input dsretc focmec_prep focmec focplt freesurf \
hypo2foc radiat ratio_prep ratio_prep_1layer ratplt sgftops symdemo synth_test vwbeetle

clean:
	cd ../bin; \rm -f convert_input dsretc focmec_prep focmec focplt freesurf \
hypo2foc radiat ratio_prep ratio_prep_1layer ratplt sgftops symdemo synth_test ttimes vwbeetle; \
cd ../lib; \rm -f ak135.hed ak135.tbl iasp91.hed iasp91.tbl libfm.a libiasp.a

convert_input:	convert_input.f
	$(FCMP) convert_input.f $(LIB) -C -o ../bin/convert_input

dsretc:	dsretc.f
	$(FCMP) dsretc.f $(LIB) -C -o ../bin/dsretc

focmec_prep: focmec_prep.f
	$(FCMP) focmec_prep.f $(LIB) -C -o ../bin/focmec_prep

focmec:	focmec.f
	$(FCMP) focmec.f $(LIB) -C -o ../bin/focmec

focplt: focplt.f
	$(FCMP) focplt.f $(LIB) -C -o ../bin/focplt

freesurf:	freesurf.f
	$(FCMP) freesurf.f $(LIB) -C -o ../bin/freesurf

hypo2foc:	hypo2foc.f
	$(FCMP) hypo2foc.f $(LIB) -C -o ../bin/hypo2foc

radiat:	radiat.f
	$(FCMP) radiat.f $(LIB) -C -o ../bin/radiat

ratio_prep: ratio_prep.f
	$(FCMP) ratio_prep.f $(LIB) -C -o ../bin/ratio_prep

ratio_prep_1layer: ratio_prep_1layer.f
	$(FCMP) ratio_prep_1layer.f $(LIB) -C -o ../bin/ratio_prep_1layer

ratplt:	ratplt.f
	$(FCMP) ratplt.f $(LIB) -C -o ../bin/ratplt

sgftops: sgftops.c 
	gcc -o ../bin/sgftops sgftops.c
	
symdemo: symdemo.f
	$(FCMP) symdemo.f $(LIB) -C -o ../bin/symdemo
	
synth_test: synth_test.f
	$(FCMP) synth_test.f $(LIB) -C -o ../bin/synth_test

vwbeetle: vwbeetle.f
	$(FCMP) vwbeetle.f $(LIB) -C -o ../bin/vwbeetle
