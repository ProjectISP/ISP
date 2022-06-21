c+
        Program synth_test
C
C   Input a dip, strike, slip, P to, S take-off aangles
C   Output is a FOCMEC input file with P, SV, SH polarities
C       and SV/P, SH/P, SH/SV amplitude rtios.  Azimuths are
C       0, 24, 48/, ..., 336.  Comment line is dip, strike, slip.
C       Output file is named symth_focmec.inp.  A second output file
C       is synthtest.lst with radiation terms.
C-
        character*80 title
        character*4 sta
        character*1 sense,sense_num
        real*4 aps(2),angs(3)
	RD = 45.0/ATAN(1.0)
c
        open(2,file='synth_test.lst',status='unknown')
        open(3,file='synth_focmec.inp',status='unknown')
        CALL PRINTX('Enter Dip, Strike and slip (degrees)')
        READ(*,*) (ANGS(J),J=1,3)
	D = angs(1)/RD
	ST = angs(2)/RD
	SL = angs(3)/RD
        call printx('P and S take-off angles (reltive to down)')
        read(*,*) ptoa, stoa
	APS(1) = PTOA/RD
	APS(2) = STOA/RD
        vpvs = rvalue('vP/vS [1.7321]',1.7321)
        write(2,*) 'dip ',angs(1),'; strike ',angs(2),';  slip ',angs(3)
        write(2,*) 'vP/vS ',vpvs,'; Ptoa ',ptoa,'; S toa',stoa
        write(3,121)'dip',angs(1),'; strike',angs(2),'; slip',angs(3),
     1      '; vP/vS',vpvs
        vpvs3 = vpvs*vpvs*vpvs
c
        azd = -24.0
        do j=1,15
          azd = azd + 24.0
          az = azd/rd
	  CALL radpat(D,ST,SL,AZ,APS,rp,rsv,rsh)
          WRITE(2,120) 'Azimuth ',AZD,'; P ',rp,'; SV ',rsv,'; SH ',rsh
          sta='XXXX'
          write(sta(3:4),'(i2.2)') j
          if(rp.ge.0.)sense='C'
          if(rp.lt.0.)sense='D'
          if(abs(rp).gt.1.e-3) write(3,122)sta,azd,ptoa,sense
          if(rsv.ge.0.)sense='F'
          if(rsv.lt.0.)sense='B'
          if(abs(rsv).gt.1.e-3)write(3,122)sta,azd,stoa,sense
          if(rsh.ge.0.)sense='L'
          if(rsh.lt.0.)sense='R'
          if(abs(rsh).gt.1.e-3)write(3,122)sta,azd,stoa,sense
c
          sense='V'    !radial sv over P amplitude
          rat=abs((rsv/rp)*vpvs3)
          if(abs(rsv).gt.1.e-3.and.abs(rp).gt.1.e-3)then
            if(rsv.ge.0.) sense_num='F'
            if(rsv.lt.0.) sense_num='B'
            write(3,123)sta,azd,stoa,sense,alog10(rat),sense_num,ptoa,
     1         ' sv/p'
          endif
          sense='H'  ! tangential SH over P amplitude
          rat=abs((rsh/rp)*vpvs3)
          if(abs(rsh).gt.1.e-3.and.abs(rp).gt.1.e-3)then
            if(rsh.ge.0.) sense_num='L'
            if(rsh.lt.0.)sense_num='R'
            write(3,123)sta,azd,stoa,sense,alog10(rat),sense_num,ptoa,
     1         ' sh/p'
          endif
          sense='S'  ! tangential SV over SH amplitude
          rat=abs((rsv/rsh))
          if(abs(rsh).gt.1.e-3.and.abs(rsv).gt.1.e-3)then
            if(rsv.ge.0.) sense_num='F'
            if(rsv.lt.0.) sense_num='B'
            write(3,123)sta,azd,stoa,sense,alog10(rat),sense_num,stoa,
     1         ' sv/sh'
          endif
        enddo
c
 120  format(a,f8.3,a,1pg11.3,a,1pg11.3,a,1pg11.3)
 121  format(a,f9.2,a,f10.2,a,f10.2,a,f10.4)
 122  format(a4,2f8.2,a1)
 123  format(a4,2f8.2,a1,f8.4,1x,a1,1x,f6.2,1x,a6)
      stop
      end
