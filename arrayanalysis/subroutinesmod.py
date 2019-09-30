# -*- coding: utf-8 -*-
def metric(st):
    import util as ut
    import numpy as np
    import scipy as sp
    '''
    function takes data matrix and returns interstation distances rx and ry (vectors) in [deg].
    '''
    nr = len(st)
    rx = np.zeros(nr)
    ry = np.zeros(nr)
    for i in range(nr):
        decl,dist,az,baz = ut.grt(st[0].stats.sac.stla,st[0].stats.sac.stlo,st[i].stats.sac.stla,st[i].stats.sac.stlo)
        rx[i] = decl*sp.cos(0.017453*(90.0-az))
        ry[i] = decl*sp.sin(0.017453*(90.0-az))
    return rx,ry
    


def get_metadata(meta_f):
    d = dict()
    with open(meta_f) as f:
        for line in f:
            x = line.split('|')
            d[x[1]] = x[4],x[5]
    return d

def metric_mseed(st,d,nr):
    import util as ut
    import numpy as np
    import scipy as sp
    '''
    function takes data matrix and returns interstation distances rx and ry (vectors) in [deg].
    '''
    rx_0,ry_0 = d[st[0].stats.station]
    rx = np.zeros(nr)
    ry = np.zeros(nr)
    for i in range(nr):
        rx_i,ry_i = d[st[i].stats.station]
        decl,dist,az,baz = ut.grt(float(rx_0),float(ry_0),float(rx_i),float(ry_i))
        rx[i] = decl*sp.cos(0.017453*(90.0-az))
        ry[i] = decl*sp.sin(0.017453*(90.0-az))
    return rx,ry
    

    
def testfir(st,cb,ct,n):
    from scipy import signal
    import numpy as np
    
    sr = st[0].stats.sampling_rate/2.
    xx = np.empty([st.count(),n],)
    a = signal.firwin(n, cutoff = cb/sr, window = 'hamming')
    b = - signal.firwin(n, cutoff = ct/sr, window = 'hamming'); b[n/2] = b[n/2] + 1
    d = - (a+b); d[n/2] = d[n/2] + 1
    fft1 = np.abs(np.fft.fft(d))
    for i in range(st.count()):
        fft = np.fft.fft(st[i][:n])*fft1
        xx[i] = np.fft.ifft(fft)
    return xx
    
def find_nearest(array, value):
    import numpy as np
    #array = np.asarray(array)
    #idx = (np.abs(array - value)).argmin()
    idx,val = min(enumerate(array), key=lambda x: abs(x[1]-value))
    return idx,val

    
def print_stats(fk,smin,smax,sinc,threshold):
    import numpy as np
    import scipy as sp
    import scipy.ndimage.filters as filters

    s = np.arange(smin,smax+sinc,sinc)
    tmp = []
    print('---------------------------')
    print('--- Arrival Information ---')
    print('---------------------------')
    print("")
    print('normalized power (dB)   ', 'velocity (km/s)   ', 'backazimuth (deg)')
    maxxi = (np.where(fk==filters.maximum_filter(fk, 5)))
    this=np.empty([2,len(maxxi[0])])
    lth = np.amin(fk)*threshold
    for i in range(len(maxxi[0])):
        this[0][i]=s[maxxi[0][i]]
        this[1][i]=s[maxxi[1][i]]
        if (fk[maxxi[0][i],maxxi[1][i]] > lth):
            baz=np.math.atan2(this[0][i],this[1][i])*180.0/3.1415926
            if(baz<0.0):
                baz+=360.0
            xvel = 111.19/sp.sqrt(this[0][i]**2+this[1][i]**2)
            xamp = fk[maxxi[0][i],maxxi[1][i]]
            tmp.append([xamp,xvel,baz])
    tmp.sort(reverse=True)
    for i in range(len(tmp)):
        #print('%12.02f %19.02f %19.02f'%(tmp[i][0],tmp[i][1],tmp[I][2]))  
          print('%12.02f %19.02f %19.02f'%(tmp[i][0],tmp[i][1],tmp[i][2]))