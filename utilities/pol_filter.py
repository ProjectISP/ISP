import math
import numpy as np
from obspy import read, UTCDateTime
import matplotlib.pyplot as plt
from scipy import signal
from mtspec import mtspec

def find_min_indx(s1, s2):
    a = s1-s2
    a = np.abs(a)
    min_value = np.min(a)
    value,  idx = find_nearest(a, min_value)
    idx = len(s1)-idx
    return idx

def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return a.flat[idx], idx



def pot_filt(st, a, multitaper = False):

    #Samson and Olson et al., 1981. Data adaptative Polarization Filter
    
    st_raw = st.copy()
    delta = st_raw[0].stats.delta
    dataE_raw = st_raw[0].data - np.mean(st_raw[0].data)
    dataN_raw = st_raw[1].data - np.mean(st_raw[1].data)
    dataZ_raw = st_raw[2].data - np.mean(st_raw[2].data)

    st.detrend(type="simple")
    st.taper(type="blackman", max_percentage=0.05)

    dataE = st[0].data
    dataN = st[1].data
    dataZ = st[2].data

    N = st[0].count()
    D = 2 ** math.ceil(math.log2(N))

    if multitaper:
        dataEf, freq = mtspec(dataE, delta=delta, time_bandwidth=4, number_of_tapers=4)
        dataNf, freq = mtspec(dataN, delta=delta, time_bandwidth=4, number_of_tapers=4)
        dataZf, freq = mtspec(dataZ, delta=delta, time_bandwidth=4, number_of_tapers=4)
        D = len(dataZf)
    else:
        dataEf = np.fft.fft(dataE, D) * np.conj(np.fft.fft(dataE, D))
        dataNf = np.fft.fft(dataN, D) * np.conj(np.fft.fft(dataN, D))
        dataZf = np.fft.fft(dataZ, D) * np.conj(np.fft.fft(dataZ, D))



    spec_mat = np.zeros([3, 1, D])
    spec_mat[0, 0, :] = dataEf**2
    spec_mat[1, 0, :] = dataNf**2
    spec_mat[2, 0, :] = dataZf**2

    Tr = np.sum(spec_mat, axis=0)

    spec_mat[0, 0, :] = dataEf ** 4
    spec_mat[1, 0, :] = dataNf ** 4
    spec_mat[2, 0, :] = dataZf ** 4

    Tr_2 = np.sum(spec_mat, axis=0)

    num = 3 * Tr_2 - Tr ** 2
    den = 2 * Tr ** 2
    pol_f = (num / den) ** a
    if multitaper:
        pol_t = np.fft.irfft(pol_f, N)
    else:
        pol_t = np.fft.ifft(pol_f, N)

    pol_t[-1] = pol_t[0]
    data_Z_filt = signal.oaconvolve(dataZ_raw, pol_t[0, :], "full")
    data_E_filt = signal.oaconvolve(dataE_raw, pol_t[0, :], "full")
    data_N_filt = signal.oaconvolve(dataN_raw, pol_t[0, :], "full")
    data_Z_filt = data_Z_filt[0:int(len(dataZ))]
    data_E_filt = data_E_filt[0:int(len(dataE))]
    data_N_filt = data_N_filt[0:int(len(dataN))]
    st[0].data = np.real(data_E_filt)
    st[1].data = np.real(data_N_filt)
    st[2].data = np.real(data_Z_filt)

    return st



def pol_st(st, win_sec, a, multitaper = False):

    fs = st[0].stats.sampling_rate
    start = st[0].stats.starttime
    step_sec = int(0.3 * win_sec)
    N = int(st[0].stats.npts) # length in seconds
    half_win_sec = int(win_sec/2)
    t = np.arange(0, (N/fs)-half_win_sec, step_sec) #steps in seconds
    j = 0
    for i in t:
        if i ==t[0]:
            st1 = st.copy()
            st1.trim(starttime = start+i, endtime = start+i+win_sec)
            st2 = pot_filt(st1, a, multitaper = multitaper)
            st2_filt = st2.copy()
            st2_filt.trim(starttime = start+i+1/50, endtime = start+i+half_win_sec+step_sec)
            data_z = st2_filt[2].data
            data_n = st2_filt[1].data
            data_e = st2_filt[0].data
            del st1
            del st2
            del st2_filt
        else:

            st1 = st.copy()
            st1.trim(starttime = start+i, endtime = start+i+win_sec)
            #print(start+i)
            st2 = pot_filt(st1, a, multitaper = multitaper)
            st2_filt = st2.copy()
            st2_filt.trim(starttime = start+i+half_win_sec, endtime = start+i+half_win_sec+step_sec)
            data_z[-1] = (data_z[-1]+st2_filt[2].data[0])/2
            data_n[-1] = (data_z[-1]+st2_filt[2].data[0])/2
            data_e[-1] = (data_z[-1]+st2_filt[2].data[0])/2
            # fig, axs = plt.subplots()
            # t = np.arange(0, len(data_z), 1)
            # tt = t+step_sec*fs
            # tt = tt[len(data_z)-len(st2_filt[2].data):len(data_z)]
            # axs.plot(t,data_z, color='black', linewidth=0.75, label='Raw')
            # axs.plot(tt, st2_filt[2].data, color='red', linewidth = 0.75, label = 'Raw')
            #plt.show()
            data_z = np.hstack([data_z, st2_filt[2].data])
            data_n = np.hstack([data_n, st2_filt[1].data])
            data_e = np.hstack([data_e, st2_filt[0].data])
            del st1
            del st2
            del st2_filt
        j = j+1

    return data_z, data_n, data_e


path = "/Users/robertocabieces/Desktop/desarrollo/denoise2/adapt_pol/3Components/*.*"
st = read(path)
#st.plot()
st_c =  st.copy()
data_z_raw = st[2].data
#st.plot()
data_z, data_n, data_e = pol_st(st, 150, 6, multitaper = True)
st.filter(type="bandpass", freqmin=0.05, freqmax=1)
st_c[0].data = data_e
st_c[1].data = data_n
st_c[2].data = data_z
st_c.filter(type="bandpass", freqmin=0.05, freqmax=1)

t1 = UTCDateTime("2016-04-17T00:00:00")
t2 = UTCDateTime("2016-04-17T02:00:00")
st.trim(starttime = t1, endtime = t2)
st_c.trim(starttime = t1, endtime = t2)
st.plot(handle = True)
st_c.plot(handle = True)
plt.show()
# to check the overlap
#
# path = "/Users/robertocabieces/Desktop/desarrollo/denoise/adapt_pol/3Components/*.*"
# t1 = UTCDateTime("2016-04-17T00:09:00")
# t2 = UTCDateTime("2016-04-17T00:13:00")
# st = read(path)
# st_c =  st.copy()
# data_z, data_n, data_e = pol_st(st, 40, 4, multitaper = False)