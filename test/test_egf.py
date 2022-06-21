from obspy import read
import os
name = "COR_CMPC.XT._PAPC.XT._TT"
path = "/Volumes/LaCie/Venezuela/STACK/LIN/LIN/ONEBIT"

file = os.path.join(path, name)

st = read(file)
st.filter(type="bandpass",freqmin=0.1,freqmax=1.0)
st.plot()


