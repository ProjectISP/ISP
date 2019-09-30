from obspy.io.xseed import Parser
from os import listdir
from os.path import isfile, join
import pandas as pd
stalat0=[]
stalon0=[]
staelev=[]
stacall=[]
path="/Users/robertocabieces/Documents/obs_array/260/dataless"
dataless = [f for f in listdir(path) if isfile(join(path, f))]
dataless.sort()

for f in dataless:
    parser = Parser(path + "/"+ f) 
    blk = parser.blockettes
    try: 
        print(f)                      
#       coord=parser.get_coordinates(seed_id="SHZ", datetime=start1)
#        #coordinates=[coord["longitude"],coord["latitude"]]
#        stalat0.append(coordinates[1])
#        stalon0.append(coordinates[0])  
        stacall.append(blk[50][0].station_call_letters)
        stalat0.append(blk[50][0].latitude)
        stalon0.append(blk[50][0].longitude)
        staelev.append(blk[50][0].elevation)

    except:        
        pass

df = pd.DataFrame({'Name':stacall,'Lon':stalon0,'Lat':stalat0,'Depth':staelev},columns=['Name','Lon','Lat','Depth'])
df.to_csv('output2.txt',sep='\t',index = False)