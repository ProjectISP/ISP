"""
     Obtiene los .mseed entre las fechas indicadas
"""

from scRoa import seiscompConnector as SC

# Conexiones BBDD y SFTP

def refilt(result):
    #time = ['2022-10-01 01:45:07' , '2022-10-01 07:04:13']
    #latitude = [39.696, 39.64]
    #longitude = [-1.194, -1.143]
    #depth = [10, 12]

    coords = [['2022-10-01 01:45:07',39.696,-1.194,10],['2022-10-01 07:04:13',39.64,-1.143,12]]
    result_new = []
    for item in result:
        item_test = [item['time'], item['latitude'], item['longitude'], item['depth']]
        for coord_test in coords:
            if coord_test == item_test:
                result_new.append(item)

        else:
            pass

    return result_new


cfg = {'hostname': 'alertes.roa.es',
     'dbname': 'seiscomp3',
     'user': 'sysop',
     'password': 'sysop',
     'sdshost': 'alertes.roa.es',
     'sdsuser': 'sysop',
     'sdspass': 'Oku7ygr$$roa',
     'sdsdir': '/home/sysop/seiscomp3/var/lib/archive/',
     'sdsout': '/Users/robertocabieces/Documents/desarrollo/ISP2021/Test/scconnector/archiveTest1/'}

# Declaracion clase seiscompConnector
sc = SC()

# Inicio clase seiscompConnector
sc.init(**cfg)
print("conexion")
# Obtener .mseed sin filtro

if not sc.checkfile():
       filter = {'network': 'WM'}
       result = sc.find('2022-10-01 00:00:00', '2022-10-02 08:00:00')
       result = refilt(result)
       #result = sc.find('2022-08-01 00:00:00', '2022-08-02 08:00:00', **filter)
       for r in result:
           print("Result: ", r)
#
#     sc.download(result)