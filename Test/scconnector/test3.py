"""
     Obtiene los .mseed usando solo como filtro
     la estacion
"""

from scRoa import seiscompConnector as SC

# Conexiones BBDD y SFTP

cfg = {'hostname': 'alertes.roa.es',
     'dbname': 'seiscomp3',
     'user': 'sysop',
     'password': 'sysop',
     'sdshost': 'alertes.roa.es',
     'sdsuser': 'sysop',
     'sdspass': 'Oku7ygr$$roa',
     'sdsdir': '/home/sysop/seiscomp3/var/lib/archive/',
     'sdsout': '/Users/robertocabieces/Documents/desarrollo/ISP2021/Test/scconnector/archiveTest1/'}

# Filtro de buqueda
filter = {'magnitude': [3, 6]}
filter = {'depth': [0, 700], 'magnitude': [2, 9]}
# Declaracion clase seiscompConnector
sc = SC()

# Inicio clase seiscompConnector
sc.init(**cfg)

# Obtener .mseed sin filtro

if not sc.checkfile():
    result = sc.find('2022-10-01 00:00:00', '2022-10-02 08:00:00', **filter)
    for r in result:
         print("Result: ", r)

    #sc.download(result)
