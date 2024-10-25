"""
     Obtiene los .mseed usando solo como filtro
     la red a la que pertenece la estacion
"""

from scRoa import seiscompConnector as SC

# Conexiones BBDD y SFTP

cfg = {'hostname': 'geo6.geo',
     'dbname': 'seiscomp3',
     'user': 'sysop',
     'password': 'sysop',
     'sdshost': 'geo6.geo',
     'sdsuser': 'sysop',
     'sdspass': 'Picking@@S3',
     'sdsdir': '/home/sysop/seiscomp3/var/lib/archive/',
     'sdsout': '/home/sysop/PycharmProjects/scconnector/archiveTest2/'}

# Filtro de buqueda
filter = {'network': 'WM'}

# Declaracion clase seiscompConnector
sc = SC()

# Inicio clase seiscompConnector
sc.init(**cfg)

# Obtener .mseed sin filtro

if not sc.checkfile():
    result = sc.find('2022-08-01 00:00:00', '2022-08-02 08:00:00', **filter)
    for r in result:
         print("Result: ", r)

    sc.download(result)
