"""
     Obtiene los .mseed usando solo como filtro
     la red y el canal y la profuncidad
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
     'sdsout': '/home/sysop/PycharmProjects/scconnector/archiveTest4/'}

# Filtro de buqueda
filter = {'network': 'ES',
          'channel': 'HHZ',
          'depth': [10, 12]}

# Declaracion clase seiscompConnector
sc = SC()

# Inicio clase seiscompConnector
sc.init(**cfg)

# Obtener .mseed sin filtro

if not sc.checkfile():
    result = sc.find('2022-08-01 00:00:00', '2022-08-02 08:00:00', **filter)
    for r in result:
         print("Result: ", r)

    #sc.download(result)
