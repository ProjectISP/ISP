import mysql.connector
from mysql.connector import Error

class sds_database:

    def __init__(self, **kwargs):

        self._connection = None
        self._hostname = None
        self._user = None
        self._password = None
        self._dbname = None

        if 'host' not in kwargs.keys():
            raise ValueError('Database hostname is mandatory')
        else:
            self._hostname = kwargs['host']

        if 'name' not in kwargs.keys():
            raise ValueError('Database name is mandatory')
        else:
            self._dbname = kwargs['name']

        if 'user' in kwargs.keys():
            self._user = kwargs['user']

        if 'user' not in kwargs.keys() and 'password' in kwargs.keys():
            raise KeyError('password is needed')
        elif 'user' in kwargs.keys() and 'password' in kwargs.keys():
            self._password = kwargs['password']

    def connect(self):
        connection = mysql.connector.connect()

        if self._user is None:
            try:
                connection = mysql.connector.connect(
                    host=self._hostname,
                    database=self._dbname
                )

                #print("MySQL Database connection successful")

            except Error as err:
                print(f"Error: '{err}'")

        elif self._user is not None:
            try:
                connection = mysql.connector.connect(
                    host=self._hostname,
                    user=self._user,
                    passwd=self._password,
                    database=self._dbname
                )

                #print("MySQL Database connection successful")

            except Error as err:
                print(f"Error: '{err}'")

        return connection

    def close(self, connection):
        return connection.close()