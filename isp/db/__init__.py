from isp.db.data_base import DataBase

db = DataBase()

"After importing db you must call db.start() and db.create_all() to initialize and create all tables into the database."

">>> db.start() "

"Before call db.create() you should import all your models models."
">>> db.create_all()"

