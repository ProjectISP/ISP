from isp.db.data_base import DataBase

db = DataBase(debug=True)
db.create_all()
