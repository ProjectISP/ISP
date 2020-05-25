from isp.Gui import pyc, qt
from isp.db.models.base_model import BaseModel
import datetime

class SQLAlchemyModel(pyc.QAbstractTableModel):
    """ This class implements a Qt model for SQL Alchemy model classes
        
        :param sql_alchemy_model: Subclass of BaseModel, representing a SQL Alchemy Model
        :param columns: List of strings representing the name of the columns. Must be the attribute name.
        :param parent: Qt parent of model
    """
    def __init__(self, sql_alchemy_model, columns=[], col_names=[], parent=None):
        super().__init__(parent)

        if not issubclass(sql_alchemy_model, BaseModel):
            raise TypeError("Received model object is not a SQL Alchemy model")

        self._model = type(sql_alchemy_model) if isinstance(sql_alchemy_model, BaseModel) else sql_alchemy_model

        columns_dict = self._model().to_dict()
        if not columns:
            self._columns = columns_dict.keys()
        elif all(c in columns_dict.keys() for c in columns):
            self._columns = columns
        else:
            raise AttributeError("Received wrong column")

        self._col_names = col_names if col_names and len(col_names) == len(columns) else columns

        self._filter = ()
        self._rows = []

        self.refresh()


    def rowCount(self, parent = pyc.QModelIndex()):
        return 0 if isinstance(parent, pyc.QModelIndex) and parent.isValid() else len(self._rows)

    def columnCount(self, parent = pyc.QModelIndex()):
        return 0 if isinstance(parent, pyc.QModelIndex) and parent.isValid() else len(self._columns) 

    def data(self, index: pyc.QModelIndex, role: int=qt.DisplayRole):
        if ((role not in [qt.DisplayRole, qt.EditRole]) or 
        (index.column() not in range(len(self._columns))) or (index.row() not in range(len(self._rows)))):
            return None

        row = self._rows[index.row()]
        ret = getattr(row, self._columns[index.column()])
        if type(ret) is datetime.datetime:
            ret = pyc.QDateTime(ret)
        return ret

    def headerData(self, section: int, orientation: qt.Orientation = qt.Horizontal, role: int = qt.DisplayRole):
        if (orientation != qt.Horizontal) or (role != qt.DisplayRole) or (section not in range(len(self._columns))):
            return None

        return self._col_names[section]

    def setFilter(self, *filter):
        self._filter = filter
        self.refresh()

    def refresh(self):
        self.layoutAboutToBeChanged.emit()
        if self._filter:
            self._rows = self._model.query.filter(*self._filter).all()
        else:
            self._rows = self._model.query.all()

        self.layoutChanged.emit()

    # TODO: setData, flags, setHeaderData for editable model
