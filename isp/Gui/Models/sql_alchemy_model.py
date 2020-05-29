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
        self._deleted_rows = []

        self.revertAll()


    def rowCount(self, parent=pyc.QModelIndex()):
        return 0 if isinstance(parent, pyc.QModelIndex) and parent.isValid() else len(self._rows)

    def columnCount(self, parent=pyc.QModelIndex()):
        return 0 if isinstance(parent, pyc.QModelIndex) and parent.isValid() else len(self._columns) 

    def data(self, index: pyc.QModelIndex, role: int=qt.DisplayRole):
        # If role or index are not in correct ranges, return None
        check_role = role not in [qt.DisplayRole, qt.EditRole]
        check_col = index.column() not in range(len(self._columns))
        check_row = index.row() not in range(len(self._rows))
        if check_role or check_col or check_row:
            return None

        row = self._rows[index.row()]
        ret = getattr(row, self._columns[index.column()])
        if type(ret) is datetime.datetime:
            ret = pyc.QDateTime(ret)
        return ret

    def headerData(self, section: int, orientation: qt.Orientation=qt.Horizontal, role: int=qt.DisplayRole):
        if (orientation != qt.Horizontal) or (role != qt.DisplayRole) or (section not in range(len(self._columns))):
            return None

        return self._col_names[section]

    # TODO: Maybe this should be in a proxy model
    def removeRows(self, row, count, parent=pyc.QModelIndex()) :
        if (row + count) > len(self._rows):
            return False

        self.beginRemoveRows(pyc.QModelIndex(), row, row + count - 1)
        self._deleted_rows = [*self._deleted_rows, *self._rows[row:row + count]]
        del self._rows[row:row + count]
        self.endRemoveRows()
        return True

    # TODO: delete does not return anything, so it cannot be checked if submit
    # is OK
    def submitAll(self):
        for row in self._deleted_rows:
            row.delete()

        self._deleted_rows = []
        return True

    def revertAll(self):
        self.layoutAboutToBeChanged.emit()
        self._deleted_rows = []
        if self._filter:
            self._rows = self._model.query.filter(*self._filter).all()
        else:
            self._rows = self._model.query.all()

        self.layoutChanged.emit()

    def setFilter(self, *filter):
        self._filter = filter


    def getRows(self):
        return self._rows

    # TODO: setData, flags, setHeaderData for editable model
