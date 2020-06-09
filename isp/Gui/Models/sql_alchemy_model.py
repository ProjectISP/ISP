from isp.Gui import pyc, qt
from isp.db.models.base_model import BaseModel, db
from sqlalchemy import Column
import datetime
# Get columns instead of column attribute names and query for every entity in realtion
class SQLAlchemyModel(pyc.QAbstractTableModel):
    """ This class implements a Qt model for SQL Alchemy model classes
        
        :param sql_alchemy_model: Subclass of BaseModel, representing a SQL Alchemy Model
        :param columns: List of strings representing the name of the columns. Must be the attribute name.
        :param col_names: List of strings representing the display name of the columns.
        :param parent: Qt parent of model
    """
    def __init__(self, sql_alchemy_models, columns=[], col_names=[], parent=None):
        super().__init__(parent)

        # If models object is not iterable convert it to list
        try: 
            iter(sql_alchemy_models)
        except Exception:
            sql_alchemy_models = [sql_alchemy_models]

        # Check if every received model is a SQL ALchemy model
        if not sql_alchemy_models or not all(self.isAlchemyModel(m) for m in sql_alchemy_models):
            raise TypeError("Some received model objects are not SQL Alchemy models")

        self._models = sql_alchemy_models
        self.setColumns(columns, col_names)
        self._filter = ()
        self._join_params = []
        self._join_dicts = []
        self._rows = []
        self._deleted_rows = []

        self.revertAll()

    def isAlchemyModel(self, model):
        return isinstance(model, type) and issubclass(model, BaseModel)

    
    def setColumns(self, columns=[], col_names=[]):
        """ 
        Set model columns. These columns may belong to any SQLAlchemy model passed to constructor.
        
        :param columns: List of Column instances which belongs to models in this instance. 
                        If empty, display every column in every model.
        :param col_names: List of names to be displayed for each column. 
                          If empty, attribute name is used. Must match columns' size
        """
        # If there is a column which not belongs to any model, raise exception
        if not all(c.class_ in self._models for c in columns):
            raise AttributeError("Column does not belong to received models")
        
        # If col_names' size is not the same as columns raise an exception
        if col_names and len(columns) != len(col_names):
            raise AttributeError("Received column names mismatch number of columns")

        # If columns is empty, then get every column in every model
        self._columns = columns if columns else [ c for m in self._models for c in m.__table__.columns ]

        # If col_names is empty, use column attribute name, otherwise use col_names
        self._col_names = col_names if col_names else [c.name for c in self._columns]


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
        
        query = db.session.query().add_columns(*self._columns)
        
        for p,d in zip(self._join_params, self._join_dicts) :
            query = query.join(*p, **d)

        if self._filter:
            self._rows = query.filter(*self._filter).all()
        else:
            self._rows = query.all()

        self.layoutChanged.emit()

    def setFilter(self, *filter):
        self._filter = filter

    def addJoinArguments(self, *args, **kwargs):
        self._join_params.append(args)
        self._join_dicts.append(kwargs) 

    # TODO: change join arguments?

    def getRows(self):
        return self._rows

    # Qt overriden methods
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

        try:
            ret = self._rows[index.row()][index.column()]
        except IndexError:
            ret = None

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

    

    # TODO: setData, flags, setHeaderData for editable model
