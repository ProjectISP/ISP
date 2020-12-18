import os

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.orm.query import Query

from isp.Utils import Singleton


class _BaseMeta(DeclarativeMeta):

    def __init__(cls, classname, bases, dict_):
        super().__init__(classname, bases, dict_)

        cls.query = None
        if getattr(cls, "__tablename__", None):
            db = DataBase()  # safe to start a new db because DataBase is a singleton.
            cls.query = Query(cls, session=db.session)


@Singleton
class DataBase:

    def __init__(self):

        dir_path = os.path.dirname(os.path.abspath(__file__))
        self.__url = "sqlite:///{}/isp.db".format(dir_path)
        self.__file_path = ""
        self.set_db_url(self.__url)

        self.engine = None
        self.session = None
        self.Model = None
        self.__has_started = False

    @property
    def has_started(self):
        return self.__has_started

    def has_sqlite_db_file(self):
        return os.path.isfile(self.__file_path)

    def remove_sqlite_db(self):
        if self.has_sqlite_db_file():
            os.remove(self.__file_path)

    def set_db_url(self, url):
        self.__url = url
        self.__file_path = self.__url.split("sqlite:///")[-1]

    def create_all(self):
        self.Model.metadata.create_all(self.engine)

    def start(self):
        if not self.has_started:
            self.engine = create_engine(self.__url)
            self.session: Session = sessionmaker(bind=self.engine, expire_on_commit=False)()
            self.Model = declarative_base(metaclass=_BaseMeta)
            self.__has_started = True
            print("Database started at the url: {}".format(self.__url))

    def __repr__(self):
        return "DataBase(url={})".format(self.__url)
