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
        self.url = "sqlite:///{}/isp.db".format(dir_path)

        self.engine = None
        self.session = None
        self.Model = None
        self.__has_started = False

    @property
    def has_started(self):
        return self.__has_started

    def create_all(self):
        self.Model.metadata.create_all(self.engine)

    def start(self):
        if not self.has_started:
            self.engine = create_engine(self.url)
            self.session: Session = sessionmaker(bind=self.engine, expire_on_commit=False)()
            self.Model = declarative_base(metaclass=_BaseMeta)
            self.__has_started = True
            print("Database started at the url: {}".format(self.url))

    def __repr__(self):
        return "DataBase(url={})".format(self.url)
