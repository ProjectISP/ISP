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

    def __init__(self, url=None, debug=False):
        print("Database started")
        self.debug = debug
        self.url = url
        if not url:
            dir_path = os.path.dirname(os.path.abspath(__file__))
            self.url = "sqlite:///{}/isp.db".format(dir_path)

        self.engine = create_engine(self.url)
        self.session: Session = sessionmaker(bind=self.engine, expire_on_commit=False)()
        self.Model = declarative_base(metaclass=_BaseMeta)

    def create_all(self):
        self.Model.metadata.create_all(self.engine)

    def __repr__(self):
        return "DataBase(url={}, debug={})".format(self.url, self.debug)
