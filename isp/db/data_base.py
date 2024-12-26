import os
from dataclasses import dataclass, field
from typing import List
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.orm.query import Query
from isp.Utils import Singleton


@dataclass
class Page:
    items: List = field(default_factory=list)
    total: int = 0


class PaginateQuery(Query):

    def __init__(self, entities, session=None):
        super().__init__(entities=entities, session=session)

    def paginate(self, per_page: int, page: int) -> Page:
        """
        Paginate the query result and return page.

        :param per_page: The number of items in the given page.

        :param page: The page starting from 1.

        :return: A page with items and total
        """
        total = self.count()
        page = max(1, page)
        r = self[(page - 1) * per_page: page * per_page]
        return Page(items=r, total=total)


class _BaseMeta(DeclarativeMeta):

    def __init__(cls, classname, bases, dict_):
        super().__init__(classname, bases, dict_)

        cls.query = None
        if getattr(cls, "__tablename__", None):
            db = DataBase()  # safe to start a new db because DataBase is a singleton.
            cls.query = PaginateQuery(cls, session=db.session)




@Singleton
class DataBase:

    def __init__(self):

        dir_path = os.path.dirname(os.path.abspath(__file__))
        self.__url = f"sqlite:///{dir_path}/isp.db"
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
