from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.query import Query


class BaseMeta(DeclarativeMeta):

    def __init__(cls, classname, bases, dict_):
        super().__init__(classname, bases, dict_)

        cls.query = None
        if getattr(cls, "__tablename__", None):
            print(cls)
            cls.query = Query(cls, session=DataBase.session)


class DataBase:
    engine = create_engine('sqlite:///isp.db')
    session = sessionmaker(bind=engine)()

    def __init__(self):
        print("Start db")
        self.Model = declarative_base(metaclass=BaseMeta)
        # self.Model.create_query(self.session)

    def create_all(self):
        self.Model.metadata.create_all(DataBase.engine)


db = DataBase()


class BaseModel:

    def save(self):
        """
        Insert or Update the given entity.
        :return: True if succeed, false otherwise.
        """
        try:
            db.session.add(self)
            db.session.commit()
        except:
            print("Bad insert")
            db.session.rollback()
        finally:
            db.session.close()


class User(db.Model, BaseModel):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    fullname = Column(String)
    nickname = Column(String)

    def __repr__(self):
        return "User(name={}, fullname={}, nickname={})".format(self.name, self.fullname, self.nickname)


class Dog(db.Model, BaseModel):
    __tablename__ = 'dogs'

    id = Column(Integer, primary_key=True)
    name = Column(String)

    def __repr__(self):
        return "Dog(name={})".format(self.name)
