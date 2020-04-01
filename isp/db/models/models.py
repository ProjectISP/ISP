# just for test implementation

from sqlalchemy import Column, Integer, String, DateTime, Float

from isp.db import db
from isp.db.models.base_model import BaseModel


class EventLocationModel(db.Model, BaseModel):
    __tablename__ = 'event_locations'

    id = Column(String(16), primary_key=True)
    origin_time = Column(DateTime, nullable=False)
    transformation = Column(String(10),  nullable=False)
    rms = Column(Float, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    depth = Column(Float, nullable=False)
    uncertainty = Column(Float, nullable=False)
    max_horizontal_error = Column(Float, nullable=False)
    min_horizontal_error = Column(Float, nullable=False)
    ellipse_azimuth = Column(Float, nullable=False)
    number_of_phases = Column(Integer, nullable=False)
    azimuthal_gap = Column(Integer, nullable=False)
    max_distance = Column(Float, nullable=False)
    min_distance = Column(Float, nullable=False)
    mb = Column(Float, nullable=True)
    mb_error = Column(Float, nullable=True)
    ms = Column(Float, nullable=True)
    ms_error = Column(Float, nullable=True)
    ml = Column(Float, nullable=True)
    ml_error = Column(Float, nullable=True)
    mw = Column(Float, nullable=True)
    mw_error = Column(Float, nullable=True)

    def __repr__(self):
        return "EventLocationModel(id={}, origin_time={}, latitude={}, longitude={})".\
            format(self.id, self.origin_time, self.latitude, self.longitude)


class Dog(db.Model, BaseModel):
    __tablename__ = 'dogs'

    id = Column(Integer, primary_key=True)
    name = Column(String)

    def __repr__(self):
        return "Dog(name={})".format(self.name)