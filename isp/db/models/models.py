# just for test implementation
from obspy.core.event import Origin
from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey
from sqlalchemy.orm import relationship
from isp.db import db, generate_id, generate_id_from_origin
from isp.db.models import RelationShip
from isp.db.models.base_model import BaseModel


class FirstPolarityModel(db.Model, BaseModel):
    __tablename__ = 'first_polarity'

    id = Column(String(16), primary_key=True)
    event_info_id = Column(String(16), ForeignKey("event_locations.id"), nullable=False, unique=True)
    strike_fp = Column(Float, nullable=True)
    dip_fp = Column(Float, nullable=True)
    rake_fp = Column(Float, nullable=True)
    misfit_fp = Column(Float, nullable=True)
    azimuthal_fp_Gap = Column(Float, nullable=True)
    station_fp_polarities_count = Column(Integer, nullable=True)

    def __repr__(self):
        return "FirstPolarityModel({})".format(self.to_dict())


class MomentTensorModel(db.Model, BaseModel):
    __tablename__ = 'moment_tensor'

    id = Column(String(16), primary_key=True)
    event_info_id = Column(String(16), ForeignKey("event_locations.id"), nullable=False, unique=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    depth = Column(Float, nullable=False)
    VR = Column(Float, nullable=False)
    CN = Column(Float, nullable=False)
    dc = Column(Float, nullable=True)
    clvd = Column(Float, nullable=True)
    iso = Column(Float, nullable=True)
    mw_mt = Column(Float, nullable=True)
    mo = Column(Float, nullable=True)
    strike_mt = Column(Float, nullable=True)
    dip_mt = Column(Float, nullable=True)
    rake_mt = Column(Float, nullable=True)
    mrr = Column(Float, nullable=True)
    mtt = Column(Float, nullable=True)
    mpp = Column(Float, nullable=True)
    mrt = Column(Float, nullable=True)
    mrp = Column(Float, nullable=True)
    mtp = Column(Float, nullable=True)



    def __repr__(self):
        return "MomentTensorModel({})".format(self.to_dict())


class PhaseInfoModel(db.Model, BaseModel):
    __tablename__ = 'phase_info'

    id = Column(String(16), primary_key=True)
    event_info_id = Column(String(16), ForeignKey("event_locations.id"), nullable=False, primary_key=True)
    station_code = Column(String(5), nullable=False)
    instrument = Column(String(5), nullable=True)
    component = Column(String(5), nullable=True)
    phase = Column(String(5), nullable=False)
    polarity = Column(String(1), nullable=True)
    time = Column(DateTime, nullable=False)
    amplitude = Column(Float, nullable=True)
    travel_time = Column(Float, nullable=False)
    time_residual = Column(Float, nullable=False)
    time_weight = Column(Float, nullable=False)
    distance_km = Column(Float, nullable=False)
    distance_degrees = Column(Float, nullable=False)
    azimuth = Column(Float, nullable=False)
    take_off_angle = Column(Float, nullable=False)


    def __repr__(self):
        return "PhaseInfoModel({})".format(self.to_dict())

    @classmethod
    def create_phases_from_origin(cls, event_info_id, picks):
        phases = []

        #if cls.find_by(latitude=origin.latitude, longitude=origin.longitude, depth=origin.depth,
         #              origin_time=origin.time.datetime):
         #   raise AttributeError("Object already exist in the database.")

        for pick in picks:
            event_dict = {"id": generate_id(16), "event_info_id": event_info_id, "station_code": pick.station,
                          "instrument": pick.instrument,
                          "component": pick.component, "phase": pick.phase,
                          "polarity": pick.polarity, "time": pick.date.datetime, "time_residual": pick.time_residual,
                          "time_weight": pick.time_weight, "travel_time": pick.travel_time,
                          "distance_degrees": pick.distance_degrees,
                          "distance_km": pick.distance_km, "azimuth": pick.azimuth, "take_off_angle": pick.takeoff_angle}

            phases.append(cls(**event_dict))

        return phases



class EventArrayModel(db.Model, BaseModel):

    # The name of the table at the database.
    __tablename__ = "event_array"

    # The table columns.
    event_info_id = Column(String(16), ForeignKey("event_locations.id"), primary_key=True)
    array_analysis_id = Column(String(16), ForeignKey("array_analysis.id"), primary_key=True)

    def __repr__(self):
        return "EventArrayModel(event_info_id={}, array_analysis_id={})".format(self.event_info_id,
                                                                                self.array_analysis_id)


class EventLocationModel(db.Model, BaseModel):
    __tablename__ = 'event_locations'

    id = Column(String(14), primary_key=True)
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
    mc = Column(Float, nullable=True)
    mc_error = Column(Float, nullable=True)
    first_polarity = relationship(RelationShip.FIRST_POLARITY, backref="event_location",
                                  cascade="save-update, merge, delete", lazy=True)
    moment_tensor = relationship(RelationShip.MOMENT_TENSOR, backref="event_location",
                                 cascade="save-update, merge, delete", lazy=True)
    phase_info = relationship(RelationShip.PHASE_INFO, backref="event_location",
                              cascade="save-update, merge, delete", lazy=True)
    event_arrays = relationship(RelationShip.EVENT_ARRAY, backref="event_location",
                                cascade="save-update, merge, delete", lazy=True)

    def __repr__(self):
        return "EventLocationModel({})".format(self.to_dict())

    @classmethod
    def create_from_origin(cls, origin: Origin):

        #origin_id = generate_id_from_origin(origin)

        if cls.find_by(latitude=origin.latitude, longitude=origin.longitude, depth=origin.depth,
                       origin_time=origin.time.datetime):
            raise AttributeError("Object already exist in the database.")

        event_dict = {"id": generate_id(16), "origin_time": origin.time.datetime, "transformation": "SIMPLE",
                      "rms": origin.quality.standard_error, "latitude": origin.latitude,
                      "longitude": origin.longitude, "depth": origin.depth,
                      "uncertainty": origin.depth_errors["uncertainty"],
                      "max_horizontal_error": origin.origin_uncertainty.max_horizontal_uncertainty,
                      "min_horizontal_error": origin.origin_uncertainty.min_horizontal_uncertainty,
                      "ellipse_azimuth": origin.origin_uncertainty.azimuth_max_horizontal_uncertainty,
                      "number_of_phases": origin.quality.used_phase_count,
                      "azimuthal_gap": origin.quality.azimuthal_gap,
                      "max_distance": origin.quality.maximum_distance,
                      "min_distance": origin.quality.minimum_distance}

        return cls(**event_dict)


    def set_magnitudes(self, mag_dict):
        import numbers
        from collections.abc import Mapping
        if not isinstance(mag_dict, Mapping):
            raise AttributeError('Received argument is not a valid mapping')

        for key, value in mag_dict.items():
            if hasattr(self, key) and isinstance(value, numbers.Number):
                setattr(self, key, value)

    @property
    def get_arrays(self):
        """ Get a list of ArrayAnalysisModel for this event_location."""
        return [ArrayAnalysisModel.find_by_id(event_array.array_analysis_id)
                for event_array in self.event_arrays if event_array]

    def add_array(self, array_analysis_id: str):
        """
        Link an ArrayAnalysisModel to this event_location.

        Important: This will not be added to the database until this entity is saved.

        :param array_analysis_id: The current id of the array_analysis.
        """


        event_arrays = EventArrayModel(event_info_id=self.id, array_analysis_id=array_analysis_id)
        self.event_arrays.append(event_arrays)


    def add_phase(self, phase):
        """
        Link an ArrayAnalysisModel to this event_location.

        Important: This will not be added to the database until this entity is saved.

        :param array_analysis_id: The current id of the array_analysis.
        """

        phase_add = PhaseInfoModel(id=phase.id, event_info_id=self.id, station_code=phase.station_code,
        instrument=phase.instrument, component=phase.component, phase=phase.phase, polarity=phase.polarity,
        time=phase.time, time_residual=phase.time_residual, time_weight=phase.time_weight, travel_time=phase.time_weight,
        distance_degrees=phase.distance_degrees, distance_km=phase.distance_km, azimuth=phase.azimuth,
        take_off_angle=phase.take_off_angle)

        self.phase_info.append(phase_add)

class ArrayAnalysisModel(db.Model, BaseModel):
    __tablename__ = 'array_analysis'

    id = Column(String(16), primary_key=True)
    latitude_ref = Column(Float, nullable=False)
    longitude_ref = Column(Float, nullable=False)
    slowness = Column(Float, nullable=True)
    slowness_err = Column(Float, nullable=True)
    back_azimuth = Column(Float, nullable=True)
    back_azimuth_err = Column(Float, nullable=True)
    rel_power = Column(Float, nullable=True)
    event_array = relationship(RelationShip.EVENT_ARRAY, backref="array_analysis",
                               cascade="save-update, merge, delete", lazy=True)

    def __repr__(self):
        return "ArrayAnalysisModel({})".format(self.to_dict())

    @classmethod
    def create(cls, lat: float, long: float):
        if cls.find_by(latitude_ref=lat, longitude_ref=long):
            return None

    @property
    def get_event_location(self):
        """ Get a list of ArrayAnalysisModel for this event_location."""
        return ArrayAnalysisModel.find_by_id(self.event_array.array_analysis_id)
