from itertools import product
from typing import List, Union
from sqlalchemy import Column, or_, and_, text, func, distinct
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm.attributes import InstrumentedAttribute

from isp.Exceptions import EntityNotFound, QueryException
from isp.Structures.structures import Search, SearchResult, QueryOperators, ColumnOperator, SearchDefault
from isp.db import db

class BaseModel:

    def __init__(self, *args, **kwargs):
        super(BaseModel).__init__(self, *args, **kwargs)

    def __repr__(self):
        atr = (f"{key}={value}" for key, value in self.to_dict().items())
        return f"{type(self).__name__}({', '.join(atr)})"

    def to_dict(self):
        """
        Transforms the entity columns into a dictionary. This is equivalent to a Dto.
        Import: This method only converts the Columns fields. For tables with
        relationship, you must extend this method and add the relationships as desired.
        :return: A dictionary representation of the entity's columns.
        """

        # Check if it is the right instance.
        if isinstance(self, db.Model):
            # construct a dictionary from column names and values.
            dict_representation = {c.name: getattr(self, c.name) for c in self.__table__.columns}
            return dict_representation
        else:
            raise AttributeError(type(self).__name__ + " is not instance of " + db.Model.__name__)

    @classmethod
    def from_dict(cls, dto: dict):
        """
        Gets a Model from a dictionary representation of it. Usually a Dto.
        :param dto: The data transfer object as a dictionary.
        :return: The model represent this class.
        """
        # Map column names back to structures fields. The keys must be equal to the column name.
        clean_dict = {c.name: dto.get(c.name, None) for c in cls.__table__.columns}
        return cls(**clean_dict)

    def validate(self, entity_id):
        """
        Validate this entity. Check if this entity exists in the database. This is useful to check
        if after convert from dictionary the entity actually exists (valid id) in the database, before update it.
        :param entity_id: The id used to validate this entity.
        :return: The valid entity.
        """

        safe_entity = self.find_by_id(entity_id)
        if not safe_entity:
            raise EntityNotFound("The param_id = {} is not valid.".format(entity_id))

        return safe_entity

    @staticmethod
    def commit():
        try:
            db.session.commit()
            return True
        except SQLAlchemyError as error_message:
            print(error_message)
            db.session.rollback()
            return False

    def save(self):
        """
        Insert or Update the given entity.
        :return: True if succeeded, false otherwise.
        """
        db.session.add(self)
        return self.commit()

    @classmethod
    def bulk_save(cls, models: List[db.Model]):
        """
        Insert or Update the given entities.
        :param: models: A list of models to save
        :return: True if succeeded, false otherwise.
        """
        all([db.session.add(m) for m in models])
        return cls.commit()

    def delete(self):
        """
        Delete the given entity.
        :return: True if succeed, false otherwise.
        """
        db.session.delete(self)
        return self.commit()

    @classmethod
    def bulk_delete(cls, filters=None):
        """
        Delete all entities which match the query from the used filters.
        :param filters: A list of filters, i.e: [UserModel.forename == "John", UserModel.age < 25]
        :return: True if succeed, false otherwise.
        """
        if type(filters) is not list:
            filters = [filters]
        cls.query.filter(*filters).delete(synchronize_session=False)
        return cls.commit()

    def __lshift__(self, other):
        """
        Set all the attribute values from other to self.
        Usage::
            entity << other_entity
        :param other: The entity that will pass the attributes values.
        :return:
        """
        for c in self.__table__.columns:
            self.__setattr__(c.name, other.__getattribute__(c.name))

    @classmethod
    def __class_validation(cls):
        """
        Used to validate the class.
        """

        # check if this class is a subClass of Model
        if not issubclass(cls, db.Model):
            raise AttributeError(cls.__name__ + " is not subclass of " + db.Model.__name__)

    @classmethod
    def _get_column_from_name(cls, c_name: str) -> Union[Column, None]:
        """
        Gets the column in the table for the given c_name.
        :param c_name: The name of the column.
        :return: A table column or None if not found.
        """

        cls.__class_validation()
        for column in cls.__table__.columns:
            if c_name.lower().strip() == column.name:
                return column
        return None

    @classmethod
    def _create_order_by_list(cls, search: Union[Search, SearchDefault]):
        """
        Creates a list of order_by with columns from search.
        :param search: A Search instance.
        :return: A list contain columns to be used as order.
        """

        cls.__class_validation()
        order_by_list = []
        if not search.OrderBy:
            return []

        for searchBy in search.OrderBy.split(","):
            order_by = cls._get_column_from_name(searchBy)
            if order_by is None:
                pass
                # raise AppException(f"The column name {searchBy} doesn't exits for the Model {cls.__name__}")
            if search.OrderDesc and order_by is not None:
                order_by = order_by.desc()

            order_by_list.append(order_by)

        return [ob for ob in order_by_list if ob is not None]

    @staticmethod
    def clear_search_columns(search_c: list, values: list):
        """
        This will clean the columns where the values are empty. i.e:
        Example:
            >>> list_a = ["a", "b", "c"]
            >>> list_b = ["hi", "", "hello"]
            >>> BaseModel.clear_search_columns(list_a, list_b)
            returns  ["a", "c"], ["hi", "hello"]
        :param search_c: A list of column names.
        :param values: A list of vaues.
        :return: The clean lists without columns that mach empty values.
        """

        filter_lists = list(filter(lambda x: x[1] != "", zip(search_c, values)))

        fc_t, fv_t = [], []
        if filter_lists:
            fc_t, fv_t = zip(*filter_lists)
        return list(fc_t), list(fv_t)

    @classmethod
    def _construct_filter(cls, search: Union[Search, SearchDefault]):
        """
        Construct a list of search filters. The constructor will use SearchBy, SearchValue and Operator
        from search to make the filters. If you want to override the operator for a given column you
        can add :operator to the column name, i.e: "Username: ==" or "UserAge: >="
        Valid operator: "==, <,>,<=,>=,!=,[],]"
        :param search: A struct Search, where SearchBy, SearchValue and Operator will be used to
            construct the filter.
        :return: A list of filters to be used by sqlalchemy filter.
        """
        search_columns: List[ColumnOperator] = []
        column_names = search.SearchBy.split(",") if type(search.SearchBy) == str else search.SearchBy
        for column_name in column_names:  # accepts multiple columns split by ,
            # for column_name in search.SearchBy.split(","):  # accepts multiple columns split by ,
            column_name, *op = column_name.split(":")  # check if column has a operator override i.e "UserName: ==".
            search_column = cls._get_column_from_name(column_name)
            if search_column is None:
                raise QueryException(f"The column {column_name} you are trying to search at {cls.__name__} "
                                     f"doesn't exists.")
            op = QueryOperators(op[0].strip()) if op else search.Operator
            search_columns.append(ColumnOperator(search_column, op))

        find_values = [f"{v}".strip() for v in search.SearchValue]

        # construct search filter.
        if search.MapColumnAndValue:
            search_columns, find_values = cls.clear_search_columns(search_columns, find_values)
            search_filters = [sc.QueryOp.apply_filter(sc.Column, value) for sc, value in
                              zip(search_columns, find_values)]
        else:
            find_values = list(filter(lambda x: x != "", find_values))
            # makes n:x search for column:value
            search_filters = [sc.QueryOp.apply_filter(sc.Column, value) for sc, value in
                              product(search_columns, find_values)]

        return search_filters

    @classmethod
    def _join_query(cls, query_to_join, search: Search):
        """
        Join a query based on search parameters.
        :param query_to_join: The query the join will be appended.
        :param search: A Search instance.
        :return: The query.
        """

        search_filters = cls._construct_filter(search)
        order_by_list = cls._create_order_by_list(search)

        # AND or OR
        if search.Use_AND_Operator:
            query = query_to_join.join(cls).filter(and_(*search_filters)).order_by(*order_by_list)
        else:
            query = query_to_join.join(cls).filter(or_(*search_filters)).order_by(*order_by_list)

        # This will call join again
        if search.Join:
            c, s = search.Join
            if issubclass(c, BaseModel):
                query = c._join_query(query, s)
            else:
                raise AttributeError("The first value at tuple must be a subclass of BaseModel")

        return query

    @classmethod
    def _create_query(cls, search: Union[Search, SearchDefault]):
        """
        Create a query based on search parameters
        :param search: A Search instance.
        :return: The query.
        """

        search_filters = cls._construct_filter(search)
        order_by_list = cls._create_order_by_list(search)

        # AND or OR
        if search.Use_AND_Operator:
            query = cls.query.filter(and_(*search_filters)).order_by(*order_by_list)
        else:
            query = cls.query.filter(or_(*search_filters)).order_by(*order_by_list)

        if search.TextualQuery:
            query = query.filter(text(search.TextualQuery)).order_by(*order_by_list)

        if search.Join:
            c, s = search.Join
            if issubclass(c, BaseModel):
                query = c._join_query(query, s)
            else:
                raise AttributeError("The first value at tuple must be a subclass of BaseModel")

        return query

    @classmethod
    def search(cls, search):
        """
        Search for entities based on :class:`Search` criteria.
        :param search: An instance of Search or an object that implements the method to_search() which
            returns an instance of Search.
        :return: A SearchResult instance
        """
        if not (isinstance(search, Search) or isinstance(search, SearchDefault)):
            if hasattr(search, "to_search"):
                search = search.to_search()
            else:
                raise AttributeError("search object has no method to_search")

        query = cls._create_query(search)

        if search.PerPage > 0 and search.Page > 0:
            page = query.paginate(per_page=search.PerPage, page=search.Page)
            entities = page.items
            total = page.total
        else:
            entities = query.all()
            total = len(entities)

        if entities:
            return SearchResult(entities, total)

        return SearchResult([], 0)

    @classmethod
    def find_by_id(cls, *entity_ids):
        """
        Find by id.
        :param entity_ids: The primary keys of the entity.
        :return: The entity if found, None otherwise.
        """
        if not entity_ids:
            return None
        # Validate class before query
        cls.__class_validation()
        entity = cls.query.get(entity_ids)
        if entity:
            return entity

        return None

    @classmethod
    def pagination(cls, per_page, page):
        """
        Get a list of entities using pagination.
        :param per_page: The maximum number entities per page.
        :param page: The current page.
        :return: The list of entities, None otherwise.
        """

        # Validate class before query
        cls.__class_validation()

        entities = cls.query.paginate(per_page=per_page, page=page).items
        if entities:
            return entities

        return None

    @classmethod
    def get_all(cls, order_by=None):
        """
        Get all entities from this model.
        :param order_by: (Optional) The Column to sort the query.
        :return: The list of entities.
        """
        # Validate class before query
        cls.__class_validation()

        if order_by:
            entity_list = cls.query.order_by(order_by).all()
        else:
            entity_list = cls.query.all()

        if entity_list:
            return entity_list

        return []

    @classmethod
    def find_by(cls, get_first=True, order_by: Union[Column, InstrumentedAttribute] = None, **kwarg):
        """
        Find by an specific column name.
        :param get_first: (default=True). If True return one value, otherwise it will try to get
        all entries that match the query.
        :param order_by: (Optional) The Column to sort the query.
        :param kwarg: The column name as key and the value to match, e.g username="Jon". If more than one
            filter is given the query will use AND to join it.
            Important: you must pass at least one kwarg to this method, otherwise a ValueError will raise.
        :return: The entity if get_first=True and it exists or a list of entity if get_first=False and exists.
            None, otherwise.
        """
        # Validate class before query
        cls.__class_validation()

        if len([*kwarg]) < 1:
            raise ValueError("find_by must have at least one **kwarg")

        query = cls.query.filter_by(**kwarg).order_by(order_by)

        if get_first:
            entity = query.first()
        else:
            entity = query.all()

        if entity:
            return entity

        return None

    @classmethod
    def find_by_filter(cls,  filters=None, order_by=None, **kwargs):
        """
        Find by the given filters. To paginate it both kwargs, page and per_page must be provide. Otherwise
        it return all the found entities.
        :param filters: A list of filters, i.e: [UserModel.forename == "John", UserModel.age < 25]
        :param order_by: (Optional) The Column to sort the query.
        :keyword kwargs:
        :keyword page: The page number
        :keyword per_page: The maximum number of entities per page.
        :keyword use_or: Set True if filters should be wrapped with 'OR' instead of 'AND'.
        :return: A list of the found entities or paginate if per_page and page were given.
        """

        per_page = kwargs.get("per_page", None)
        page = kwargs.get("page", None)
        use_or = kwargs.get("use_or", False)

        if use_or:
            query = cls.query.filter(or_(*filters))
        else:
            query = cls.query.filter(*filters)
        if per_page and page:
            return query.order_by(order_by).paginate(per_page=per_page, page=page)
        else:
            return query.order_by(order_by).all()

    @classmethod
    def total(cls, count_column: InstrumentedAttribute = None, filters=None) -> int:
        """
        Get the total number of entities from this model.
        :param count_column: Specify the column to perform the count, i.e: User.user_id. If
            None is used, the method will try to use the id column if it exists. Otherwise, it will
            perform a generic count(*). If a count_column is passed then it also will perform a distinct count.
        :param filters: A list of filters to be used, i.e: filters = [UserModel.forename == "John"]
        :return: The total number of entities in the database.
        """
        if filters is None:
            filters = []
        if count_column:
            return cls.query.session.query(func.count(distinct(count_column))).filter(*filters).scalar()
        elif cls.__dict__.get("id", None):
            return cls.query.session.query(func.count(cls.id)).filter(*filters).scalar()
        else:
            return cls.query.filter(*filters).count()
