from sqlalchemy import Column, or_, and_, text
from sqlalchemy.orm import Query
from isp.Exceptions import EntityNotFound, QueryException
from isp.Structures.structures import Search, SearchResult
from isp.db import db


class BaseModel:
    #TODO: This must be updated to include a more powerful searching filters
    def __init__(self, *args, **kwargs):
        super(BaseModel).__init__(self, *args, **kwargs)

    def to_dict(self):
        """
        Transforms the entity columns into a dictionary. This is equivalent to a Dto.

        Import: This method only converts the Columns fields. For tables with
        relationship, you must extend this method and add the relationships as desired.

        :return: A dictionary representation of the entity's columns.
        """

        # Check if is the right instance.
        if isinstance(self, db.Model):
            # construct a dictionary from column names and values.
            dict_representation = {c.name: getattr(self, c.name) for c in self.__table__.columns}
            return dict_representation
        else:
            raise AttributeError(type(self).__name__ + " is not instance of " + db.Model.__name__)

    @classmethod
    def from_dict(cls, dto):
        """
        Gets a Model from a dictionary representation of it. Usually a Dto.

        :param dto: The data transfer object as a dictionary.

        :return: The model represent this class.
        """
        # Map column names back to structures fields. The keys must be equal to the column name.
        clean_dict = {c.name: dto[c.name] for c in cls.__table__.columns}
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

    def save(self):
        """
        Insert or Update the given entity.

        :return: True if succeed, false otherwise.
        """
        try:
            db.session.add(self)
            db.session.commit()
        except Exception as e:
            print("Bad insert")
            print(e)
            db.session.rollback()
        finally:
            pass
            db.session.close()

    def delete(self):
        """
        Delete the given entity.

        :return: True if succeed, false otherwise.
        """
        try:
            db.session.delete(self)
            db.session.commit()
        except Exception as e:
            print("Bad delete", e)
            db.session.rollback()
        finally:
            db.session.close()

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
    def _get_column_from_name(cls, c_name: str):
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
    def _create_order_by_list(cls, search: Search):
        """
        Creates a list of order_by with columns from search.

        :param search: A Search instance.

        :return: A list contain columns to be used as order.
        """

        cls.__class_validation()
        order_by_list = []
        for searchBy in search.OrderBy.split(","):
            order_by = cls._get_column_from_name(searchBy)
            if search.OrderDesc and order_by is not None:
                order_by = order_by.desc()
            order_by_list.append(order_by)

        return order_by_list

    @classmethod
    def _create_query(cls, search: Search):
        """
        Create a query based on search parameters

        :param search: A Search instance.

        :return: The query.
        """

        search_columns = []
        for column_name in search.SearchBy.split(","):  # accepts multiple columns split by ,
            if len(column_name) > 0:
                search_column = cls._get_column_from_name(column_name)
                if search_column is None:
                    raise QueryException("The column {} you are trying to search at don't exists.".format(column_name))
                search_columns.append(search_column)

        find_values = []
        for value in search.SearchValue.split(","):  # accepts multiple values split by ,
            find_value = "%{}%".format(value.strip())
            find_values.append(find_value)

        # construct search filter.
        if search.MapColumnAndValue:
            # makes a 1:1 search for column:value
            search_filters = [sc.like(value) for sc, value in zip(search_columns, find_values)]
        else:
            # makes n:x search for column:value
            search_filters = [sc.like(value) for sc in search_columns for value in find_values]

        order_by_list = cls._create_order_by_list(search)

        # AND or OR
        if search.Use_AND_Operator:
            query = cls.query.filter(and_(*search_filters)).order_by(*order_by_list)
        else:
            query = cls.query.filter(or_(*search_filters)).order_by(*order_by_list)

        if search.TextualQuery:
            query = query.filter(text(search.TextualQuery)).order_by(*order_by_list)

        return query

    @classmethod
    def search(cls, search: Search):
        """
        Search for entities based on :class:`Search` criteria.

        :param search: A Search instance.

        :return: A SearchResult instance
        """

        query: Query = cls._create_query(search)

        # page = query.paginate(per_page=search.PerPage, page=search.Page)

        # entities = page.items
        entities = query.all()
        # total = page.total
        total = len(entities)
        if entities:
            return SearchResult(entities, total)

        return SearchResult([], 0)

    @classmethod
    def find_by_id(cls, entity_id):
        """
        Find by id.

        :param entity_id: The id of the entity.

        :return: The entity if found, None otherwise.
        """

        # Validate class before query
        cls.__class_validation()

        entity = cls.query.get(entity_id)
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
    def get_all(cls, order_by: Column = None):
        """
        Get all entities from this model.

        :param order_by: (Optional) The Column to sort the query.

        :return: The list of entities, None otherwise.
        """
        # Validate class before query
        cls.__class_validation()

        if order_by:
            entity_list = cls.query.order_by(order_by).all()
        else:
            entity_list = cls.query.all()

        if entity_list:
            return entity_list

        return None

    @classmethod
    def find_by(cls, get_first: bool = True, order_by: Column = None, **kwarg):
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

        if get_first:
            entity = cls.query.filter_by(**kwarg).first()
        else:
            entity = cls.query.filter_by(**kwarg).order_by(order_by).all()

        if entity:
            return entity

        return None

    @classmethod
    def total(cls) -> int:
        """
        Get the total number of entities of this model.

        :return: The total number of entities in the database.
        """
        entity_list = cls.query.all()
        if entity_list:
            return len(entity_list)
        return 0
