import json
import os
from abc import ABC, abstractmethod
from functools import wraps

from cachetools import LRUCache

from amadeusgpt.config import Config


class BaseManager(ABC):
    """
    The subclass of this should maintain lists of serializeable objects.
    During serialization, it should serialize all the objects and save them to a json file.
    During deserialization, it should read the json file and deserialize all the objects.
    """

    @abstractmethod
    def serialize(self, base_path: str):
        pass

    @abstractmethod
    def deserialize(self, base_path: str):
        pass


def make_hashable(obj):
    if isinstance(obj, (tuple, list)):
        return tuple(make_hashable(e) for e in obj)
    elif isinstance(obj, dict):
        return frozenset((k, make_hashable(v)) for k, v in obj.items())
    elif isinstance(obj, set):
        # Convert set to frozenset, which is hashable
        return frozenset(make_hashable(e) for e in obj)
    else:
        return obj


class cache_decorator:
    def __init__(self, func):
        self.func = func
        wraps(func)(self)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        # Return a new, bound version of the decorator, with the instance bound.

        return lambda *args, **kwargs: self.__call__(instance, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        # The first argument is now the instance.
        instance, *args = args

        if not instance.use_cache:
            return self.func(instance, *args, **kwargs)

        hashable_args = make_hashable(args)

        hashable_kwargs = make_hashable(frozenset(kwargs.items()))

        cache_key = (hashable_args, hashable_kwargs)

        if cache_key in instance._cache:
            return instance._cache[cache_key]

        result = self.func(instance, *args, **kwargs)
        instance._cache[cache_key] = result
        return result


class Manager(BaseManager):
    def __init__(self, config: Config | dict, use_cache: bool = False):
        self.config = config
        self.use_cache = use_cache
        self._cache = LRUCache(maxsize=128)

    def serialize(self, base_path: str):
        ret = {}

        for attr_name in self.__dict__:
            if attr_name in self.get_serializeable_list_names():
                object_list = self.__dict__[attr_name]
                for idx, object in enumerate(object_list):
                    ret[attr_name].append(object.serialize(base_path, idx))

            else:
                ret[attr_name] = self.__dict__[attr_name]

        with open(os.path.join(base_path, f"{self.__class__.__name__}.json"), "w") as f:
            json.dump(ret, f)

    def deserialize(self, base_path: str) -> BaseManager:
        json_path = os.path.join(base_path, f"{self.__class__.__name__}.json")
        with open(json_path, "r") as f:
            data = json.load(f)

        serializeable_list_names = self.get_serializeable_list_names()

        for attr_name in data:
            if attr_name in serializeable_list_names:
                serialized_data = data[attr_name]
                for idx, serialized_data in serialized_data.items():
                    instance = object.deserialize(serialized_data)
                    self.__dict__[attr_name].append(instance)
            else:
                self.__dict__[attr_name] = data[attr_name]

    def summary(self):
        for attr_name in self.__dict__:
            if attr_name in self.get_serializeable_list_names():
                print(f"{attr_name} has {len(self.__dict__[attr_name])} objects")
                if len(self.__dict__[attr_name]) > 0:
                    self.__dict__[attr_name][0].summary()
            else:
                print(f"{attr_name} has {self.__dict__[attr_name]}")
