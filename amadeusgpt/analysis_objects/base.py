import os
from abc import ABC, abstractmethod

import numpy as np


class SerializeableObject(ABC):
    @abstractmethod
    def serialize(self):
        pass

    @abstractmethod
    def deserialize(self):
        pass


class AnalysisObject(SerializeableObject):
    """
    This class should handle serialization and deserialization.
    Should support both nonsql database or nunpy / json etc.
    """

    def serialize(self, manager_save_path: str, index: int):
        ret = {}
        ret = {"__class__": self.__class__.__name__}
        for attr_name in self.__dict__:
            base_path = os.path.join(manager_save_path, f"object{index}")
            if isinstance(getattr(self, attr_name), np.ndarray):
                npy_path = os.path.join(base_path, "{attr_name}.npy")
                os.makedirs(npy_path, exist_ok=True)
                np.save(getattr(self, attr_name), npy_path)
                ret[attr_name] = npy_path
            else:
                ret[attr_name] = getattr(self, attr_name)
        return ret

    @classmethod
    def deserialize(cls, data):
        instance = cls()  # Placeholder for actual instantiation logic
        for attr, value in data.items():
            if attr != "__class__":
                if isinstance(value, str) and value.endswith(".npy"):
                    setattr(instance, attr, np.load(value))
                else:
                    setattr(instance, attr, value)
        return instance
