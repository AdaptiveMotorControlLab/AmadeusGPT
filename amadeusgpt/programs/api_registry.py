from collections import defaultdict
from inspect import signature
from sre_constants import IN

DEFAULT_REGISTRY = defaultdict(dict)
CORE_API_REGISTRY = {}
INTEGRATION_API_REGISTRY = {}


ignore_functions = [
    "summary",
    "serialize",
    "deserialize",
    "get_serializeable_list_names",
    "init",
]


def register_class_methods(cls):
    for attr_name in dir(cls):
        if attr_name in ignore_functions:
            continue
        attr = getattr(cls, attr_name)
        cls_name = cls.__name__
        if callable(attr) and not attr_name.startswith("__"):
            DEFAULT_REGISTRY[cls_name][attr_name] = attr
    return cls


def register_core_api(func):
    # Capture the function's signature for input arguments
    sig = signature(func)
    inputs = {name: str(param.annotation) for name, param in sig.parameters.items()}
    return_type = (
        str(sig.return_annotation) if sig.return_annotation is not sig.empty else None
    )
    for name, _type in list(inputs.items()):
        if name == "self":
            inputs.pop(name)

    CORE_API_REGISTRY[func.__name__] = {
        "name": func.__name__,
        "parameters": inputs,
        "description": func.__doc__,
    }
    # If possible, you might want to capture output details here, but note that
    # without executing the function or further annotations, this can be challenging.
    return func


def register_integration_api(func):
    # Capture the function's signature for input arguments
    sig = signature(func)
    inputs = {name: str(param.annotation) for name, param in sig.parameters.items()}
    return_type = (
        str(sig.return_annotation) if sig.return_annotation is not sig.empty else None
    )
    for name, _type in list(inputs.items()):
        if name == "self":
            inputs.pop(name)

    INTEGRATION_API_REGISTRY[func.__name__] = {
        "name": func.__name__,
        "parameters": inputs,
        "description": func.__doc__,
        "func": func,
    }
    # If possible, you might want to capture output details here, but note that
    # without executing the function or further annotations, this can be challenging.
    return func
