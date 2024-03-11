from inspect import signature
from collections import defaultdict

DEFAULT_REGISTRY = defaultdict(dict)
CORE_API_REGISTRY = {}


ignore_functions = ['summary', 'serialize', 'deserialize',
                    'get_serializeable_list_names', 'init']

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
    return_type = str(sig.return_annotation) if sig.return_annotation is not sig.empty else None

    CORE_API_REGISTRY[func.__name__] = {
        'name': func.__name__,
        'parameters': inputs,
        'description': func.__doc__
    }
    # If possible, you might want to capture output details here, but note that
    # without executing the function or further annotations, this can be challenging.
    return func

def get_api_docs_json_objs():
    """
    Will pass the API_REGISTRY to a template engine so that OpenAI function call can use them    
    Here is an example format of OpenAI function call api
        {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }    
    """
    # pay extra attention to the format of the properties
    # This is the raw inputs field look like {'self': "<class 'inspect._empty'>", 'events': "<class 'inspect._empty'>", 'render': "<class 'inspect._empty'>", 'axs': 'typing.Optional[matplotlib.axes._axes.Axes]'}
    json_objs = []
    for func_name, func_details in CORE_API_REGISTRY.items():

        function_api = {
            "type": "function",
            "function": {
                "name": func_details['name'],
                "description": func_details.get('description', ''),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": list(func_details['parameters'].keys()),  # Assuming all parameters are required
                }                
            },
        }

        # Process parameters
        required_params = []
        for param_name, param_type in func_details['parameters'].items():
            # Ensure all parameters except 'self' have a type annotation
            if param_name != 'self':
                assert param_type != 'inspect._empty' , f"Missing type annotation for parameter '{param_name}' in function '{func_name}'"
            param_description = ""  # Placeholder as descriptions are not captured in your setup
            # Convert the type string to a more structured format if necessary
            param_type_formatted = param_type.replace("<class '", "").replace("'>", "")
            #if "Optional" not in param_type and "self" not in param_name:
            if "self" not in param_name:
                required_params.append(param_name)
            function_api["function"]["parameters"]["properties"][param_name] = {
                "type": param_type_formatted,
                "description": param_description,
            }        
        function_api["function"]["parameters"]["required"] = required_params

        json_objs.append(function_api)
    return json_objs
    