import os

import yaml


class Config:
    """
    The config loads from a YAML file and supports overriding with environment variables.
    """

    def __init__(self, config_file_path: str, default_config: dict = None):
        self.default_config = default_config or {}
        self.config_file_path = config_file_path
        assert os.path.exists(
            self.config_file_path
        ), f"Config file {self.config_file_path} not found."
        self.data = self.load_config()

    def __repr__(self):
        return repr(self.data)

    def __setitem__(self, key, value):
        self.data[key] = value

    def to_dict(self):
        return self.data

    def load_config(self):
        # Load the YAML config file
        if os.path.exists(self.config_file_path):
            with open(self.config_file_path, "r") as f:
                file_config = yaml.safe_load(f) or {}
        else:
            print(
                f"Warning: Config file {self.config_file_path} not found. Using default configurations."
            )
            file_config = {}

        # Merge with default config
        config = self.merge_configs(self.default_config, file_config)

        # Override with environment variables
        for key, value in config.items():
            env_value = os.getenv(key)
            if env_value is not None:
                config[key] = type(value)(env_value)  # Cast to original type

        return config

    @staticmethod
    def merge_configs(default, override):
        """
        Recursively merges two configurations.
        """
        if not isinstance(override, dict):
            return override
        result = dict(default)
        for k, v in override.items():
            if k in result and isinstance(result[k], dict):
                result[k] = Config.merge_configs(result[k], v)
            else:
                result[k] = v
        return result

    def get(self, key, default={}):
        return self.data.get(key, default)

    def __getitem__(self, key):
        """
        Get a value from the configuration data.
        """
        return self.data.get(key, {})

    def copy(self):
        return Config(self.config_file_path, self.default_config)

    def __str__(self):
        """
        Such that the indentation is similar to one in yaml
        """
        return yaml.dump(self.data, default_flow_style=False)
