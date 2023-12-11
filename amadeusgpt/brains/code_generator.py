from amadeusgpt.brains.base import BaseBrain
from amadeusgpt.system_prompts.code_generator import _get_system_prompt


class CodeGenerationBrain(BaseBrain):
    """
    Resource management for the behavior analysis part of the system
    """

    @classmethod
    def get_system_prompt(cls, interface_str, behavior_module_str):
        return _get_system_prompt(interface_str, behavior_module_str)

    @classmethod
    def update_system_prompt(cls, interface_str, behavior_modules_str):
        cls.system_prompt = cls.get_system_prompt(interface_str, behavior_modules_str)
        # update both history and context window
        cls.update_history("system", cls.system_prompt)
