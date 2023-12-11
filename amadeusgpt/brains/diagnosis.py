from amadeusgpt.brains.base import BaseBrain
from amadeusgpt.system_prompts.diagnosis import _get_system_prompt
from amadeusgpt.logger import AmadeusLogger


class DiagnosisBrain(BaseBrain):
    """
    Resource management for testing and error handling
    """

    @classmethod
    def get_system_prompt(
        cls, task_description, function_code, interface_str, traceback_output
    ):
        return _get_system_prompt(
            task_description, function_code, interface_str, traceback_output
        )

    @classmethod
    def get_diagnosis(
        cls, task_description, function_code, interface_str, traceback_output
    ):
        AmadeusLogger.info("traceback seen in error handling")
        AmadeusLogger.info(traceback_output)
        message = [
            {
                "role": "system",
                "content": cls.get_system_prompt(
                    task_description, function_code, interface_str, traceback_output
                ),
            },
            {
                "role": "user",
                "content": f"<query:> {task_description}\n  <func_str:> {function_code}\n  <errors:> {traceback_output}\n",
            },
        ]

        response = cls.connect_gpt(message, max_tokens=400, gpt_model="gpt-3.5-turbo")
        return response.choices[0]["message"]["content"]
