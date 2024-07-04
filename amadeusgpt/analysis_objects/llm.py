import json
import os
import re
import time
import traceback
from amadeusgpt.utils import AmadeusLogger
from .base import AnalysisObject
import openai
from openai import OpenAI
import base64

class LLM(AnalysisObject):
    total_tokens = 0
    prices = {"gpt-4o": {"input": 5 / 10**6, "output": 15 / 10**6}}
    total_cost = 0

    def __init__(self, config):
        self.config = config
        self.max_tokens = config.get("max_tokens", 6000)
        self.gpt_model = config.get("gpt_model", "gpt-4o")
        self.keep_last_n_messages = config.get("keep_last_n_messages", 2)

        # the list that is actually sent to gpt
        self.context_window = []
        # only for logging and long-term memory usage.
        self.history = []
    

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def speak(self):
        """
        Speak to the chat channel
        """
        raise NotImplementedError("This method should be implemented in the subclass")

    def connect_gpt(self, messages, **kwargs):
        # if openai version is less than 1
        return self.connect_gpt_oai_1(messages, **kwargs)

    def connect_gpt_oai_1(self, messages, **kwargs): 
        """
        This is routed to openai > 1.0 interfaces
        """       

        if self.config.get("use_streamlit", False):
            if "OPENAI_API_KEY" in os.environ:
                openai.api_key = os.environ["OPENAI_API_KEY"]
        else:
            openai.api_key = os.environ["OPENAI_API_KEY"]
        response = None
        # gpt_model is default to be the cls.gpt_model, which can be easily set
        gpt_model = self.gpt_model
        # in streamlit app, "gpt_model" is set by the text box

        client = OpenAI()

        if self.config.get("use_streamlit", False):
            if "gpt_model" in st.session_state:
                self.gpt_model = st.session_state["gpt_model"]

        configurable_params = ["gpt_model", "max_tokens"]

        for param in configurable_params:
            if param in kwargs:
                setattr(self, param, kwargs[param])

        # the usage was recorded from the last run. However, since we have many LLMs that
        # share the call of this function, we will need to store usage and retrieve them from the database class
        num_retries = 3
        for _ in range(num_retries):
            try:
                json_data = {
                    "model": self.gpt_model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": 0.0,
                }
                response = client.chat.completions.create(**json_data)

                LLM.total_tokens =  LLM.total_tokens + response.usage.prompt_tokens + response.usage.completion_tokens
                LLM.total_cost += (
                    LLM.prices[self.gpt_model]["input"] * response.usage.prompt_tokens
                    + LLM.prices[self.gpt_model]["output"] * response.usage.completion_tokens
                )
                print("current total cost", round(LLM.total_cost, 2), "$")
                print("current total tokens", LLM.total_tokens)
                # TODO we need to calculate the actual dollar cost
                break

            except Exception as e:

                error_message = traceback.format_exc()
                print("error", error_message)

                if "This model's maximum context" in error_message:
                    if len(self.context_window) > 2:
                        AmadeusLogger.info("doing active forgetting")
                        self.context_window.pop(1)
                        # and the corresponding bot answer
                        self.context_window.pop(1)

                elif "Rate limit reached" in error_message:
                    print("Hit rate limit. Sleeping for 10 sec")
                    time.sleep(10)

        return response

    def update_history(self, role, content, encoded_image = None, replace=False):
        if role == "system":
            if len(self.history) > 0:
                self.history[0]["content"] = content
                self.context_window[0]["content"] = content
            else:
                self.history.append({"role": role, "content": content})
                self.context_window.append({"role": role, "content": content})
        else:

            if replace == True:
                if len(self.history) == 2:
                    self.history[1]["content"] = content
                    self.context_window[1]["content"] = content
                else:
                    self.history.append({"role": role, "content": content})
                    self.context_window.append({"role": role, "content": content})

            else:
                if encoded_image is None:
                    self.history.append({"role": role, "content": content})
                    num_AI_messages = (len(self.context_window) - 1) // 2
                    if num_AI_messages == self.keep_last_n_messages:
                        print ("doing active forgetting")
                        # we forget the oldest AI message and corresponding answer
                        self.context_window.pop(1)
                        self.context_window.pop(1)
                    self.context_window.append({"role": role, "content": content})
                else:
                    message = {
                        "role": "user", "content": [
                        {"type": "text", "text": content},
                        {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{encoded_image}"}
                        }]
                    }                                            
                    self.context_window.append(message) 
    
                    
               

    def clean_context_window(self):
        while len(self.context_window) > 1:
            AmadeusLogger.info("cleaning context window")
            self.context_window.pop()

    def parse_openai_response(cls, response):
        """
        Take the chat history, parse gpt's answer and execute the code

        Returns
        -------
        Dictionary
        """
        # there should be better way to do this
        if response is None:
            text = "Something went wrong"
        else:
            text = response.choices[0].message.content.strip()

        # we need to consider better ways to parse functions
        # and save them in a more structured way
        pattern = r"```python(.*?)```"
        function_code = re.findall(pattern, text, re.DOTALL)[0]

        # create a placeholder
        thought_process = text.replace(function_code, "<python_code>")

        return text, function_code, thought_process


class VisualLLM(LLM):
    def __init__(self, config):
        super().__init__(config)
    def speak(self, sandbox):
        """
        Only to comment about one image
        #1) What animal is there, how many and what superanimal model we should use
        #2) report the background object list
        #3) We format them in json format

        """

        from amadeusgpt.system_prompts.visual_llm import _get_system_prompt
        self.system_prompt = _get_system_prompt()
        analysis = sandbox.exec_namespace["behavior_analysis"]
        scene_image = analysis.visual_manager.get_scene_image()
        encoded_image = self.encode_image(scene_image)
        self.update_history("user", encoded_image)

        pattern = r"```json(.*?)```"
        if len(re.findall(pattern, text, re.DOTALL)) == 0:
            return None
        else:
            json_string = re.findall(pattern, text, re.DOTALL)[0]
            json_obj = json.loads(json_string)
            return json_obj

class CodeGenerationLLM(LLM):
    """
    Resource management for the behavior analysis part of the system
    """

    def __init__(self, config):
        super().__init__(config)
  

    def speak(self, sandbox):
        """
        Speak to the chat channel
        """
        qa_message = sandbox.messages[-1]
        query = qa_message["query"]

        self.update_system_prompt(sandbox)
        self.update_history("user", query)
        response = self.connect_gpt(self.context_window, max_tokens=2000)
        text = response.choices[0].message.content.strip()
        self.update_history("assistant", text)

        # we need to consider better ways to parse functions
        # and save them in a more structured way
        pattern = r"```python(.*?)```"
        if len(re.findall(pattern, text, re.DOTALL)) == 0:
            pass
        else:
            function_code = re.findall(pattern, text, re.DOTALL)[0]
            qa_message["code"] = function_code

        # this is for debug use
        with open("temp_answer.json", "w") as f:
            obj = {}
            obj["chain_of_thought"] = text
            json.dump(obj, f, indent=4)

        # create a placeholder
        thought_process = text

        qa_message["chain_of_thought"] = thought_process

    def update_system_prompt(self, sandbox):
        from amadeusgpt.system_prompts.code_generator import _get_system_prompt

        # get the formatted docs / blocks from the sandbox
        core_api_docs = sandbox.get_core_api_docs()
        task_program_docs = sandbox.get_task_program_docs()
        query_block = sandbox.get_query_block()

        behavior_analysis = sandbox.exec_namespace[
            "behavior_analysis"
        ]
       
        self.system_prompt = _get_system_prompt(
            query_block, core_api_docs, task_program_docs, behavior_analysis
        )

        # update both history and context window
        self.update_history("system", self.system_prompt)


class MutationLLM(LLM):
    def __init__(self, config):
        super().__init__(config)
    

    def update_system_prompt(self, sandbox):
        from amadeusgpt.system_prompts.mutation import _get_system_prompt

        self.system_prompt = _get_system_prompt(sandbox)
        # update both history and context window
        self.update_history("system", self.system_prompt)

    def speak(self, sandbox):
        # TODO maybe we don't need to keep the history
        """
        Speak to the chat channel
        """
        # query = "Please start. Make sure you provide one task program a time. Thanks a million!"
        query = "Please start. Thanks a million!"
        self.update_system_prompt(sandbox)
        self.update_history("user", query, replace=True)
        response = self.connect_gpt(self.context_window, max_tokens=4000)
        text = response.choices[0].message.content.strip()
        sandbox.chat_channel.chain_of_thought.append(response)
        return text


class BreedLLM(LLM):
    def __init__(self, config):
        super().__init__(config)
    

    def update_system_prompt(self, sandbox):
        from amadeusgpt.system_prompts.breed import _get_system_prompt

        behavior1_docs, behavior2_docs, composition_type = sandbox.get_breed_info()

        self.system_prompt = _get_system_prompt(
            behavior1_docs, behavior2_docs, composition_type
        )

        # update both history and context window

        self.update_history("system", self.system_prompt)

    def speak(self, sandbox):
        # TODO maybe we don't need to keep the history
        """
        Speak to the chat channel
        """
        query = "Now write the template function. Make sure your answer is concise and don't mention anything about filtering such as smooth_window or min_window\n"
        self.update_system_prompt(sandbox)
        self.update_history("user", query, replace=True)

        response = self.connect_gpt(self.context_window, max_tokens=400)
        text = response.choices[0].message.content.strip()
        sandbox.chat_channel.chain_of_thought.append(response)

        return text


class DiagnosisLLM(LLM):
    """
    Resource management for testing and error handling
    """
    

    @classmethod
    def get_system_prompt(
        cls, task_description, function_code, interface_str, traceback_output
    ):
        from amadeusgpt.system_prompts.diagnosis import _get_system_prompt

        return _get_system_prompt(
            task_description, function_code, interface_str, traceback_output
        )

    def get_diagnosis(
        self, task_description, function_code, interface_str, traceback_output
    ):
        AmadeusLogger.info("traceback seen in error handling")
        AmadeusLogger.info(traceback_output)
        message = [
            {
                "role": "system",
                "content": self.get_system_prompt(
                    task_description, function_code, interface_str, traceback_output
                ),
            },
            {
                "role": "user",
                "content": f"<query:> {task_description}\n  <func_str:> {function_code}\n  <errors:> {traceback_output}\n",
            },
        ]
        response = self.connect_gpt(message, max_tokens=400, gpt_model="gpt-3.5-turbo")
        return response.choices[0]["message"]["content"]


class SelfDebugLLM(LLM):

    def update_system_prompt(
        self,
    ):
        from amadeusgpt.system_prompts.self_debug import _get_system_prompt

        self.system_prompt = _get_system_prompt()

    def speak(self, sandbox):
        qa_message = sandbox.messages[-1]
        error_message = qa_message["error_message"]
        code = qa_message["code"]

        self.update_system_prompt()
        self.update_history("system", self.system_prompt)
        print("the code that gave errors was", code)
        query = f""" The code that caused error was {code}
And the error message was {error_message}. 
All the modules were already imported so you don't need to import them again.
Can you correct the code?
"""
        self.update_history("user", query)
        response = self.connect_gpt(self.context_window, max_tokens=700)
        text = response.choices[0].message.content.strip()

        print(text)
        thought_process = text
        pattern = r"```python(.*?)```"
        function_code = re.findall(pattern, text, re.DOTALL)[0]
        qa_message["code"] = function_code
        qa_message["chain_of_thought"] = thought_process


if __name__ == "__main__":
    from amadeusgpt.config import Config   
    from amadeusgpt.main import create_amadeus
    config = Config("amadeusgpt/configs/EPM_template.yaml")

    amadeus = create_amadeus(config)
    sandbox = amadeus.sandbox
    visualLLm = VisualLLM(config)
    visualLLm.speak(sandbox)
