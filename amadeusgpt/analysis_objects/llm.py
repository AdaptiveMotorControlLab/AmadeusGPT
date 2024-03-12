from .base import AnalysisObject
import openai
import os
import traceback
import time 
from amadeusgpt.utils import AmadeusLogger
from amadeusgpt.utils import search_generated_func
import re

class LLM(AnalysisObject):
    def __init__(self, config):
        self.config = config
        self.max_tokens = config['max_tokens']
        self.gpt_model = config['gpt_model']
        self.context_window = []
        self.history = []
        self.usage = 0
        self.short_term_memory = []
        self.long_term_memory = {}
        self.accumulated_tokens = 0

    def connect_gpt(self, messages, **kwargs):
        # if openai version is less than 1
        if openai.__version__ < 1:
            return self.connect_gpt_oai_less_than_1(messages, **kwargs)
        else:
            return self.connect_gpt_oai_1(messages, **kwargs)

    def connect_gpt_oai_less_than_1(self, messages, **kwargs):
        if self.config.get('use_streamlit', False):            
            if "OPENAI_API_KEY" in os.environ:
                openai.api_key = os.environ["OPENAI_API_KEY"]
            else:
                import streamlit as st
                openai.api_key = st.session_state.get("OPENAI_API_KEY", "")
        else:
            openai.api_key = os.environ["OPENAI_API_KEY"]

        openai.Model.list()
        response = None
        # gpt_model is default to be the cls.gpt_model, which can be easily set
        gpt_model = self.gpt_model
        # in streamlit app, "gpt_model" is set by the text box

        if self.config.get('use_streamlit', False):
            if "gpt_model" in st.session_state:
                self.gpt_model = st.session_state["gpt_model"]

        configurable_params = ['gpt_model',
                               'max_tokens']
        
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
                    "stop": None,
                    "top_p": 1,
                    "temperature": 0.0,
                }               
                response = openai.ChatCompletion.create(**json_data)
                print(response)
                self.usage = response["usage"]
                break
            except openai.error.RateLimitError:
                if "rate_limit_error" not in st.session_state:
                    st.error(
                        "It appears you are out of funds/free tokens from openAI - please check your account settings"
                    )
                return None

            except Exception as e:
                error_message = traceback.format_exc()
               
                if self.usage is None:
                    print("OpenAI server not responding")

                if "This model's maximum context" in error_message:
                    if len(self.context_window) > 2:
                        AmadeusLogger.info("doing active forgetting")
                        self.context_window.pop(1)
                        # and the corresponding bot answer
                        self.context_window.pop(1)

                elif "Rate limit reached" in error_message:
                    print("Hit rate limit. Sleeping for 10 sec")
                    time.sleep(10)

        # Extract the parsed information from the response
        parsed_info = response
        return parsed_info



    def update_history(self, role, content):
        if role == "system":
            if len(self.history) > 0:
                self.history[0]["content"] = content
                self.context_window[0]["content"] = content
            else:
                self.history.append({"role": role, "content": content})
                self.context_window.append({"role": role, "content": content})
        else:
            self.history.append({"role": role, "content": content})

            self.context_window.append({"role": role, "content": content})       

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
            text = response["choices"][-1]["message"]["content"].strip()

        AmadeusLogger.info(
            "Full response from openAI before only matching function string:"
        )
        AmadeusLogger.info(text)

        # we need to consider better ways to parse functions
        # and save them in a more structured way

        function_codes, _ = search_generated_func(text)

        thought_process = text

        if len(function_codes) >= 1:
            function_code = function_codes[0]
        else:
            function_code = None

        return text, function_code, thought_process
    

class CodeGenerationLLM(LLM):
    """
    Resource management for the behavior analysis part of the system
    """

    def get_system_prompt(self, interface_str, behavior_module_str):
        from amadeusgpt.system_prompts.code_generator import _get_system_prompt
        return _get_system_prompt(interface_str, behavior_module_str)

    def update_system_prompt(self, interface_str, behavior_modules_str):
        self.system_prompt = self.get_system_prompt(interface_str, behavior_modules_str)
        # update both history and context window
        self.update_history("system", self.system_prompt)

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
    

class ExplainerLLM(LLM):
    
    def get_system_prompt(
        self, user_input, thought_process, answer
    ):
        from amadeusgpt.system_prompts.explainer import _get_system_prompt
        return _get_system_prompt(
            user_input, thought_process, answer
        )
    def generate_explanation(
            self,
            user_input,
            thought_process,
            answer,
            plots
    ):
        """
        Explain to users how the solution came up
        """

        captions = ''
        if isinstance(plots, list):
            for plot in plots:
                if plot.plot_caption !='':
                    captions+=plot.plot_caption
            if captions!='':
                answer+=captions
                
        messages = [
            {
                "role": "system",
                "content": self.get_system_prompt(
                    user_input,
                    thought_process,
                    answer,
                ),
            }
        ]
        response = self.connect_gpt(messages, max_tokens=500)["choices"][0]["message"][
            "content"
        ]
        return response


class RephraserLLM(LLM):
    def __init__(self, config):
        super().__init__(config)
        self.correction_k = 1

    def get_system_prompt(self):
        from amadeusgpt.system_prompts.rephraser import _get_system_prompt
        return _get_system_prompt()

    @classmethod
    def generate_iid(self, user_input):
        """
        Try to ask the question like asked in API docs
        """
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": f"{user_input}"},
        ]
        # use gpt 3.5 for rephraser to avoid using gpt-4 too fast to run into rate limit
        ret = self.connect_gpt(messages, max_tokens=200, gpt_model="gpt-3.5-turbo")
        if ret is None:
            return None
        response = ret["choices"][0]["message"]["content"].strip()
        return response

    def generate_equivalent(self, user_input, k=5):
        """
        Generate equivalent messages based on base questions
        """

        messages = [
            {
                "role": "system",
                "content": f"Generate equivalent sentences with proper English using {k} different expression. \
                                                    Every generated sentence starts with <start> and end with </end>",
            },
            {"role": "user", "content": f"{user_input}"},
        ]

        AmadeusLogger.info("user input:")
        AmadeusLogger.info(user_input)
        response = self.connect_gpt(messages)["choices"][0]["message"]["content"]
        AmadeusLogger.info("response:")
        AmadeusLogger.info(response)
        pattern = r"<start>\s*(.*?)\s*<\/end>"
        # parse the equivalent sentences
        matches = re.findall(pattern, response)
        sentences = [match.strip() for match in matches]

        return sentences
    
class SelfDebugLLM(LLM):

    def get_system_prompt(self):
        from amadeusgpt.system_prompts.self_debug import _get_system_prompt
        return _get_system_prompt()


    def debug_and_retry(self, 
                        user_query = "",
                        error_code = "",
                        diagnosis = "",
                        api_docs = "", 
                        error_message = ""):
      

        system_prompt = self.get_system_prompt()

        user_prompt = f"""
        <user_query> {user_query} </user_query>
        <error_code> {error_code} </error_code>
        <api_docs> {api_docs} </api_docs>
        <error_message> {error_message} </error_message>
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
 
        response = self.connect_gpt(messages, max_tokens=500)
        text, function_codes, thought_process = self.parse_openai_response(response)
        return text, function_codes, thought_process
