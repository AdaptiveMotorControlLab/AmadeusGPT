from distutils import core
from .base import AnalysisObject
import openai
import os
import traceback
import time 
from amadeusgpt.utils import AmadeusLogger
from amadeusgpt.utils import search_generated_func
import re

class LLM(AnalysisObject):
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    def __init__(self, config):
        self.config = config
        self.max_tokens = config.get('max_tokens', 2000)
        #self.gpt_model = config.get('gpt_model', "gpt-4-1106-preview")
        #self.gpt_model = config.get('gpt_model', "gpt-3.5-turbo-0125")
        self.gpt_model = config.get('gpt_model', "gpt-4-turbo-preview")
        self.context_window = []
        self.history = []     

    def whetehr_speak(self):
        """
        Handcrafted rules to decide whether to speak
        1) If there is a error in the current chat channel        
        """
        return False

    def speak(self):
        """
        Speak to the chat channel
        """
        raise NotImplementedError("This method should be implemented in the subclass")

    def connect_gpt(self, messages, **kwargs):
        # if openai version is less than 1        
        return self.connect_gpt_oai_1(messages, **kwargs)

    def connect_gpt_oai_1(self, messages, **kwargs):
        import openai 
        from openai import OpenAI
        if self.config.get('use_streamlit', False):            
            if "OPENAI_API_KEY" in os.environ:
                openai.api_key = os.environ["OPENAI_API_KEY"]            
        else:
            openai.api_key = os.environ["OPENAI_API_KEY"]
        response = None
        # gpt_model is default to be the cls.gpt_model, which can be easily set
        gpt_model = self.gpt_model
        # in streamlit app, "gpt_model" is set by the text box
                
        client = OpenAI()

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

                response = client.chat.completions.create(**json_data)

                LLM.prompt_tokens += response.usage.prompt_tokens
                LLM.completion_tokens += response.usage.completion_tokens
                LLM.total_tokens = LLM.prompt_tokens + LLM.completion_tokens
                print ('current total tokens', LLM.total_tokens)
                #TODO we need to calculate the actual dollar cost
                break
            

            except Exception as e:
                
                error_message = traceback.format_exc()
                print ("error", error_message)               

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
    
    def update_history(self, role, content, replace=False):
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
            text = response.choices[0].message.content.strip()
       
     

        # we need to consider better ways to parse functions
        # and save them in a more structured way
        pattern = r"```python(.*?)```"
        function_code = re.findall(pattern, text, re.DOTALL)[0]

        # create a placeholder   
        thought_process = text.replace(function_code, "<python_code>")
       

        return text, function_code, thought_process
    

class CodeGenerationLLM(LLM):
    """
    Resource management for the behavior analysis part of the system
    """
    def __init__(self, config):
        super().__init__(config)

    def whether_speak(self, chat_channel):
        """
        1) if there is a error from last iteration, don't speak
        """

        error = chat_channel.get_last_message().get("error", None)
        if error is not None:
            return False
        else:
            return True      

    def speak(self, sandbox):
        """
        Speak to the chat channel
        """         
        query = sandbox.get_user_query()
        self.update_system_prompt(sandbox)
        self.update_history("user", query)

        response = self.connect_gpt(self.context_window, max_tokens=700)
        text = response.choices[0].message.content.strip()            

        # we need to consider better ways to parse functions
        # and save them in a more structured way
        pattern = r"```python(.*?)```"
        function_code = re.findall(pattern, text, re.DOTALL)[0]

        # create a placeholder   
        thought_process = text.replace(function_code, "<python_code>")

        sandbox.chat_channel.code_history.append(function_code)
        sandbox.chat_channel.chain_of_thought.append(thought_process)               
       
        return thought_process


    def update_system_prompt(self, sandbox):
        from amadeusgpt.system_prompts.code_generator import _get_system_prompt

        core_api_docs = sandbox.get_core_api_docs()
        helper_functions = sandbox.get_helper_functions()
        task_program_docs = sandbox.get_task_program_docs()
        variables = sandbox.get_variables()

        self.system_prompt = _get_system_prompt(core_api_docs, 
                                                             helper_functions, 
                                                             task_program_docs, 
                                                          variables)
        

        # update both history and context window
        self.update_history("system", self.system_prompt)


class MutationLLM(LLM):
    def __init__(self, config):
        super().__init__(config)

    def whether_speak(self, chat_channel):
        """
        1) if there is a error from last iteration, don't speak
        """

        error = chat_channel.get_last_message().get("error", None)
        if error is not None:
            return False
        else:
            return True     

    def update_system_prompt(self, sandbox):
        from amadeusgpt.system_prompts.mutation import _get_system_prompt
        core_api_docs = sandbox.get_core_api_docs()
        task_program_docs = sandbox.get_task_program_docs()
        self.system_prompt = _get_system_prompt(core_api_docs, task_program_docs)


        # update both history and context window        
        self.update_history("system", self.system_prompt)

    def speak(self, sandbox):
        #TODO maybe we don't need to keep the history
        """
        Speak to the chat channel
        """               
        query = "Now write the function for the new behavior. Make sure your code is within```{Code here}``\n"
        self.update_system_prompt(sandbox)
        self.update_history("user", query, replace = True)
               
        response = self.connect_gpt(self.context_window, max_tokens=2000)                
       
        text = response.choices[0].message.content.strip() 
        sandbox.chat_channel.chain_of_thought.append(response)            
        return text

class BreedLLM(LLM):
    def __init__(self, config):
        super().__init__(config)
    def whether_speak(self, chat_channel):
        """
        1) if there is a error from last iteration, don't speak
        """

        error = chat_channel.get_last_message().get("error", None)
        if error is not None:
            return False
        else:
            return True 
    def update_system_prompt(self, sandbox):
        from amadeusgpt.system_prompts.breed import _get_system_prompt

        behavior1_docs, behavior2_docs, composition_type = sandbox.get_breed_info()        

        self.system_prompt = _get_system_prompt(behavior1_docs, behavior2_docs, composition_type)

        # update both history and context window        
       
        self.update_history("system", self.system_prompt)  
    def speak(self, sandbox):
        #TODO maybe we don't need to keep the history
        """
        Speak to the chat channel
        """ 
        query = "Now write the template function. Make sure your answer is concise and don't mention anything about filtering such as smooth_window or min_window\n"
        self.update_system_prompt(sandbox)
        self.update_history("user", query, replace  = True)

        response = self.connect_gpt(self.context_window, max_tokens=400)                
        text = response.choices[0].message.content.strip() 
        sandbox.chat_channel.chain_of_thought.append(response)

        return text              
    

class DiagnosisLLM(LLM):
    """
    Resource management for testing and error handling
    """

    def whether_speak(self, chat_channel):
        """
        Handcrafted rules to decide whether to speak
        1) If there is a error in the current chat channel        
        """
        if chat_channel.get_last_message() is None:
            return False
        else:
            error = chat_channel.get_last_message().get("error", None)

            return error is None

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

    def get_system_prompt(self):
        from amadeusgpt.system_prompts.self_debug import _get_system_prompt
        return _get_system_prompt()

    def whetehr_speak(self, chat_channel):
        if chat_channel.get_last_message() is None:
            return False
        else:
            error = chat_channel.get_last_message().get("error", None)

            return error is None

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

