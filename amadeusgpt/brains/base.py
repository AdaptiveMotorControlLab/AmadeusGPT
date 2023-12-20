from amadeusgpt.implementation import (
    AnimalBehaviorAnalysis,
    Event,
    EventList,
    AnimalEvent,
    AnimalAnimalEvent,
    Object,
    Orientation,
    Database,
)
import time
import openai
import os
from amadeusgpt.utils import *
from amadeusgpt.logger import AmadeusLogger
import streamlit as st
import os
import traceback


class classproperty(property):
    # use this class to support class attribute getter and setter that are similar to those of instance attribute
    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)

    def __set__(self, owner_cls, value):
        return self.fset(owner_cls, value)


class BaseBrain:
    @classproperty
    def get_system_prompt(cls):
        raise NotImplementedError()

    @classproperty
    def context_window(cls):
        if not Database.exist(cls.__name__, "context_window"):
            Database.add(cls.__name__, "context_window", [])
        return Database.get(cls.__name__, "context_window")

    @classproperty
    def max_tokens(cls):
        if not Database.exist(cls.__name__, "max_tokens"):
            Database.add(cls.__name__, "max_tokens", 600)
        return Database.get(cls.__name__, "max_tokens")

    @classproperty
    def history(cls):
        if not Database.exist(cls.__name__, "history"):
            Database.add(cls.__name__, "history", [])
        return Database.get(cls.__name__, "history")

    @classproperty
    def usage(cls):
        if not Database.exist(cls.__name__, "usage"):
            Database.add(cls.__name__, "usage", {})
        return Database.get(cls.__name__, "usage")

    @classproperty
    def short_term_memory(cls):
        if not Database.exist(cls.__name__, "short_term_memory"):
            Database.add(cls.__name__, "short_term_memory", [])
        return Database.get(cls.__name__, "short_term_memory")

    @classproperty
    def long_term_memory(cls):
        if not Database.exist(cls.__name__, "long_term_memory"):
            Database.add(cls.__name__, "long_term_memory", {})
        return Database.get(cls.__name__, "long_term_memory")

    @classproperty
    def accumulated_tokens(cls):
        if not Database.exist(cls.__name__, "accumulated_tokens"):
            Database.add(cls.__name__, "accumulated_tokens", 0)
        return Database.get(cls.__name__, "accumulated_tokens")

    @classproperty
    def gpt_model(cls):
        if not Database.exist(cls.__name__, "gpt_model"):
            Database.add(cls.__name__, "gpt_model", "gpt-3.5-turbo-16k-0613")
        return Database.get(cls.__name__, "gpt_model")

    @classmethod
    @timer_decorator
    def connect_gpt(cls, messages, functions=None, function_call=None, **kwargs):
        # in cloud, we only store api key on the session and let streamlit handle security
        if "streamlit_app" in os.environ:
            if "OPENAI_API_KEY" in os.environ:
                openai.api_key = os.environ["OPENAI_API_KEY"]
            else:
                openai.api_key = st.session_state.get("OPENAI_API_KEY", "")
        else:
            openai.api_key = os.environ["OPENAI_API_KEY"]

        openai.Model.list()
        response = None

        # gpt_model is default to be the cls.gpt_model, which can be easily set
        gpt_model = cls.gpt_model
        # in streamlit app, "gpt_model" is set by the text box

        if "streamlit_app" in os.environ:
            if "gpt_model" in st.session_state:
                gpt_model = st.session_state["gpt_model"]

        # allow kwargs to override gpt_model. This is to make sure child class of BaseBrain can use different option
        if "gpt_model" in kwargs:
            gpt_model = kwargs["gpt_model"]

        AmadeusLogger.info(f"the gpt model that is being used {gpt_model}")
        max_tokens = kwargs.get("max_tokens", cls.max_tokens)
        # the usage was recorded from the last run. However, since we have many LLMs that
        # share the call of this function, we will need to store usage and retrieve them from the database class
        num_retries = 3
        for _ in range(num_retries):
            try:
                json_data = {
                    "model": gpt_model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "stop": None,
                    "top_p": 1,
                    "temperature": 0.0,
                }
                if functions is not None:
                    json_data.update({"functions": functions})
                if function_call is not None:
                    json_data.update({"function_call": function_call})
                response = openai.ChatCompletion.create(**json_data)
                print(response)
                cls.usage = response["usage"]
                break
            except openai.error.RateLimitError:
                if "rate_limit_error" not in st.session_state:
                    st.error(
                        "It appears you are out of funds/free tokens from openAI - please check your account settings"
                    )
                return None

            except Exception as e:
                AmadeusLogger.info("something was wrong in the connect ")
                error_message = traceback.format_exc()
                AmadeusLogger.info(error_message)
                if "streamlit_cloud" in os.environ:
                    AmadeusLogger.store_chats(
                        "connect_errors", str(e) + "\n" + error_message
                    )
                if cls.usage is None:
                    AmadeusLogger.debug("OpenAI server not responding")

                if "This model's maximum context" in error_message:
                    if len(cls.context_window) > 2:
                        AmadeusLogger.info("doing active forgetting")
                        cls.context_window.pop(1)
                        # and the corresponding bot answer
                        cls.context_window.pop(1)

                elif "Rate limit reached" in error_message:
                    print("Hit rate limit. Sleeping for 10 sec")
                    time.sleep(10)

        # Extract the parsed information from the response
        parsed_info = response
        return parsed_info

    @classmethod
    def update_history(cls, role, content):
        if role == "system":
            if len(cls.history) > 0:
                cls.history[0]["content"] = content
                cls.context_window[0]["content"] = content
            else:
                cls.history.append({"role": role, "content": content})

                cls.context_window.append({"role": role, "content": content})
        else:
            cls.history.append({"role": role, "content": content})

            cls.context_window.append({"role": role, "content": content})

    @classmethod
    def manage_memory(cls, user_msg, bot_answer):
        """
        instead of memorizing function str, we only memorize the reference to the answer
        """
        if cls.usage:
            if cls.accumulated_tokens == 0:
                cls.accumulated_tokens += cls.usage["total_tokens"]
            else:
                cls.accumulated_tokens += cls.usage["completion_tokens"]

        forget_mode = False
        if forget_mode:
            # under forget mode, we forget about user's question too
            for chat in cls.context_window:
                if chat["role"] == "user":
                    chat["content"] = ""

        # it's maybe important to keep the answer in the context window. Otherwise we are teaching the model to output empty string
        # need to be very careful as LLMs fewshot learning can wrongly link the answer (even if it is invalid) to the question
        if bot_answer.ndarray:
            cls.update_history("assistant", bot_answer.function_code)
        else:
            
            answer_for_memory = ''
                                    
            if bot_answer.function_code:
                for code in bot_answer.function_code:
                    answer_for_memory += code

            answer_for_memory += '\n' + bot_answer.str_answer

            captions = ''
            if isinstance(bot_answer.plots, list):
                for plot in bot_answer.plots:
                    if plot.plot_caption !='':
                        captions+=plot.plot_caption
                if captions!='':
                    answer_for_memory+=captions            
                    
            cls.update_history("assistant", answer_for_memory)

        # is this used anymore?
        # turn context window memory into long term memory
        if len(cls.short_term_memory) > 0:
            # adding a new symbol or adding a new task program will trigger this
            # producer-consumer pattern. The producer adds a symbol
            # the consumder pops the symbol
            latest_symbol = cls.short_term_memory.pop()
            cls.long_term_memory[latest_symbol] = user_msg
            cls.manage_task_programs(latest_symbol, bot_answer)

    @classmethod
    @timer_decorator
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

    @classmethod
    def clean_context_window(cls):
        while len(cls.context_window) > 1:
            AmadeusLogger.info("cleaning context window")
            cls.context_window.pop()

    @classmethod
    def manage_task_programs(cls, symbol_name, bot_answer):
        """
        Given the symbol is in the short memory, puts function code in task program table
        """

        # if there is valid function code and there is a corresponding task program in the task program table
        if (
            bot_answer.function_code
            and symbol_name in AnimalBehaviorAnalysis.task_programs
        ):
            AnimalBehaviorAnalysis.task_programs[symbol_name] = bot_answer.function_code            

    @classmethod
    def print_history(cls):
        """
        Print the history in a human friendly way. Skip the system prompt that has the code
        """
        history = cls.history
        parsed_history = []
        for chat in history:
            content = chat["content"].strip()
            cur_role = chat["role"].strip()
            AmadeusLogger.info(f"{cur_role}: {content} \n")
            his_dict = {}
            his_dict[cur_role] = content
            parsed_history.append(his_dict)
        return parsed_history

    @classmethod
    def print_context_window(cls):
        history = cls.context_window
        parsed_history = []
        for chat in history:
            content = chat["content"].strip()
            cur_role = chat["role"].strip()
            AmadeusLogger.info(f"{cur_role}: {content} \n")
            his_dict = {}
            his_dict[cur_role] = content
            parsed_history.append(his_dict)
        return parsed_history
