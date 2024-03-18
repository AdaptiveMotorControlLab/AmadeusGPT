import copy
import glob
import inspect
import os
import pickle
import re
import subprocess
import warnings
import json
import openai
import amadeusgpt
from amadeusgpt.middle_end import AmadeusAnswer
from amadeusgpt.utils import parse_error_message_from_python


##########
# all these are providing the customized classes for the code execution
from amadeusgpt.implementation import (
    AnimalBehaviorAnalysis,  
)
from amadeusgpt.modules.implementation import *

##########

from amadeusgpt.utils import *
from amadeusgpt.module_matching import match_module
import streamlit as st
from amadeusgpt.logger import AmadeusLogger
from amadeusgpt.middle_end import AmadeusAnswer

warnings.filterwarnings("ignore")
from amadeusgpt.analysis_objects.llm import CodeGenerationLLM, ExplainerLLM, SelfDebugLLM, RephraserLLM, DiagnosisLLM 



class Mediator:
    """
    Mediator invites all llms to look at the chat channel and update the chat channel
    """
    def __init__(self, chat_channel):
        self.llms = {}
        self.chat_channel = chat_channel
    def register_llm(self, llm):
        self.llms[llm.__class__.__name__] = llm
    def update_chat_channel(self):
        prev_N = len(self.chat_channel)
        for llm_name, llm in self.llms.items():
            if llm.whether_speak(self.chat_channel):
                llm.speak(self.chat_channel)
        for attr in self.chat_channel.__dict__.values():
            if len(attr) != prev_N:
                attr.append('') # fill in the empty string


class ChatChannel:
    """
    All llms are supposed to check and update the chat channel
    """

    def __init__(self):
        self.reflection = []
        self.error_signal = []
        self.code_history = []
        self.chain_of_thought = []
        self.analysis_result = []
        self.user_query = []
        self.error_diagnosis = []
        self.task_program_description = []
    def __len__(self):
        # assert all attributes have the same length first
        N = len(self.reflection)
        for attr in self.__dict__.values():
            assert len(attr) == N, f"length of {attr} is not the same as the rest"
        return N




class AMADEUS:
    def __init__(self, config: Dict[str, Any]):
        self.config = config    
        # functionally different llms
        self.code_generator_llm = CodeGenerationLLM()
        self.explainer_llm = ExplainerLLM()
        self.self_debug_llm = SelfDebugLLM()
        self.rephraser_llm = RephraserLLM()
        self.diagnosis_llm = DiagnosisLLM()
        self.chat_chnanel = ChatChannel()
        self.mediator = Mediator(self.chat_channel)      
        ### fields that decide the behavior of the application
        self.use_explainer = False
        self.use_self_debug = False
        self.use_rephraser = False
        self.use_diagnosis = False        
        self.behavior_modules_in_context = True
        self.smart_loading = False        
        self.load_module_top_k = 3
        self.module_threshold = 0.7
        self.enforce_prompt = "#"
        self.code_generator_llm.enforce_prompt = ""  
        ## register the llm to the mediator
        self.mediator.register_llm(self.code_generator_llm)
        if self.use_explainer:
            self.mediator.register_llm(self.explainer_llm)
        if self.use_self_debug:
            self.mediator.register_llm(self.self_debug_llm)
        if self.use_rephraser:
            self.mediator.register_llm(self.rephraser_llm)
        if self.use_diagnosis:
            self.mediator.register_llm(self.diagnosis_llm)
        ### fileds that serve as important storage    
        self.context_window_dict = {}
        self.behavior_modules_str = ""
        ####            
  
    def load_module_smartly(self, user_input):
        # TODO: need to improve the module matching by vector database
        sorted_query_results = match_module(user_input)
        if len(sorted_query_results) == 0:
            return None
        # query result sorted by most relevant module text
        modules = []
        for i in range(self.load_module_top_k):
            query_result = sorted_query_results[i]
            query_module = query_result[0]
            query_score = query_result[1][0][0]

            if query_score > self.module_threshold:
                modules.append(query_module)
                # parse the query result by loading active loading
                module_path = os.sep.join(query_module.split(os.sep)[-2:]).replace(
                    ".py", ""
                )
                # print(f"loading {module_path} for relevant score {query_score}")
                AmadeusLogger.log(
                    f"loading {module_path} for relevant score {query_score}", level=1
                )
                self.load_behavior_modules(f"load module {module_path}")
            else:
                AmadeusLogger.info(
                    f"{query_module} has low similarity score of {query_score}"
                )
    
    def magic_command(self, user_input):
        user_input = user_input.replace("%", "")
        command_list = user_input.split()
        result = subprocess.run(command_list, stdout=subprocess.PIPE)
        AmadeusLogger.info(result.stdout.decode("utf-8"))
   
    def _search_missing_symbols_in_context_window(self, symbols):
        # if the symbol is either defined or retrieved previously, then it is not missing.

        context_window_text = "".join(
            [e["content"] for e in self.behavior_brain.context_window[1:]]
        )
        ret = []
        for symbol in symbols:
            # search for the symbol in context_window_text
            pattern = rf"<({symbol})>"
            read_match = re.findall(pattern, context_window_text)
            pattern = rf"<\|({symbol})\|>"
            write_match = re.findall(pattern, context_window_text)
            if len(read_match) > 0 or len(write_match) > 0:
                pass
            else:
                ret.append(symbol)
        return ret

    def process_write_symbol(self, user_input):
        """
        Using regular expression to process write symbol
        """
        pattern = r"<\|(.*?)\|>"
        matches = re.findall(pattern, user_input)
        if len(matches) == 0:
            return
        # it is ok to have two write symbols in one sentence. But we only look at the first one. We do want to warn users in that case
        symbol_name = matches[0]
        self.code_generator_llm.short_term_memory.append(symbol_name)

    def process_read_symbol(self, user_input):
        """
        Using regular expression to process read symbol
        """

        memory_replay = ""
        pattern = r"<(.*?)>"
        matches = re.findall(pattern, user_input)
        matches = [match for match in matches if "|" not in match]
        assert len(matches) <= 1
        if len(matches) == 0:
            return ""
        missing_symbols = self._search_missing_symbols_in_context_window(matches)
        for symbol_name in missing_symbols:
            if symbol_name in self.code_generator_llm.long_term_memory:
                memory_replay += self.code_generator_llm.long_term_memory[symbol_name]
        memory_replay = memory_replay.replace(self.enforce_prompt, "").strip()

        return memory_replay

    def process_symbol(self, user_input):
        """
        <|symbol_name|> is the writing access to the symbol
            for writing the symbol, add task program function already does it
        <symbol_name> is the retrieving access to the symbol from the long term memory
            for retrieving, searching the symbol name in the context window and task program table
        """
        self.process_write_symbol(user_input)
        memory_replay = self.process_read_symbol(user_input)
        if len(memory_replay) > 0:
            ret = f"#{memory_replay}.\n"
            ret += f"# Only use the previous line as a context and focus on instruction of this line: {user_input}.\n"
        else:
            ret = f"#{user_input}\n"

        return ret
      
    def export_function_code(self, query, code, filename):
        temp = {"query": query, "code": code}
        with open(filename, "w") as f:
            json.dump(temp, f, indent=4)

    def step(self):
        # 0) load helper module from the user input
        # 2) A controller to decide who speaks next
        # 1) round roubin for all llms to take control
        # 2) parallel inference of all llms (how orchestration is done?)
        
        # chat channel
        """
        [[reflection,
         error_signal,
         code_history,
         chain_of_thought,
         analysis_result,
         user_query,
         error_diagnosis,
         task_program_description,         
         ] ...
        ] indexed by the step 
        """

        # a) scientist that only attends to description and outputs of the code
        # b) code generation takes user input or reflection
        # c) diagnosis takes error (if there is error)
        # d) explainer takes the analysis results
        # e) self debug takes reflection and diagnosis and the code generation results

        self.mediator.update_chat_channel()               

    # this should become an async function so the user can continue to ask question  
    def execute_python_function(
        self,
        function_code,
    ):
        """      
        Can execute arbitrary python code
           -> Can reuse an existing task program after looking up the task program library
           -> Can first draft an task program and use it later
           -> Can first load a helper module and use these help modules
        """


        result = None
        exec(function_code, globals())
        if "task_program" not in globals():
            return None

        result = task_program()
   
        return result
   

    def update_behavior_modules_str(self):
        """
        Called during loading behavior modules from disk or when task program is updated
        """
        modules_str = []
        # context_window_dict is where integration modules are stored in current AMADEUS class        
        for name, task_program in self.context_window_dict.items():
            modules_str.append(task_program)
        modules_str = modules_str[-self.load_module_top_k :]
        self.behavior_modules_str = "\n".join(modules_str)
        # behavior modules str is part of system prompt. So we update it
        self.code_generator_llm.update_system_prompt(
            self.interface_str, self.behavior_modules_str
        )

    def load_behavior_modules(self, user_input):
        """
        load modules/api/.. from modules.
        Load them into the task program table
        Put them in the front of the context window.
        But we do not want to accumulate them in the context window.
        """

        path = user_input.split("load module")[1]

        # no more load all as it easily blows up the context window
        # api docs at modules/api/MODULE_CATEGORY/*.py
        contrib_files = []
        path = os.path.join("modules", "api", path.strip())

        amadeus_root = os.path.dirname(os.path.realpath(__file__))
        if os.path.isdir(os.path.join(amadeus_root, path)):
            modules = glob.glob(os.path.join(amadeus_root, path, "*.py"))
            for module in modules:
                contrib_files.append(module)
        elif os.path.isfile(os.path.join(amadeus_root, path + ".py")):
            contrib_files.append(path + ".py")
        else:
            AmadeusLogger.info(f"{path} does not exist in our hub")

        context_window_dict = {}
        for _file in contrib_files:
            with open(os.path.join(amadeus_root, _file), "r") as f:
                text = f.read()
            try:
                # it is very important to note that what is in the task program table is executable. So it only contains the example code part. Also every task program should have its own name
                # Also importantly, what is in the context window should have everything --  this is to make sure GPT can understand context correctly

                (
                    context_func_matches,
                    context_func_names_matches,
                ) = search_external_module_for_context_window(text)
                for name, func_body in zip(
                    context_func_names_matches, context_func_matches
                ):
                    context_window_dict[name] = func_body

                (
                    table_func_matches,
                    table_func_names_matches,
                ) = search_external_module_for_task_program_table(text)

                assert len(context_func_names_matches) == len(
                    context_func_matches
                ), f"context_func_names_matches {len(context_func_names_matches)}, context_func_matches {len(context_func_matches)}"
                assert len(table_func_names_matches) == len(
                    table_func_matches
                ), f"table_func_names_matches {len(table_func_names_matches)}, table_func_matches {len(table_func_matches)}"

                # this is to make sure all python functions are executable
                for table_func_str in table_func_matches:
                    code_object = compile(table_func_str, "<string>", "exec")

                for table_func_name, table_func_body in zip(
                    table_func_names_matches, table_func_matches
                ):
                    AnimalBehaviorAnalysis.task_programs[
                        table_func_name
                    ] = table_func_body

                message = f"module {_file} parsed successfully.\n"

            except SyntaxError as e:
                print(parse_error_message_from_python())
                message = f"{_file} does not contain valid function string.\n"
            # print(message)
            AmadeusLogger.log(message)

        # instead of update the history
        # we add this directly to end of interface str which is later forming system prompt
        self.context_window_dict.update(context_window_dict)
        self.update_behavior_modules_str()

    def collect_function_result(self, function_returns, function_code, thought_process):
        """
        Assuming the generated code gives no errors, we need to parse the function results
        function_returns is a list of returns
        """     
        if isinstance(function_returns, tuple):
            function_returns = flatten_tuple(function_returns)

        amadeus_answer = AmadeusAnswer.from_function_returns(
            function_returns, function_code, thought_process
        )

        #if cls.plot:
        #    plt.show()

        # deduplicate as both events and plot could append plots
        return amadeus_answer
   

    def chat_iteration(self, user_input):
        if len(user_input.strip()) == 0:
            return
        answer = {"str_answer": ""}
        if user_input == "save":
            self.save_state()
            answer = {"str_answer": "saved the status!"}
        elif user_input == "load":
            self.load_state()
            answer = {"str_answer": "loaded the status!"}
        elif user_input == "history":
            parsed_history = self.code_generator_llm.print_history()
            answer = {"str_answer": parsed_history}
        elif user_input == "context":
            parsed_history = self.code_generator_llm.print_context_window()
            answer = {"str_answer": parsed_history}
        elif user_input == "programs":
            self.print_task_programs()
        elif user_input.startswith("contrib"):
            program_name = user_input.split(" ")[1].strip()
            self.contribute(program_name)
        elif user_input.startswith("load module"):
            self.load_behavior_modules(user_input)
        elif user_input.startswith("%"):
            self.magic_command(user_input)
        elif user_input.startswith("run pipeline"):
            self.handle_pipeline_run(user_input)
        elif user_input.startswith("forget"):
            self.behavior_brain.clean_context_window()
        elif user_input.startswith("quit"):
            return "quit"
        else:
            original_user_msg = user_input         
            if self.smart_loading:
                self.load_module_smartly(original_user_msg)

            # process symbol inserts stuff in to the original user input
            user_input = self.process_symbol(copy.deepcopy(original_user_msg))

            # if rephrased_user_msg is not None:
            #     user_input = user_input.replace(original_user_msg, rephrased_user_msg)

            if len(self.chat_chnanel) == 0:
                self.chat_chnanel.user_query.append(user_input)

            self.step()
            
       
        return answer
   
    def compile_amadeus_answer_when_no_error(
        self, function_returns, user_query, text, function_code, thought_process
    ):
        """
        if no error (either due to successful code run or no code return),
        this function collects results, text and return a AmadeusAnswer instance
        for frontend to handle
        """

        # reminder to handle corner cases
        if function_returns is not None:
            amadeus_answer = self.collect_function_result(
                function_returns, function_code, thought_process
            )

            if (
                "streamlit_app" in os.environ
                and st.session_state["enable_explainer"] == "Yes"
            ):
                explanation = self.explainer_llm.generate_explanation(
                    user_query,
                    amadeus_answer.chain_of_thoughts,
                    amadeus_answer.str_answer,
                    amadeus_answer.plots
                )
                amadeus_answer.summary = explanation
                AmadeusLogger.info("Generated explanation from the explainer:")
                AmadeusLogger.info(explanation)
            else:
                amadeus_answer.summary = ""
        else:
            # if gpt apologies or asks for clarification, it will be no error but no function
            amadeus_answer = AmadeusAnswer()
            amadeus_answer.chain_of_thoughts = thought_process
        return amadeus_answer

    def handle_error_in_function_exec(self, e, user_query, function_code):
        """
        Note there are two types of errors in function exec
        type 1: runtime error that is caused by function hallucination
        type 2: illegal actions from malicious code
        1. parse error message from python
        2. log errors
        3. generate diagnosis
        4. trying to fix error using self debug
        5. the output of self debug is new code and history of self debug
        """
        # parse error message from python
        # log errors
        if "streamlit_cloud" in os.environ:
            AmadeusLogger.store_chats(
                "code_generation_errors",
                {
                    "query": user_query,
                    "error_message": str(e) + "\n" + parse_error_message_from_python()
                },
            )

        AmadeusLogger.info(parse_error_message_from_python())
        text, function_code, thought_process = self.self_debug_brain.debug_and_retry(
            user_query = user_query,
            error_code = function_code,
            api_docs = self.interface_str,
            #diagnosis,
            error_message = parse_error_message_from_python()
        )

        return text, function_code, thought_process

    def core_loop(self, user_query, text, function_code, thought_process):
        """
        execute function and collect results, which results 2 cases
        case 1: Able to handle error by retrying
        case 2: Not able to handle error even after retrying
        """

        AmadeusLogger.log("---- function code -----")
        AmadeusLogger.log(function_code)

        # this loop tries to
        # function_code not None means gpt is still trying
        num_retries = 2
        intermediate_error = None
        intermediate_code = None
        for trial_id in range(num_retries):
            if function_code is None:
                amadeus_answer = AmadeusAnswer.from_text_answer(thought_process)
                if intermediate_error:
                    amadeus_answer.error_message = intermediate_error
                if intermediate_code:
                    amadeus_answer.error_function_code = intermediate_code
                return amadeus_answer
                
            try:
                # only the execution function needs to take multiple function string                
                function_returns = self.execute_python_function(function_code)
                amadeus_answer = self.compile_amadeus_answer_when_no_error(
                    function_returns, user_query, text, function_code, thought_process
                )
                if intermediate_error:
                    amadeus_answer.error_message = intermediate_error
                if intermediate_code:
                    amadeus_answer.error_function_code = intermediate_code
                return amadeus_answer
            except Exception as e:
                # Handle the error and redirect the error message to stdout
                intermediate_code = function_code
                (
                    text,
                    function_code,
                    thought_process,
                ) = self.handle_error_in_function_exec(e, user_query, function_code)
                intermediate_error = parse_error_message_from_python()
                
                AmadeusLogger.info("full revised text")
                AmadeusLogger.info(text)
                AmadeusLogger.info(f"revised function code: \n  {function_code}")

        # if not returned by now, the error was not fixed
        amadeus_answer = AmadeusAnswer.from_error_message()

        # clean the state of the debug brain
        return amadeus_answer

