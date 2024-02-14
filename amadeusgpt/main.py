import copy
import glob
import inspect
import os
import pickle
import re
import subprocess
import sys
import time
import traceback
import warnings
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
from typing import Dict, List, Tuple
import amadeusgpt
from amadeusgpt.datamodel import AmadeusAnswer
from amadeusgpt.utils import parse_error_message_from_python


##########
# all these are providing the customized classes for the code execution
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
from amadeusgpt.modules.implementation import *

##########


from amadeusgpt.utils import *
from amadeusgpt.module_matching import match_module
import streamlit as st
from amadeusgpt.logger import AmadeusLogger

############
# all these are for safe execution of generated code
from amadeusgpt.amadeus_security import check_code
from RestrictedPython import (
    compile_restricted,
    safe_globals,
    safe_builtins,
    utility_builtins,
    limited_builtins,
)
from RestrictedPython.Guards import (
    guarded_unpack_sequence,
    guarded_iter_unpack_sequence,
)

############


from amadeusgpt.datamodel.amadeus_answer import AmadeusAnswer

from amadeusgpt.brains import (
    BaseBrain,
    DiagnosisBrain,
    ExplainerBrain,
    CodeGenerationBrain,
    SelfDebugBrain,
    RephraserBrain,
)

warnings.filterwarnings("ignore")


class AMADEUS:
    behavior_modules_str = ""
    openai.organization = ""
    ### fields here require persisetence storage for restoring the state of Amadeus
    code_generator_brain = CodeGenerationBrain
    explainer_brain = ExplainerBrain
    self_debug_brain = SelfDebugBrain
    rephraser_brain = RephraserBrain
    diagnosis_brain = DiagnosisBrain
    behavior_analysis = AnimalBehaviorAnalysis
    state_list = [
        code_generator_brain,
        explainer_brain,
        self_debug_brain,
        rephraser_brain,
        diagnosis_brain,
        behavior_analysis,
    ]
    ###
    reflection = True
    log = False
    enforce_prompt = "#"
    code_generator_brain.enforce_prompt = ""
    usage = 0
    behavior_modules_in_context = True
    # load the integration modules to context
    smart_loading = True
    # number of topk integration modules to load
    load_module_top_k = 3
    module_threshold = 0.7
    context_window_dict = {}
    plot = False
    use_rephraser = True
    cache_objects = True

    root = os.path.dirname(os.path.realpath(__file__))
    interface_path = os.path.join(root, "interface.txt")
    with open(interface_path, "r") as f:
        interface_str = f.readlines()
    temp = []
    for line in interface_str:
        temp.append(f"{line.rstrip()}")
    interface_str = "\n".join(temp)
    interface_str = interface_str

    @classmethod
    def release_cache_objects(cls):
        AnimalBehaviorAnalysis.release_cache_objects()

    @classmethod
    def load_module_smartly(cls, user_input):
        # TODO: need to improve the module matching by vector database
        sorted_query_results = match_module(user_input)
        if len(sorted_query_results) == 0:
            return None
        # query result sorted by most relevant module text
        modules = []
        for i in range(cls.load_module_top_k):
            query_result = sorted_query_results[i]
            query_module = query_result[0]
            query_score = query_result[1][0][0]

            if query_score > cls.module_threshold:
                modules.append(query_module)
                # parse the query result by loading active loading
                module_path = os.sep.join(query_module.split(os.sep)[-2:]).replace(
                    ".py", ""
                )
                # print(f"loading {module_path} for relevant score {query_score}")
                AmadeusLogger.log(
                    f"loading {module_path} for relevant score {query_score}", level=1
                )
                cls.load_behavior_modules(f"load module {module_path}")
            else:
                AmadeusLogger.info(
                    f"{query_module} has low similarity score of {query_score}"
                )

    @classmethod
    def magic_command(cls, user_input):
        user_input = user_input.replace("%", "")
        command_list = user_input.split()
        result = subprocess.run(command_list, stdout=subprocess.PIPE)
        AmadeusLogger.info(result.stdout.decode("utf-8"))

    @classmethod
    def save_state(cls, output_path = 'soul.pickle'):
        # save the class attributes of all classes that are under state_list.
        def get_class_variables(_class):
            return {
                k: v
                for k, v in vars(AnimalBehaviorAnalysis).items()
                if not callable(v)
                and not k.startswith("__")
                and not isinstance(v, classmethod)
            }

        state = {k.__name__: get_class_variables(k) for k in cls.state_list}

        with open(output_path, "wb") as f:
            pickle.dump(state, f)
        AmadeusLogger.info(f"memory saved to {output_path}")

    @classmethod
    def load_state(cls, ckpt_path = 'soul.pickle'):
        # load the class variables into 3 class
        memory_filename = ckpt_path
        AmadeusLogger.info(f"loading memory from {memory_filename}")
        with open(memory_filename, "rb") as f:
            state = pickle.load(f)

        def get_class_by_name(name):
            return globals()[name]

        for class_name, class_states in state.items():
            _cls = get_class_by_name(class_name)
            for k, v in class_states.items():
                setattr(_cls, k, v)

    @classmethod
    def _search_missing_symbols_in_context_window(cls, symbols):
        # if the symbol is either defined or retrieved previously, then it is not missing.

        context_window_text = "".join(
            [e["content"] for e in cls.behavior_brain.context_window[1:]]
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

    @classmethod
    def process_write_symbol(cls, user_input):
        """
        Using regular expression to process write symbol
        """
        pattern = r"<\|(.*?)\|>"
        matches = re.findall(pattern, user_input)
        if len(matches) == 0:
            return
        # it is ok to have two write symbols in one sentence. But we only look at the first one. We do want to warn users in that case
        symbol_name = matches[0]
        cls.code_generator_brain.short_term_memory.append(symbol_name)

    @classmethod
    def process_read_symbol(cls, user_input):
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
        missing_symbols = cls._search_missing_symbols_in_context_window(matches)
        for symbol_name in missing_symbols:
            if symbol_name in cls.code_generator_brain.long_term_memory:
                memory_replay += cls.code_generator_brain.long_term_memory[symbol_name]
        memory_replay = memory_replay.replace(cls.enforce_prompt, "").strip()

        return memory_replay

    @classmethod
    def process_symbol(cls, user_input):
        """
        <|symbol_name|> is the writing access to the symbol
            for writing the symbol, add task program function already does it
        <symbol_name> is the retrieving access to the symbol from the long term memory
            for retrieving, searching the symbol name in the context window and task program table
        """
        cls.process_write_symbol(user_input)
        memory_replay = cls.process_read_symbol(user_input)
        if len(memory_replay) > 0:
            ret = f"#{memory_replay}.\n"
            ret += f"# Only use the previous line as a context and focus on instruction of this line: {user_input}.\n"
        else:
            ret = f"#{user_input}\n"
        # ret += cls.enforce_prompt
        return ret

    @classmethod
    def print_task_programs(cls):
        for k, v in AnimalBehaviorAnalysis.task_programs.items():
            AmadeusLogger.info(k)
            AmadeusLogger.info(v)

    @classmethod
    def get_task_programs(cls):
        return AnimalBehaviorAnalysis.task_programs

    @classmethod
    def export_function_code(cls, query, code, filename):
        temp = {"query": query, "code": code}
        with open(filename, "w") as f:
            json.dump(temp, f, indent=4)

    @classmethod
    def chat(
        cls, rephrased_user_msg, original_user_msg, functions=None, code_output=""
    ):
        """
        1. update system prompt of code generation brain, because of dynamic integration module loading
        2. manage memory
        3. call code generator
        4. execute code from code generator
        5. collect function outputs and do further processing
        """

        cls.code_generator_brain.update_system_prompt(
            cls.interface_str, cls.behavior_modules_str
        )
        cls.code_generator_brain.update_history("user", rephrased_user_msg)
        
        response = cls.code_generator_brain.connect_gpt(
            cls.code_generator_brain.context_window, max_tokens=700, functions=functions
        )

        # parse response gives the text and function codes
        (
            text,
            function_code,
            thought_process,
        ) = cls.code_generator_brain.parse_openai_response(response)        

        # write down the task program for offline processing
        with open("temp_for_debug.json", "w") as f:
            out = {'function_code': function_code,
                   'query': rephrased_user_msg}
            json.dump(out, f, indent=4)
            
        # handle_function_codes gives the answer with function outputs   
        amadeus_answer = cls.core_loop(
            rephrased_user_msg, text, function_code, thought_process
        )
        # export the generated function to code_output
        if amadeus_answer.function_code and code_output != "":
            cls.export_function_code(
                original_user_msg, amadeus_answer.function_code, code_output
            )


        # Could be used for in context feedback learning. Costly
        if amadeus_answer.has_error:
            cls.code_generator_brain.context_window[-1][
                "content"
            ] += "\n While executing the code above, there was error so it is not correct answer\n"

        
        # if there is an error or the function code is empty, we want to make sure we prevent ChatGPT to learn to output nothing from few-shot learning            
        #elif amadeus_answer.has_error:
        #    cls.code_generator_brain.context_window.pop()
        #    cls.code_generator_brain.history.pop()
        
        else:
            # needs to manage memory of Amadeus for context window management and state restore etc.
            # we have it remember user's original question instead of the rephrased one for better
            cls.code_generator_brain.manage_memory(
                original_user_msg, amadeus_answer
            )        
        return amadeus_answer

    # this should become an async function so the user can continue to ask question
    @classmethod
    @timer_decorator
    def execute_python_function(
        cls,
        function_code,
    ):
        # we might register a few helper functions into globals()
        result = None
        exec(function_code, globals())
        if "task_program" not in globals():
            return None

        # TODO: to serialize and support different function arguments
        func_sigs = inspect.signature(task_program)
        if not func_sigs.parameters:
            result = task_program()
        else:
            # TODO: We don't do this anymore. But in the future, Is passing result buffer from each function sustainable?
            result_buffer = AnimalBehaviorAnalysis.result_buffer
            AmadeusLogger.info(f"result_buffer: {result_buffer}")
            if isinstance(result_buffer, tuple):
                AmadeusLogger.info(f"length of result buffer: {len(result_buffer)}")
                result_buffer = list(result_buffer)
                result = task_program(*result_buffer)
            else:
                result = task_program(result_buffer)
        return result

    @classmethod
    def contribute(cls, program_name):
        """
        Deprecated
        Takes the program from the task program registry and write it into contribution folder
        TODO: split the task program into implementation and api
        """        
        
        AmadeusLogger.info(f"contributing {program_name}")
        task_program = AnimalBehaviorAnalysis.task_programs[program_name]
        # removing add_symbol or add_task_program line
        lines = task_program.splitlines()
        lines = [
            line
            for line in lines
            if "add_symbol" not in line and "add_task_program" not in line
        ]
        task_program = "\n".join(lines)
        with open(f"contribution/{program_name}.py", "w") as f:
            f.write(task_program)

    @classmethod
    def update_behavior_modules_str(cls):
        """
        Called during loading behavior modules from disk or when task program is updated
        """
        modules_str = []
        # context_window_dict is where integration modules are stored in current AMADEUS class        
        for name, task_program in cls.context_window_dict.items():
            modules_str.append(task_program)
        modules_str = modules_str[-cls.load_module_top_k :]
        cls.behavior_modules_str = "\n".join(modules_str)
        # behavior modules str is part of system prompt. So we update it
        cls.code_generator_brain.update_system_prompt(
            cls.interface_str, cls.behavior_modules_str
        )

    @classmethod
    def load_behavior_modules(cls, user_input):
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
        cls.context_window_dict.update(context_window_dict)
        cls.update_behavior_modules_str()

    @classmethod
    def handle_pipeline_run(cls, user_input):
        pipeline_list = user_input.split("run pipeline")[1]
        pipeline_list = eval(pipeline_list)
        program_table = AnimalBehaviorAnalysis.task_programs
        res = {}
        task_program_results = AnimalBehaviorAnalysis.task_program_results
        output = None
        for name in pipeline_list:
            program = program_table[name]
            lines = program.splitlines()
            lines = [
                line
                for line in lines
                if "add_symbol" not in line and "add_task_program" not in line
            ]
            program = "\n".join(lines)
            _, func_names = search_external_module_for_context_window(program)

            func_name = func_names[0]
            exec(program, globals())
            func_sigs = eval(f"inspect.signature({func_name})")
            if func_sigs.parameters:
                output = eval(f"{func_name}(output)")
            else:
                output = eval(f"{func_name}()")

            # log the results
            task_program_results[name] = output

        return output

    @classmethod
    def collect_function_result(cls, function_returns, function_code, thought_process):
        """
        Assuming the generated code gives no errors, we need to parse the function results
        function_returns is a list of returns
        """
        # result buffer is only updated if there is a early result from a function call
        # TODO update this feature to fit the current code
        AnimalBehaviorAnalysis.result_buffer = function_returns
        # the result can be nested tuples such as (obj1, (obj2, obj3))
        if isinstance(function_returns, tuple):
            function_returns = flatten_tuple(function_returns)

        amadeus_answer = AmadeusAnswer.from_function_returns(
            function_returns, function_code, thought_process
        )

        #if cls.plot:
        #    plt.show()

        # deduplicate as both events and plot could append plots
        return amadeus_answer

    @classmethod
    def chat_loop(cls, warmups):
        assert isinstance(warmups, list), "has to be a list of questions"
        for warmup in warmups:
            cls.chat_iteration(warmup)
        for i in range(100):
            user_input = input("Ask Amadeus a question: ")
            status = cls.chat_iteration(user_input)
            if status == "quit":
                break

    @classmethod
    def chat_iteration(cls, user_input, code_output="", functions=None, rephrased=[]):
        if len(user_input.strip()) == 0:
            return
        answer = {"str_answer": ""}
        if user_input == "save":
            cls.save_state()
            answer = {"str_answer": "saved the status!"}
        elif user_input == "load":
            cls.load_state()
            answer = {"str_answer": "loaded the status!"}
        elif user_input == "history":
            parsed_history = cls.code_generator_brain.print_history()
            answer = {"str_answer": parsed_history}
        elif user_input == "context":
            parsed_history = cls.code_generator_brain.print_context_window()
            answer = {"str_answer": parsed_history}
        elif user_input == "programs":
            cls.print_task_programs()
        elif user_input.startswith("contrib"):
            program_name = user_input.split(" ")[1].strip()
            cls.contribute(program_name)
        elif user_input.startswith("load module"):
            cls.load_behavior_modules(user_input)
        elif user_input.startswith("%"):
            cls.magic_command(user_input)
        elif user_input.startswith("run pipeline"):
            cls.handle_pipeline_run(user_input)
        elif user_input.startswith("forget"):
            cls.behavior_brain.clean_context_window()
        elif user_input.startswith("quit"):
            return "quit"
        else:
            # assuming there are symbols to process
            # if smart module loading is true, load the module quietly
            original_user_msg = user_input
            # if cls.use_rephraser:
            #     AmadeusLogger.log(f"Before rephrasing: {user_input}")
            #     rephrased_user_msg = cls.rephraser_brain.generate_iid(user_input)
            #     AmadeusLogger.log(f"After rephrasing: {rephrased_user_msg}")
            #     rephrased.append(user_input)

            if cls.smart_loading:
                cls.load_module_smartly(original_user_msg)

            # process symbol inserts stuff in to the original user input
            user_input = cls.process_symbol(copy.deepcopy(original_user_msg))

            # if rephrased_user_msg is not None:
            #     user_input = user_input.replace(original_user_msg, rephrased_user_msg)

            answer = cls.chat(
                user_input,
                original_user_msg,
                code_output=code_output,
                functions=functions,
            )
       
        return answer

    # @classmethod
    # def test_robustness(cls, user_input, num_test=5):
    #     sentences = cls.reflection_brain.generate_equivalent(user_input, k=num_test)
    #     root = os.path.dirname(os.path.realpath(__file__))
    #     AmadeusLogger.info("equivalent sentences:")
    #     AmadeusLogger.info(sentences)
    #     ret = defaultdict(dict)
    #     for sentence in sentences:
    #         rephrased = []
    #         res = cls.chat_iteration(sentence, rephrased)
    #         if "str_answer" in res:
    #             ret[sentence]["str_answer"] = res["str_answer"]
    #         if "function_code" in res:
    #             ret[sentence]["function_code"] = res["function_code"]
    #         if len(rephrased) > 0:
    #             ret[sentence]["rephrased"] = rephrased[0]
    #     return ret

    @classmethod
    def compile_amadeus_answer_when_no_error(
        cls, function_returns, user_query, text, function_code, thought_process
    ):
        """
        if no error (either due to successful code run or no code return),
        this function collects results, text and return a AmadeusAnswer instance
        for frontend to handle
        """

        # reminder to handle corner cases
        if function_returns is not None:
            amadeus_answer = cls.collect_function_result(
                function_returns, function_code, thought_process
            )

            if (
                "streamlit_app" in os.environ
                and st.session_state["enable_explainer"] == "Yes"
            ):
                explanation = cls.explainer_brain.generate_explanation(
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

    @classmethod
    def handle_error_in_function_exec(cls, e, user_query, function_code):
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
        text, function_code, thought_process = cls.self_debug_brain.debug_and_retry(
            user_query = user_query,
            error_code = function_code,
            api_docs = cls.interface_str,
            #diagnosis,
            error_message = parse_error_message_from_python()
        )

        return text, function_code, thought_process

    @classmethod
    def core_loop(cls, user_query, text, function_code, thought_process):
        """
        execute function and collect results, which results 2 cases
        case 1: Able to handle error by retrying
        case 2: Not able to handle error even after retrying
        """

        AmadeusLogger.log("---- function code -----")
        AmadeusLogger.log(function_code)

        task_program_code = None

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
                function_returns = cls.execute_python_function(function_code)
                amadeus_answer = cls.compile_amadeus_answer_when_no_error(
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
                ) = cls.handle_error_in_function_exec(e, user_query, function_code)
                intermediate_error = parse_error_message_from_python()
                
                AmadeusLogger.info("full revised text")
                AmadeusLogger.info(text)
                AmadeusLogger.info(f"revised function code: \n  {function_code}")

        # if not returned by now, the error was not fixed
        amadeus_answer = AmadeusAnswer.from_error_message()

        # clean the state of the debug brain
        return amadeus_answer


if __name__ == "__main__":
    log = True
    behavior_modules_in_context = True
    amadeus = AMADEUS(
        log=log,
        behavior_modules_in_context=behavior_modules_in_context,
    )

    keypoint_file_path = "../experiments/EPM/modelzoo/EPM_1DLC_snapshot-1000.h5"

    # AnimalBehaviorAnalysis.set_sam_info(ckpt_path, model_type)
    AnimalBehaviorAnalysis.set_keypoint_file_path(keypoint_file_path)
    # AnimalBehaviorAnalysis.set_video_file_path(video_file_path)

    for i in range(100):
        user_input = input("Ask Amadeus a question: ")
        amadeusgpt.chat_iteration(user_input)
